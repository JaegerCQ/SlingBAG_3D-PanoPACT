"""Differentiable continuous Gaussian photoacoustic forward model.

p(s,t) = sum_i P_i/(2*r_si) * (r_si-c*t) * exp(-(r_si-c*t)^2/(2*sigma_i^2)).
This is the waveform in PA-SfM-NC/forward_propagation.py, evaluated directly
at the acquisition times, without its fixed-sigma KDE interpolation.
Coordinates/sigma: metres; dt: seconds; pressure: Gaussian peak amplitude.
"""
from functools import lru_cache
from pathlib import Path
import math
import os
import shutil
import subprocess
import sys

import torch

from .paths import PROJECT_ROOT as ROOT, PACKAGE_ROOT
DISTANCE_EPS_SQUARED = 1e-12
MIN_SIGMA = 1e-8


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load
    build = ROOT / '.build' / 'gaussian'
    build.mkdir(parents=True, exist_ok=True)
    # Keep build artefacts in this independent project, not the original tree.
    os.environ.setdefault('MAX_JOBS', '2')
    # Some hosts put a stale Python Ninja wrapper before the system executable.
    ninja = shutil.which('ninja')
    if ninja is None or subprocess.run([ninja, '--version'], capture_output=True).returncode:
        for candidate in (Path(sys.executable).parent/'ninja', Path('/usr/bin/ninja')):
            if candidate.is_file() and subprocess.run([str(candidate), '--version'], capture_output=True).returncode == 0:
                os.environ['PATH'] = str(candidate.parent) + os.pathsep + os.environ.get('PATH', '')
                break
    return load(
        name='slingbag_continuous_gaussian',
        sources=[str(PACKAGE_ROOT/'csrc'/'gaussian.cpp'), str(PACKAGE_ROOT/'csrc'/'gaussian.cu')],
        build_directory=str(build), extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--fmad=false'], verbose=False,
    )


def _validate(sensors, xyz, pressure, sigma, dt, num_times, sound_speed, cutoff):
    if sensors.ndim != 2 or sensors.shape[1] != 3 or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('sensors and xyz must have shape [N, 3]')
    if pressure.shape != (xyz.shape[0],) or sigma.shape != pressure.shape:
        raise ValueError('pressure and sigma must have shape [num_sources]')
    if sensors.dtype not in (torch.float32, torch.float64):
        raise TypeError('Inputs must be float32 or float64')
    if any(t.device != sensors.device or t.dtype != sensors.dtype for t in (xyz, pressure, sigma)):
        raise ValueError('Inputs must share dtype and device')
    if not isinstance(num_times, int) or isinstance(num_times, bool) or num_times <= 0:
        raise ValueError('num_times must be a positive integer')
    if not all(math.isfinite(float(v)) for v in (dt, sound_speed, cutoff)):
        raise ValueError('Sampling parameters must be finite')
    if dt <= 0 or sound_speed <= 0 or cutoff < 0:
        raise ValueError('dt/sound_speed must be positive; cutoff must be nonnegative')


def gaussian_reference(sensors, xyz, pressure, sigma, dt=25e-9, num_times=4096,
                       sound_speed=1500.0, cutoff=6.0, source_chunk=128):
    """Autograd reference (float64 supported); cutoff=0 disables truncation.

    This dense, chunked implementation is for validation and small CPU jobs.
    The cutoff mask has zero derivative; away from its boundary the backward
    is the derivative of exactly the waveform evaluated by the forward.
    """
    _validate(sensors, xyz, pressure, sigma, dt, num_times, sound_speed, cutoff)
    if source_chunk < 1:
        raise ValueError('source_chunk must be positive')
    ns, n = sensors.shape[0], xyz.shape[0]
    u = torch.arange(num_times, dtype=xyz.dtype, device=xyz.device)*(dt*sound_speed)
    # Ensure even the empty output retains zero gradients for all source inputs.
    out = torch.zeros((ns, num_times), dtype=xyz.dtype, device=xyz.device)
    out = out + 0*(xyz.sum()+pressure.sum()+sigma.sum()+sensors.sum())
    chunk = min(source_chunk, max(1, 4_000_000//max(1, ns*num_times)))
    for start in range(0, n, chunk):
        x = xyz[start:start+chunk]
        p = pressure[start:start+chunk]
        width = sigma[start:start+chunk].clamp_min(MIN_SIGMA)
        r = ((sensors[:, None, :]-x[None, :, :]).square().sum(-1)
             + DISTANCE_EPS_SQUARED).sqrt()
        d = r[..., None]-u
        z = d/width[None, :, None]
        wave = p[None, :, None]/(2*r[..., None])*d*torch.exp(-0.5*z.square())
        if cutoff > 0:
            wave = torch.where(z.abs() <= cutoff, wave, 0.0)
        out = out + wave.sum(1)
    return out


class _GaussianCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sensors, xyz, pressure, sigma, dt, num_times, speed, cutoff, shards):
        ext = _extension()
        ctx.save_for_backward(sensors, xyz, pressure, sigma)
        ctx.options = (dt, num_times, speed, cutoff)
        return ext.forward(sensors, xyz, pressure, sigma, dt, num_times, speed, cutoff, shards)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        sensors, xyz, pressure, sigma = ctx.saved_tensors
        gx, gp, gs = _extension().backward(
            sensors, xyz, pressure, sigma, grad_output.contiguous(), *ctx.options,
            ctx.needs_input_grad[1],
        )
        return None, gx, gp, gs, None, None, None, None, None


def gaussian_forward(sensors, xyz, pressure, sigma, dt=25e-9, num_times=4096,
                     sound_speed=1500.0, cutoff=6.0, backend='auto', shards=None):
    """Return [num_sensors, num_times], differentiable in xyz, pressure, sigma.

    CUDA uses FP32 accumulation (no mixed precision). Forward atomics can
    vary in their final few bits; backward uses an atomic-free block reduction.
    Use backend='torch' for float64 validation or higher derivatives. Sensors
    are fixed in the CUDA implementation. No device-wide synchronizations.
    """
    _validate(sensors, xyz, pressure, sigma, dt, num_times, sound_speed, cutoff)
    if backend not in ('auto', 'cuda', 'torch'):
        raise ValueError("backend must be 'auto', 'cuda', or 'torch'")
    if backend == 'auto':
        backend = 'cuda' if xyz.is_cuda and xyz.dtype == torch.float32 else 'torch'
    if backend == 'torch':
        return gaussian_reference(sensors, xyz, pressure, sigma, dt, num_times, sound_speed, cutoff)
    if not xyz.is_cuda or xyz.dtype != torch.float32:
        raise ValueError('CUDA backend requires CUDA float32 inputs')
    if sensors.requires_grad:
        raise ValueError("Sensor-coordinate gradients require backend='torch'")
    if shards is None:
        # Bound extra output storage to about 128 MiB; no pairwise intermediates.
        shards = min(32, max(1, xyz.shape[0]//512),
                     max(1, (128*1024*1024)//max(4, 4*sensors.shape[0]*num_times)))
    if not isinstance(shards, int) or not 1 <= shards <= 64:
        raise ValueError('shards must be an integer in [1, 64]')
    return _GaussianCUDA.apply(sensors.contiguous(), xyz.contiguous(), pressure.contiguous(),
                               sigma.clamp_min(MIN_SIGMA).contiguous(), float(dt), num_times,
                               float(sound_speed), float(cutoff), shards)


def simulate(sensor_location, source_location, source_p0, source_dx, dt,
             num_sensors, num_sources, num_times):
    """Legacy notebook signature, returning a flat sensor-major signal."""
    if sensor_location.numel() != num_sensors*3 or source_location.numel() != num_sources*3:
        raise ValueError('Declared sensor/source counts do not match location arrays')
    return gaussian_forward(sensor_location.reshape(num_sensors, 3),
                            source_location.reshape(num_sources, 3),
                            source_p0.reshape(num_sources), source_dx.reshape(num_sources),
                            dt=dt, num_times=num_times).flatten()
