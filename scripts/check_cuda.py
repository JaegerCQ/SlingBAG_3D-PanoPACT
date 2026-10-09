#!/usr/bin/env python3
"""Explicit, small CUDA forward/backward and Gaussian voxelizer verification.

Use --run to enable GPU work. The first run can compile the extension and Triton
voxel kernel. This is a numerical check, not a reconstruction or speed benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slingbag.paths import PROJECT_ROOT, resolve_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Enable the CUDA checks (otherwise only print this help)')
    parser.add_argument('--device', default='cuda:0', help='CUDA device, e.g. cuda:0')
    parser.add_argument('--output', help='Optional JSON report inside outputs/')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return
    destination = resolve_output(args.output) if args.output else None
    if destination is not None and not destination.is_relative_to(PROJECT_ROOT / 'outputs'):
        raise ValueError('CUDA reports must be written inside outputs/')

    import torch
    from slingbag.rapid_forward import gaussian_forward
    from slingbag.voxelize_rapid import gaussian_voxelize

    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('--device must select a CUDA device')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; run the CPU unit tests instead')
    started = time.perf_counter()
    sensors32 = torch.tensor([[7.1, 1.2, 3.4], [-6.3, 2.1, 4.4], [2.7, -7.7, 4.6]],
                             dtype=torch.float32) * 1e-3
    xyz32 = torch.tensor([[.12, .23, -.14], [-.22, .11, .08],
                          [.30, -.25, .12], [-.10, -.05, .22]], dtype=torch.float32) * 1e-3
    pressure32 = torch.tensor([.8, 1.1, .7, 1.4], dtype=torch.float32)
    sigma32 = torch.tensor([.18, .23, .29, .34], dtype=torch.float32) * 1e-3
    inputs32 = (xyz32, pressure32, sigma32)
    generator = torch.Generator(device='cpu').manual_seed(1013)
    upstream32 = torch.randn((3, 512), generator=generator, dtype=torch.float32)
    sensors_gpu = sensors32.to(device)
    comparisons = {}

    def compare(name, actual, expected, *, relative_limit, atol=5e-6, rtol=1e-4):
        actual = actual.detach().cpu().double()
        expected = expected.detach().cpu().double()
        if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
            raise AssertionError(name + ': nonfinite values')
        error = actual - expected
        relative = float(torch.linalg.vector_norm(error) /
                         torch.linalg.vector_norm(expected).clamp_min(1e-30))
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=name)
        if relative > relative_limit:
            raise AssertionError(f'{name}: relative L2 {relative:.3g} exceeds {relative_limit:.3g}')
        comparisons[name] = {'relative_l2_error': relative,
                             'maximum_absolute_error': float(error.abs().max()) if error.numel() else 0.,
                             'relative_l2_limit': relative_limit, 'atol': atol, 'rtol': rtol}

    for cutoff in (0., 6.):
        cpu = tuple(value.double().clone().requires_grad_() for value in inputs32)
        gpu = tuple(value.to(device).clone().requires_grad_() for value in inputs32)
        reference = gaussian_forward(sensors32.double(), *cpu, num_times=512,
                                     cutoff=cutoff, backend='torch')
        actual = gaussian_forward(sensors_gpu, *gpu, num_times=512, cutoff=cutoff, backend='cuda')
        reference_grad = torch.autograd.grad((reference * upstream32.double()).sum(), cpu)
        actual_grad = torch.autograd.grad((actual * upstream32.to(device)).sum(), gpu)
        tag = 'untruncated' if cutoff == 0 else 'cutoff6'
        compare(tag + '/forward', actual, reference, relative_limit=1e-5)
        for name, cuda_gradient, cpu_gradient, scale in zip(
                ('xyz_mm', 'pressure', 'sigma_mm'), actual_grad, reference_grad, (1e-3, 1., 1e-3)):
            compare(tag + '/' + name, cuda_gradient * scale, cpu_gradient * scale,
                    relative_limit=1e-4)
        # The optimized kernel also has a branch that skips position gradients.
        p_only = pressure32.to(device).requires_grad_()
        s_only = sigma32.to(device).requires_grad_()
        frozen = gaussian_forward(sensors_gpu, xyz32.to(device), p_only, s_only,
                                  num_times=512, cutoff=cutoff, backend='cuda')
        p_grad, s_grad = torch.autograd.grad((frozen * upstream32.to(device)).sum(), (p_only, s_only))
        compare(tag + '/frozen_xyz_pressure', p_grad, reference_grad[1], relative_limit=1e-4)
        compare(tag + '/frozen_xyz_sigma_mm', s_grad * 1e-3, reference_grad[2] * 1e-3,
                relative_limit=1e-4)

    empty_inputs = (torch.empty((0, 3), device=device, requires_grad=True),
                    torch.empty(0, device=device, requires_grad=True),
                    torch.empty(0, device=device, requires_grad=True))
    empty_signal = gaussian_forward(sensors_gpu, *empty_inputs, num_times=16, backend='cuda')
    empty_grad = torch.autograd.grad(empty_signal.sum(), empty_inputs)
    if empty_signal.shape != (3, 16) or bool(torch.count_nonzero(empty_signal)) or any(g.numel() for g in empty_grad):
        raise AssertionError('Empty-source forward/backward failed')

    dimensions = (11, 12, 13)
    voxel_options = dict(dims=dimensions, resolution=1e-4, origin=(-.00055, -.0006, -.00065), cutoff=6.)
    reference_volume = gaussian_voxelize(*(value.double() for value in inputs32),
                                         backend='torch', **voxel_options)
    actual_volume = gaussian_voxelize(*(value.to(device) for value in inputs32),
                                      backend='cuda', **voxel_options)
    compare('voxelizer', actual_volume, reference_volume, relative_limit=2e-5, atol=1e-5)
    torch.cuda.synchronize(device)
    report = {
        'status': 'passed', 'device': str(device), 'gpu': torch.cuda.get_device_name(device),
        'compute_capability': list(torch.cuda.get_device_capability(device)),
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'fixture': {'sensors': 3, 'sources': 4, 'time_samples': 512, 'voxel_dims_xyz': list(dimensions)},
        'reference': 'CPU float64 evaluated from the exact float32 CUDA input values; same fixed upstream gradient',
        'gradient_units': 'xyz and sigma derivatives scaled to millimetre parameters; pressure uses peak units',
        'empty_source_forward_backward': 'passed', 'comparisons': comparisons,
        'elapsed_seconds_including_possible_compilation': time.perf_counter() - started,
        'scope': 'Synthetic first-order correctness check, not a training-quality or performance benchmark',
    }
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, AssertionError) as error:
        print(f'CUDA check failed: {error}', file=sys.stderr)
        sys.exit(1)
