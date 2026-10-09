"""Rasterize continuous Gaussian sources using only their local support."""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from .paths import PROJECT_ROOT as _ROOT
for _variable, _directory in (("TRITON_CACHE_DIR", "triton"),
                              ("TRITON_DUMP_DIR", "triton_dump")):
    _cache = (_ROOT / ".build" / _directory).resolve()
    if not _cache.is_relative_to(_ROOT):
        raise ValueError(f"Cache must remain inside {_ROOT}: {_cache}")
    os.environ[_variable] = str(_cache)

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _voxel_kernel(XYZ, PRESSURE, SIGMA, BOXES, SOURCES, OFFSETS, OUTPUT,
                      RES: tl.constexpr, OX: tl.constexpr, OY: tl.constexpr,
                      OZ: tl.constexpr, NX: tl.constexpr, NY: tl.constexpr,
                      CUTOFF2: tl.constexpr, BLOCK: tl.constexpr):
        job = tl.program_id(0)
        source = tl.load(SOURCES + job)
        start = tl.load(OFFSETS + job)
        xlo = tl.load(BOXES + source * 6)
        ylo = tl.load(BOXES + source * 6 + 1)
        zlo = tl.load(BOXES + source * 6 + 2)
        sx = tl.load(BOXES + source * 6 + 3).to(tl.int64)
        sy = tl.load(BOXES + source * 6 + 4).to(tl.int64)
        sz = tl.load(BOXES + source * 6 + 5).to(tl.int64)
        local = start + tl.arange(0, BLOCK).to(tl.int64)
        valid = local < sx * sy * sz
        ix = xlo + local % sx
        iy = ylo + local // sx % sy
        iz = zlo + local // (sx * sy)
        sigma = tl.load(SIGMA + source)
        dx = (OX + ix.to(tl.float32) * RES - tl.load(XYZ + source * 3)) / sigma
        dy = (OY + iy.to(tl.float32) * RES - tl.load(XYZ + source * 3 + 1)) / sigma
        dz = (OZ + iz.to(tl.float32) * RES - tl.load(XYZ + source * 3 + 2)) / sigma
        radius2 = dx * dx + dy * dy + dz * dz
        value = tl.load(PRESSURE + source) * tl.exp(-0.5 * radius2)
        index = (iz * NY + iy) * NX + ix
        tl.atomic_add(OUTPUT + index, value, mask=valid & (radius2 <= CUTOFF2), sem="relaxed")


def _source_boxes(xyz, sigma, dims, resolution, origin, cutoff):
    centers = xyz.detach().cpu().numpy().astype(np.float64)
    radii = sigma.detach().cpu().numpy().astype(np.float64)[:, None] * cutoff
    # Clip before conversion to integers, including far-outside sources.
    low = np.ceil((centers - radii - origin) / resolution)
    high = np.floor((centers + radii - origin) / resolution) + 1
    low = np.clip(low, 0, dims).astype(np.int64)
    high = np.clip(high, 0, dims).astype(np.int64)
    size = np.maximum(high - low, 0)
    return np.concatenate((low, size), axis=1)


@torch.no_grad()
def gaussian_voxelize(xyz, pressure, sigma, *, dims=(400, 520, 200),
                     resolution=0.0002, origin=(0.0, 0.0, 0.0), cutoff=6.0,
                     backend="auto", chunk_size=262144):
    """Return a [z, y, x] tensor of sum(P exp(-distance² / (2 sigma²))).

    ``dims`` are (nx, ny, nz), coordinates and resolution use metres, and
    ``origin`` gives voxel (0, 0, 0)'s position. Radius equals sigma without
    any normalization. Each source is truncated at ``cutoff * sigma``.
    CUDA uses local bounding boxes with packed Triton work blocks. The torch
    fallback chunks those same local boxes, never source × full-grid arrays.
    This export operation is intentionally not differentiable.
    """
    xyz = torch.as_tensor(xyz)
    if not xyz.is_floating_point():
        xyz = xyz.float()
    if xyz.dtype not in (torch.float32, torch.float64):
        xyz = xyz.float()
    pressure = torch.as_tensor(pressure, device=xyz.device, dtype=xyz.dtype).reshape(-1)
    sigma = torch.as_tensor(sigma, device=xyz.device, dtype=xyz.dtype).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(pressure) != len(xyz) or len(sigma) != len(xyz):
        raise ValueError("expected xyz[N,3], pressure[N], sigma[N]")
    if not bool(torch.isfinite(xyz).all() & torch.isfinite(pressure).all()
                & torch.isfinite(sigma).all() & (sigma > 0).all()):
        raise ValueError("source parameters must be finite and sigma strictly positive")
    dims_array = np.asarray(dims)
    if dims_array.shape != (3,) or not np.isfinite(dims_array).all() or (dims_array < 1).any() \
            or not np.equal(dims_array, np.floor(dims_array)).all() or (dims_array > 2**31 - 1).any():
        raise ValueError("dims must contain three positive integer dimensions")
    dims = tuple(int(v) for v in dims_array)
    origin = np.asarray(origin, dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("origin must contain three finite coordinates")
    if not math.isfinite(resolution) or resolution <= 0 or not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("resolution and cutoff must be finite and positive")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if backend not in {"auto", "cuda", "torch"}:
        raise ValueError("backend must be auto, cuda or torch")
    use_triton = xyz.is_cuda and xyz.dtype == torch.float32 and triton is not None and backend != "torch"
    if backend == "cuda" and not use_triton:
        raise ValueError("cuda backend requires CUDA float32 input tensors and Triton")
    xyz, pressure, sigma = xyz.contiguous(), pressure.contiguous(), sigma.contiguous()
    output = torch.zeros((dims[2], dims[1], dims[0]), dtype=xyz.dtype, device=xyz.device)
    if not len(xyz):
        return output
    boxes = _source_boxes(xyz, sigma, dims, resolution, origin, cutoff)
    volumes = boxes[:, 3:].prod(axis=1)
    if use_triton:
        block = 256
        counts = (volumes + block - 1) // block
        count = int(counts.sum())
        if not count:
            return output
        sources = np.repeat(np.arange(len(xyz), dtype=np.int32), counts)
        prefix = np.cumsum(counts) - counts
        offsets = (np.arange(count, dtype=np.int64) - np.repeat(prefix, counts)) * block
        boxes_t = torch.as_tensor(boxes.astype(np.int32), device=xyz.device)
        sources_t = torch.as_tensor(sources, device=xyz.device)
        offsets_t = torch.as_tensor(offsets, device=xyz.device)
        with torch.cuda.device(xyz.device):
            _voxel_kernel[(count,)](
                xyz, pressure, sigma, boxes_t, sources_t, offsets_t, output,
                resolution, float(origin[0]), float(origin[1]), float(origin[2]),
                dims[0], dims[1], cutoff * cutoff, block,
                enable_fp_fusion=False,
            )
    else:
        flattened = output.view(-1)
        for source, (box, volume) in enumerate(zip(boxes, volumes)):
            if not volume:
                continue
            xlo, ylo, zlo, sx, sy, _ = (int(v) for v in box)
            for start in range(0, int(volume), chunk_size):
                offset = torch.arange(start, min(start + chunk_size, int(volume)), device=xyz.device)
                ix = xlo + offset % sx
                iy = ylo + (offset // sx) % sy
                iz = zlo + offset // (sx * sy)
                dx = (float(origin[0]) + ix.to(xyz.dtype) * resolution - xyz[source, 0]) / sigma[source]
                dy = (float(origin[1]) + iy.to(xyz.dtype) * resolution - xyz[source, 1]) / sigma[source]
                dz = (float(origin[2]) + iz.to(xyz.dtype) * resolution - xyz[source, 2]) / sigma[source]
                radius2 = dx.square() + dy.square() + dz.square()
                value = pressure[source] * torch.exp(-0.5 * radius2)
                value = torch.where(radius2 <= cutoff * cutoff, value, 0)
                indices = (iz * dims[1] + iy) * dims[0] + ix
                flattened[indices] += value
    return output


def voxelize_from_file(ply_path="outputs/final/point_cloud.ply",
                       output="outputs/final/volume_zyx.mat", *, device="auto",
                       backend="auto", dims=(400, 400, 400), resolution=0.0001,
                       origin=(-.01995, -.01995, -.01995), cutoff=6.0,
                       mat_key="volume_zyx"):
    from .io_utils import load_ply, resolve_input, resolve_output
    from .rapid_forward import MIN_SIGMA
    destination = resolve_output(output) if output is not None else None
    if destination is not None and destination.suffix.lower() not in {".mat", ".npy", ".npz"}:
        raise ValueError("output extension must be .mat, .npy or .npz")
    if not mat_key or not mat_key[0].isalpha() or not all(c.isalnum() or c == "_" for c in mat_key):
        raise ValueError("mat_key must start with a letter and contain letters, digits or underscores")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    xyz, pressure, sigma = load_ply(resolve_input(ply_path))
    clamped_count = int(np.count_nonzero(sigma < MIN_SIGMA))
    if clamped_count:
        nonpositive_count = int(np.count_nonzero(sigma <= 0))
        print(f"Clamped {clamped_count} radii to {MIN_SIGMA:g} m "
              f"({nonpositive_count} nonpositive), matching the forward model.")
    sigma = np.maximum(sigma, MIN_SIGMA)
    sources = [torch.as_tensor(a, dtype=torch.float32, device=device)
               for a in (xyz, pressure, sigma)]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    volume = gaussian_voxelize(*sources, dims=dims, resolution=resolution,
                              origin=origin, cutoff=cutoff, backend=backend)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = volume.cpu().numpy()
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix.lower() == ".mat":
            from scipy.io import savemat
            savemat(destination, {mat_key: result})
        elif destination.suffix.lower() == ".npy":
            np.save(destination, result)
        else:
            np.savez(destination, **{mat_key: result}, origin=origin,
                     resolution=resolution, gaussian_cutoff=cutoff)
    print(f"Continuous Gaussian: {len(sources[0])} sources, volume [z,y,x]={result.shape}; "
          f"{elapsed:.3f} s (includes first-call compilation).")
    if destination is not None:
        print(f"Saved {destination}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", default="outputs/final/point_cloud.ply")
    parser.add_argument("--output", default="outputs/final/volume_zyx.mat")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--backend", choices=("auto", "cuda", "torch"), default="auto")
    parser.add_argument("--dims", type=int, nargs=3, default=(400, 400, 400), metavar=("NX", "NY", "NZ"))
    parser.add_argument("--resolution", type=float, default=0.0001, help="voxel spacing in metres")
    parser.add_argument("--origin", type=float, nargs=3, default=(-.01995,)*3, metavar=("X", "Y", "Z"))
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--mat-key", default="volume_zyx")
    args = parser.parse_args(argv)
    voxelize_from_file(args.ply, args.output, device=args.device, backend=args.backend,
                       dims=args.dims, resolution=args.resolution, origin=args.origin,
                       cutoff=args.cutoff, mat_key=args.mat_key)


if __name__ == "__main__":
    main()
