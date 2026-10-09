#!/usr/bin/env python3
"""CPU maximum-intensity projections of a ZYX pressure volume.

Relative paths are resolved from the project root, independent of the shell cwd.
Inputs may be elsewhere; outputs and caches stay inside this project.
"""
from __future__ import annotations

import argparse
import json
import sys

from _script_common import ROOT, configure_caches, geometry_values, input_path, output_path, read_metadata


def load_volume(path):
    import numpy as np
    path = input_path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "volume_zyx" not in archive:
                raise ValueError("NPZ must contain a volume_zyx array")
            volume = archive["volume_zyx"]
    elif path.suffix.lower() == ".npy":
        volume = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        raise ValueError("Expected a .npy volume or .npz containing volume_zyx")
    if volume.ndim != 3 or any(v == 0 for v in volume.shape) or volume.dtype.kind not in "fiu":
        raise ValueError(f"Expected a nonempty real numeric ZYX array, got {volume.shape}, {volume.dtype}")
    for start in range(0, volume.shape[0], 8):
        if not np.isfinite(volume[start:start + 8]).all():
            raise ValueError("Volume contains NaN or infinity")
    return volume


def projections(volume):
    """One full-volume maximum is shared by XY, XZ and YZ; no clipping."""
    minimum, maximum = float(volume.min()), float(volume.max())
    if maximum <= 0 and minimum < 0:
        raise ValueError("Full-maximum display requires a positive maximum or an all-zero volume")
    divisor = maximum if maximum > 0 else 1.0
    mips = [volume.max(axis=axis) / divisor for axis in (0, 1, 2)]
    # Signed inputs remain signed; never silently take abs or clip negatives.
    lower = min(0.0, minimum / divisor)
    return mips, divisor, lower, maximum


def preview(args):
    configure_caches()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    volume = load_volume(args.volume)
    if args.spacing_mm is not None:
        spacing = tuple(v / 1000 for v in args.spacing_mm)
        if len(spacing) == 1:
            spacing *= 3
        if len(spacing) != 3:
            raise ValueError("--spacing-mm accepts one value or three XYZ values")
        origin = tuple(v / 1000 for v in (args.origin_mm or (0., 0., 0.)))
        metadata = {"voxel_origin_xyz_m": origin, "voxel_dims_xyz": volume.shape[::-1], "voxel_spacing_m": spacing}
        geometry_source = "explicit command-line geometry; origin is the first voxel centre"
    else:
        if args.origin_mm is not None:
            raise ValueError("--origin-mm requires --spacing-mm")
        metadata = read_metadata(args.metadata)
        geometry_source = str(input_path(args.metadata))
    origin, dims, spacing = geometry_values(metadata)
    if volume.shape != dims[::-1]:
        raise ValueError(f"Volume ZYX shape {volume.shape} differs from metadata {dims[::-1]}; provide matching --metadata or --spacing-mm [--origin-mm]")
    mips, divisor, lower, maximum = projections(volume)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5), layout="constrained")
    for ax, mip, (horizontal, vertical, name) in zip(axes, mips, ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))):
        extent = []
        for axis in (horizontal, vertical):
            extent.extend(((origin[axis] - spacing[axis] / 2) * 1000,
                           (origin[axis] + (dims[axis] - .5) * spacing[axis]) * 1000))
        artist = ax.imshow(mip, origin="lower", extent=extent, cmap="gray", vmin=lower, vmax=1,
                           interpolation="nearest", aspect="equal")
        ax.set(title=f"{name} MIP", xlabel=f"{'XYZ'[horizontal]} (mm)", ylabel=f"{'XYZ'[vertical]} (mm)")
    fig.colorbar(artist, ax=axes, shrink=.78, label="Amplitude / full-volume maximum" if maximum > 0 else "Amplitude (all-zero volume)")
    fig.suptitle(args.title + "\nLinear scale shared by all views; no clipping or display smoothing", fontsize=12)
    target = output_path(args.output)
    fig.savefig(target, dpi=args.dpi)
    plt.close(fig)
    report = {"volume": str(input_path(args.volume)), "shape_zyx": list(volume.shape), "dtype": str(volume.dtype),
              "axis_order": "ZYX", "geometry_source": geometry_source, "voxel_origin_xyz_m": origin,
              "voxel_dims_xyz": dims, "voxel_spacing_xyz_m": spacing, "full_volume_maximum": maximum,
              "display_divisor": divisor, "display_limits": [lower, 1.], "linear": True,
              "percentile_clipping": False, "display_smoothing": False, "output": str(target)}
    output_path(str(target) + ".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def self_test():
    import numpy as np
    a = np.zeros((3, 4, 5), dtype=np.float32)
    a[1, 2, 3] = 7
    mips, divisor, lower, maximum = projections(a)
    assert [p.shape for p in mips] == [(4, 5), (3, 5), (3, 4)]
    assert [np.unravel_index(p.argmax(), p.shape) for p in mips] == [(2, 3), (1, 3), (1, 2)]
    assert divisor == maximum == 7 and lower == 0 and all(p.max() == 1 for p in mips)
    zero = projections(np.zeros((2, 2, 2), dtype=np.float32))
    assert zero[1] == 1 and zero[3] == 0 and all(np.isfinite(p).all() for p in zero[0])
    print("PASS: ZYX projection orientation, shared full maximum, and zero-volume handling")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--volume", default="examples/pose000/final/volume_zyx.npy", help="NPY ZYX array or NPZ with volume_zyx")
    parser.add_argument("--metadata", default="data/pose000/metadata.json", help="JSON containing geometry in metres")
    parser.add_argument("--output", default="outputs/preview.png", help="Output inside the project; also writes .png.json")
    parser.add_argument("--spacing-mm", nargs="+", type=float, help="Override metadata: one spacing or XYZ spacings in mm")
    parser.add_argument("--origin-mm", nargs=3, type=float, help="XYZ centre of voxel [0,0,0], in mm; default zero with --spacing-mm")
    parser.add_argument("--title", default="SlingBAG pressure volume")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            if args.dpi <= 0:
                raise ValueError("--dpi must be positive")
            preview(args)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(1, f"Preview failed: {error}\nInstall the project requirements with: {sys.executable} -m pip install -r {ROOT / 'requirements.txt'}\n")


if __name__ == "__main__":
    main()
