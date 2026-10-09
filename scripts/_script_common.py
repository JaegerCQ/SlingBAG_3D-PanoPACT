"""Small path and geometry helpers for standalone CPU scripts."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def input_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def output_path(value):
    path = input_path(value)
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Output must remain inside the project: {ROOT}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def configure_caches():
    for variable, relative in {
        "MPLCONFIGDIR": ".cache/matplotlib", "XDG_CACHE_HOME": ".cache/xdg",
        "TORCH_HOME": ".cache/torch", "TORCH_EXTENSIONS_DIR": ".build",
        "TRITON_CACHE_DIR": ".cache/triton", "CUDA_CACHE_PATH": ".cache/cuda",
    }.items():
        directory = output_path(relative)
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)


def geometry_values(metadata):
    """Return voxel-centre origin XYZ, dimensions XYZ, and spacing XYZ in metres."""
    geometry = metadata.get("geometry", metadata)
    origin = tuple(float(v) for v in geometry["voxel_origin_xyz_m"])
    raw_dims = tuple(geometry["voxel_dims_xyz"])
    if len(raw_dims) != 3 or any(isinstance(v, bool) or int(v) != v or int(v) <= 0 for v in raw_dims):
        raise ValueError("voxel_dims_xyz must contain three positive integers")
    dims = tuple(int(v) for v in raw_dims)
    raw_spacing = geometry["voxel_spacing_m"]
    spacing = (float(raw_spacing),) * 3 if isinstance(raw_spacing, (int, float)) else tuple(float(v) for v in raw_spacing)
    if len(origin) != 3 or len(spacing) != 3:
        raise ValueError("Voxel origin and spacing must have XYZ length three")
    if not all(math.isfinite(v) for v in origin + spacing) or min(spacing) <= 0:
        raise ValueError("Voxel origin must be finite; spacing must be finite and positive")
    return origin, dims, spacing


def read_metadata(path):
    return json.loads(input_path(path).read_text())
