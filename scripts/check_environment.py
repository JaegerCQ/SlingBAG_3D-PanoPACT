#!/usr/bin/env python3
"""Check installation and prepared-dataset consistency without training.

Default: CPU imports and full input-data checks. --cuda additionally runs a tiny
GPU calculation and checks the CUDA compiler. No extension is built here.
Relative paths are resolved from the project root, independent of the shell cwd.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from _script_common import ROOT, configure_caches, geometry_values, input_path, output_path


def add(checks, name, status, detail, remedy=None):
    def json_finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {key: json_finite(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_finite(item) for item in value]
        return value
    detail = json_finite(detail)
    entry = {"check": name, "status": status, "detail": detail}
    if remedy:
        entry["remedy"] = remedy
    checks.append(entry)


def executable_version(name):
    executable = shutil.which(name)
    candidates = [executable] if executable else []
    if name == "ninja":
        candidates.extend((str(Path(sys.executable).parent / "ninja"), "/usr/bin/ninja"))
    if name == "nvcc":
        for prefix in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"):
            if prefix and (Path(prefix) / "bin/nvcc").is_file():
                candidates.append(str(Path(prefix) / "bin/nvcc"))
    for candidate in dict.fromkeys(candidates):
        if not Path(candidate).is_file():
            continue
        try:
            result = subprocess.run([candidate, "--version"], text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=15, check=False)
            if result.returncode == 0:
                return candidate, result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return executable, None


def check_software(checks, use_cuda=False):
    install = f"{sys.executable} -m pip install -r {ROOT / 'requirements.txt'}"
    add(checks, "python", "PASS" if sys.version_info >= (3, 10) else "FAIL",
        {"version": sys.version.split()[0], "executable": sys.executable}, "Use Python 3.10 or newer" if sys.version_info < (3, 10) else None)
    modules = {}
    for name in ("numpy", "scipy", "matplotlib", "torch"):
        try:
            module = importlib.import_module(name)
            modules[name] = module
            add(checks, name, "PASS", {"version": getattr(module, "__version__", "unknown"), "path": getattr(module, "__file__", None)})
        except Exception as error:
            add(checks, name, "FAIL", str(error), install)
    for name in ("ninja", "c++", "nvcc"):
        executable, version = executable_version(name)
        add(checks, name + " executable", "PASS" if version else ("FAIL" if use_cuda else "WARN"),
            {"path": executable, "version": version},
            None if version else (f"{sys.executable} -m pip install ninja; put this environment's bin directory on PATH" if name == "ninja" else
                                  "Install a C++ compiler (for example GCC/g++) and make c++ available on PATH" if name == "c++" else
                                  "Install NVIDIA CUDA Toolkit matching torch.version.cuda; set CUDA_HOME to its root and add CUDA_HOME/bin to PATH. The PyTorch runtime alone does not provide nvcc."))
    try:
        spec = importlib.util.find_spec("triton")
        add(checks, "triton (optional voxelizer)", "PASS" if spec else "WARN", "Available" if spec else "Absent; voxelization uses the slower torch fallback", None if spec else install)
    except (ImportError, ValueError) as error:
        add(checks, "triton (optional voxelizer)", "WARN", str(error))
    torch = modules.get("torch")
    if torch is None:
        return
    add(checks, "torch compiled CUDA", "PASS" if torch.version.cuda else ("FAIL" if use_cuda else "WARN"),
        torch.version.cuda or "CPU-only torch build", None if torch.version.cuda else "Install a CUDA-enabled PyTorch build compatible with your NVIDIA driver; then install this project's requirements")
    if not use_cuda:
        add(checks, "CUDA device", "SKIP", "CPU mode: no device query, allocation, kernel, or extension compilation. Run --cuda for a short device check.")
        return
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        devices = [{"index": i, "name": torch.cuda.get_device_name(i),
                    "compute_capability": list(torch.cuda.get_device_capability(i)),
                    "memory_gib": torch.cuda.get_device_properties(i).total_memory / 2**30}
                   for i in range(torch.cuda.device_count())]
        a = torch.arange(16, device="cuda", dtype=torch.float32)
        actual = float((a * a).sum().cpu())
        if actual != 1240:
            raise RuntimeError(f"CUDA arithmetic returned {actual}, expected 1240")
        add(checks, "CUDA device and arithmetic", "PASS", devices)
    except Exception as error:
        add(checks, "CUDA device and arithmetic", "FAIL", str(error),
            "Check NVIDIA driver/GPU access with nvidia-smi, CUDA_VISIBLE_DEVICES, and that this Python environment has CUDA-enabled torch")
    _, nvcc = executable_version("nvcc")
    if nvcc and torch.version.cuda:
        match = re.search(r"release\s+(\d+)\.(\d+)", nvcc)
        if match:
            toolkit = match.group(1) + "." + match.group(2)
            same_major = match.group(1) == torch.version.cuda.split(".")[0]
            add(checks, "CUDA compiler/runtime compatibility", "PASS" if toolkit == torch.version.cuda else ("WARN" if same_major else "FAIL"),
                {"nvcc": toolkit, "torch_cuda": torch.version.cuda},
                None if toolkit == torch.version.cuda else "Prefer a CUDA Toolkit matching torch.version.cuda. Different major versions cannot build the extension reliably.")
    add(checks, "extension compilation", "SKIP", "Compiler discovery and tiny arithmetic only; the first gaussian_forward CUDA call still compiles the extension into the project .build directory.")


def check_data(directory, checks):
    try:
        import numpy as np
    except ImportError:
        add(checks, "input data", "FAIL", "numpy is unavailable; data checks could not run")
        return
    directory = input_path(directory)
    arrays = {}
    for name in ("signal_original", "signal_normalized", "sensor_locations"):
        path = directory / (name + ".npy")
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype.kind != "f" or array.ndim != 2 or any(v == 0 for v in array.shape):
                raise ValueError(f"Expected a nonempty 2D floating array, got {array.shape}, {array.dtype}")
            if not np.isfinite(array).all():
                raise ValueError("NaN or infinity found")
            arrays[name] = array
            add(checks, name, "PASS" if array.dtype == np.float32 else "FAIL", {"path": str(path), "shape": list(array.shape), "dtype": str(array.dtype), "finite": True},
                None if array.dtype == np.float32 else "Float32 is the supported training/export dtype; convert intentionally rather than silently changing units")
        except (OSError, ValueError, TypeError) as error:
            add(checks, name, "FAIL", str(error), f"Place {name}.npy in {directory}; use original signed RF rows [sensor,time] and native sensor coordinates [sensor,XYZ] in metres")
    try:
        metadata = json.loads((directory / "metadata.json").read_text())
        gain = float(metadata["global_gain"])
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError("global_gain must be finite and positive")
        add(checks, "metadata global_gain", "PASS", gain)
    except (OSError, ValueError, TypeError, KeyError) as error:
        add(checks, "metadata", "FAIL", str(error), "Supply metadata.json with global_gain, sampling, geometry and coordinate units")
        return
    original, normalized, sensors = (arrays.get(k) for k in ("signal_original", "signal_normalized", "sensor_locations"))
    sensor_count = original.shape[0] if original is not None else None
    if original is not None and normalized is not None:
        if original.shape != normalized.shape:
            add(checks, "RF shape correspondence", "FAIL", f"original {original.shape}, normalized {normalized.shape}")
        else:
            expected_energy = residual_energy = original_energy = normalized_energy = 0.
            for start in range(0, sensor_count, 32):
                raw = np.asarray(original[start:start + 32], dtype=np.float64)
                actual = np.asarray(normalized[start:start + 32], dtype=np.float64)
                expected = raw * gain
                original_energy += float(np.square(raw).sum())
                normalized_energy += float(np.square(actual).sum())
                expected_energy += float(np.square(expected).sum())
                residual_energy += float(np.square(actual - expected).sum())
            relative = math.sqrt(residual_energy / expected_energy) if expected_energy else (0. if residual_energy == 0 else math.inf)
            add(checks, "global_gain relationship", "PASS" if relative <= 1e-6 else "FAIL",
                {"rule": "signal_normalized = signal_original * global_gain", "full_array_relative_l2": relative,
                 "original_rms": math.sqrt(original_energy / original.size)},
                None if relative <= 1e-6 else "Regenerate signal_normalized from original * global_gain; do not take abs, apply per-channel scaling, or divide by 50")
            if not original_energy:
                add(checks, "RF signal energy", "FAIL", "All-zero signal cannot constrain a reconstruction")
            actual_original_rms = math.sqrt(original_energy / original.size)
            actual_normalized_rms = math.sqrt(normalized_energy / normalized.size)
            original_key = "processed_original" if "processed_original" in metadata else "raw"
            for key, array in ((original_key, original), ("normalized", normalized)):
                if "shape" in metadata.get(key, {}):
                    ok = tuple(metadata[key]["shape"]) == array.shape
                    add(checks, key + " metadata shape", "PASS" if ok else "FAIL", {"metadata": metadata[key]["shape"], "actual": list(array.shape)})
                if "rms" in metadata.get(key, {}):
                    try:
                        recorded_rms = float(metadata[key]["rms"])
                        actual_rms = actual_original_rms if key == original_key else actual_normalized_rms
                        ok = math.isfinite(recorded_rms) and math.isclose(recorded_rms, actual_rms, rel_tol=1e-6, abs_tol=1e-12)
                        add(checks, key + " metadata RMS", "PASS" if ok else "FAIL", {"metadata_rms": recorded_rms, "actual_rms": actual_rms})
                    except (TypeError, ValueError) as error:
                        add(checks, key + " metadata RMS", "FAIL", str(error))
            if "normalization" in metadata:
                try:
                    target_rms = float(metadata["normalization"]["target_rms"])
                    if not math.isfinite(target_rms) or target_rms <= 0:
                        raise ValueError("normalization.target_rms must be finite and positive")
                    expected_gain = target_rms / actual_original_rms if actual_original_rms > 0 else math.inf
                    ok = math.isclose(gain, expected_gain, rel_tol=1e-6) and math.isclose(actual_normalized_rms, target_rms, rel_tol=1e-6)
                    add(checks, "automatic RMS normalization", "PASS" if ok else "FAIL", {
                        "target_rms": target_rms, "actual_normalized_rms": actual_normalized_rms,
                        "expected_gain_from_this_frame": expected_gain, "recorded_global_gain": gain},
                        None if ok else "Prepare this frame again: global_gain must equal target_rms / RMS(signal_original); do not reuse another frame's gain")
                    for field, actual_rms in (("source_rms", actual_original_rms), ("achieved_rms", actual_normalized_rms)):
                        if field in metadata["normalization"]:
                            recorded = float(metadata["normalization"][field])
                            valid = math.isfinite(recorded) and math.isclose(recorded, actual_rms, rel_tol=1e-6, abs_tol=1e-12)
                            add(checks, "normalization " + field, "PASS" if valid else "FAIL", {"recorded_rms": recorded, "actual_rms": actual_rms})
                except (KeyError, TypeError, ValueError) as error:
                    add(checks, "automatic RMS normalization", "FAIL", str(error))
            elif "rms" in metadata.get("simulation", {}) and "rms" in metadata.get(original_key, {}):
                raw_rms = float(metadata[original_key]["rms"])
                reference_gain = float(metadata["simulation"]["rms"]) / raw_rms if raw_rms > 0 else math.nan
                ok = math.isfinite(reference_gain) and math.isclose(gain, reference_gain, rel_tol=1e-6)
                add(checks, "metadata RMS normalization rule", "PASS" if ok else "FAIL", {"global_gain": gain, "simulation_rms_divided_by_raw_rms": reference_gain})
    try:
        origin, dims, spacing = geometry_values(metadata)
        bounds = metadata["geometry"].get("bounds_m")
        if bounds is not None:
            derived = np.asarray([[origin[i] - spacing[i] / 2, origin[i] + (dims[i] - .5) * spacing[i]] for i in range(3)]).reshape(-1)
            if len(bounds) != 6 or not np.allclose(derived, bounds, rtol=1e-7, atol=1e-9):
                raise ValueError(f"bounds_m disagrees with voxel-centre origin, spacing and dimensions: derived {derived.tolist()}")
        if (dims != (400, 400, 400) or not np.allclose(origin, [-.01995] * 3, rtol=0, atol=1e-10)
                or not np.allclose(spacing, [.0001] * 3, rtol=0, atol=1e-12)):
            raise ValueError("This release requires a fixed ±20 mm field of view: 400³ voxels, 0.1 mm spacing, first centre XYZ = -19.95 mm")
        add(checks, "native physical grid", "PASS", {"origin_xyz_m": origin, "dims_xyz": dims, "spacing_xyz_m": spacing, "volume_axis_order": "ZYX"})
    except (KeyError, ValueError, TypeError) as error:
        add(checks, "native physical grid", "FAIL", str(error), "Specify voxel-centre origin XYZ and positive spacing in metres; exported array axes are ZYX")
    try:
        sampling = metadata["sampling"]
        dt, speed = float(sampling["dt_seconds"]), float(sampling["sound_speed_m_s"])
        frequency = float(sampling.get("sampling_frequency_hz", 1. / dt if dt > 0 else math.nan))
        t0 = float(sampling.get("t0_seconds", 0.))
        samples = int(sampling["samples"])
        if isinstance(sampling["samples"], bool) or samples != sampling["samples"]:
            raise ValueError("sampling.samples must be a positive integer")
        if not all(math.isfinite(v) for v in (dt, frequency, speed, t0)) or dt <= 0 or frequency <= 0 or speed <= 0 or samples <= 0:
            raise ValueError("Sampling dt/frequency/speed/samples must be positive; t0 must be finite")
        if not math.isclose(dt * frequency, 1., rel_tol=1e-10, abs_tol=0.):
            raise ValueError("sampling_frequency_hz and dt_seconds must be reciprocal")
        if original is not None and samples != original.shape[1]:
            raise ValueError("sampling.samples differs from the RF time-axis length")
        if t0 != 0:
            raise ValueError("This packaged forward model requires t0_seconds = 0; resample time origins explicitly first")
        add(checks, "sampling and time units", "PASS", {"dt_seconds": dt, "sampling_frequency_hz": frequency,
            "frequency_source": "metadata" if "sampling_frequency_hz" in sampling else "derived from dt_seconds (legacy metadata)",
            "sound_speed_m_s": speed, "samples": samples, "t0_seconds": t0})
    except (KeyError, ValueError, TypeError) as error:
        add(checks, "sampling and time units", "FAIL", str(error))
    if sensors is not None:
        ok = sensors.shape[1] == 3 and (sensor_count is None or sensors.shape[0] == sensor_count)
        add(checks, "sensor/RF row correspondence", "PASS" if ok else "FAIL", {"sensor_shape": list(sensors.shape), "RF_sensor_count": sensor_count,
            "limitation": "Dimensions are checked; physical row identity relies on acquisition ordering and cannot be inferred from array values"})
        if ok:
            radius = np.linalg.norm(sensors.astype(np.float64), axis=1)
            median = float(np.median(radius))
            plausible = .001 <= median <= 10.
            add(checks, "sensor coordinate scale", "PASS" if plausible else "WARN",
                {"coordinate_unit": "metres as declared", "min_xyz_m": sensors.min(axis=0).tolist(), "max_xyz_m": sensors.max(axis=0).tolist(), "median_distance_from_origin_m": median},
                None if plausible else "Coordinate scale is unusual for this acquisition; verify metres versus millimetres and native XYZ ordering")
    declared = str(metadata.get("coordinate_order", ""))
    declared_metres = any(word in declared.lower() for word in ("metres", "meters", "metre", "meter"))
    valid_unit = metadata["coordinate_unit"] == "m" if "coordinate_unit" in metadata else declared_metres
    add(checks, "coordinate units declaration", "PASS" if valid_unit else "FAIL", {
        "coordinate_unit": metadata.get("coordinate_unit", "metres inferred from legacy coordinate_order" if declared_metres else None),
        "coordinate_order": declared}, None if valid_unit else "Prepared sensor coordinates must be in metres; run prepare_data.py with the correct source coordinate_unit")
    add(checks, "RF units and preprocessing", "PASS", {"raw_units": "Original supplied processed RF amplitude; absolute pressure calibration is not implied",
        "training_units": "Original RF multiplied by global_gain once", "recorded_preprocessing": metadata.get("preprocessing", "unspecified")})
    try:
        with np.load(directory / "split.npz", allow_pickle=False) as split:
            train, validation = split["train"], split["validation"]
        for name, index in (("train", train), ("validation", validation)):
            if index.ndim != 1 or index.dtype.kind not in "iu" or index.size == 0:
                raise ValueError(f"{name} must be a nonempty one-dimensional integer array")
            if np.unique(index).size != index.size or np.any(index < 0):
                raise ValueError(f"{name} contains duplicate or negative indices")
        combined = np.concatenate((train, validation))
        if np.unique(combined).size != combined.size:
            raise ValueError("train and validation overlap")
        if sensor_count is not None and not np.array_equal(np.sort(combined), np.arange(sensor_count)):
            raise ValueError("train and validation must partition all RF sensor indices exactly once")
        add(checks, "sensor split", "PASS", {"train": int(train.size), "validation": int(validation.size), "disjoint": True,
            "interpretation": "Reserved sensors become tuning data when used to select settings; in all-sensor refits both subsets are training data"})
    except (OSError, KeyError, ValueError, TypeError) as error:
        add(checks, "sensor split", "FAIL", str(error), "Provide split.npz with disjoint integer train/validation arrays covering 0..sensor_count-1")


def self_test():
    import numpy as np
    temporary_root = output_path(".cache/check_environment_tests")
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as name:
        folder = Path(name)
        raw = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        np.save(folder / "signal_original.npy", raw)
        np.save(folder / "signal_normalized.npy", raw * .5)
        np.save(folder / "sensor_locations.npy", np.asarray([[.1, 0, 0], [0, .1, 0], [-.1, 0, 0], [0, -.1, 0]], dtype=np.float32))
        np.savez(folder / "split.npz", train=np.asarray([0, 1, 2]), validation=np.asarray([3]))
        metadata = {"global_gain": .5, "coordinate_order": "native XYZ metres; sensor rows follow RF rows", "geometry": {
            "voxel_origin_xyz_m": [-.01995] * 3, "voxel_dims_xyz": [400] * 3, "voxel_spacing_m": .0001,
            "bounds_m": [-.02, .02] * 3}, "sampling": {"dt_seconds": 25e-9, "samples": 4, "sound_speed_m_s": 1500, "t0_seconds": 0}}
        (folder / "metadata.json").write_text(json.dumps(metadata))
        checks = []
        check_data(folder, checks)
        assert not any(c["status"] == "FAIL" for c in checks), checks
        np.save(folder / "signal_normalized.npy", raw)
        checks = []
        check_data(folder, checks)
        assert any(c["check"] == "global_gain relationship" and c["status"] == "FAIL" for c in checks)
        np.savez(folder / "split.npz", train=np.asarray([0, 1, 2]), validation=np.asarray([2]))
        checks = []
        check_data(folder, checks)
        assert any(c["check"] == "sensor split" and c["status"] == "FAIL" for c in checks)
    print("PASS: valid CPU fixture; wrong normalization and overlapping split are rejected")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", "--data", dest="data", default=os.environ.get("SLINGBAG_DATA_DIR", "data/pose000"),
                        help="Prepared dataset containing signal_original.npy, signal_normalized.npy, sensor_locations.npy, metadata.json and split.npz")
    parser.add_argument("--cuda", action="store_true", help="Also inspect CUDA devices and run a 16-element arithmetic check")
    parser.add_argument("--skip-data", action="store_true", help="Check software/toolchain only")
    parser.add_argument("--json", default="outputs/environment_check.json", help="JSON report path inside the project")
    parser.add_argument("--self-test", action="store_true", help="CPU-only synthetic data validation test")
    args = parser.parse_args()
    configure_caches()
    if args.self_test:
        self_test()
        return 0
    checks = []
    check_software(checks, use_cuda=args.cuda)
    if not args.skip_data:
        check_data(args.data, checks)
    success = not any(c["status"] == "FAIL" for c in checks)
    report = {"project_root": str(ROOT), "data_directory": str(input_path(args.data)), "cuda_requested": args.cuda,
              "success": success, "checks": checks, "no_training_performed": True}
    target = output_path(args.json)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for entry in checks:
        print(f"[{entry['status']}] {entry['check']}: {json.dumps(entry['detail'], ensure_ascii=False)}")
        if entry.get("remedy"):
            print("  Fix: " + entry["remedy"])
    print(f"{'PASS' if success else 'FAIL'} — report: {target}")
    return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, ImportError) as error:
        print(f"Environment check failed: {error}", file=sys.stderr)
        sys.exit(1)
