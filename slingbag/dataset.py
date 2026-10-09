"""CPU preparation and validation of one signed RF frame.

Inputs are read-only. Preparation commits a complete canonical dataset inside
outputs/ and reuses it only when its configuration, sources and file hashes
still match. No filtering, polarity change, truncation or resampling occurs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import os
import shutil
import tempfile

import numpy as np
from scipy.io import loadmat

from .paths import PROJECT_ROOT, dataset_root, display_path, resolve_input


_DATA_FILES = ("signal_original.npy", "signal_normalized.npy", "sensor_locations.npy",
               "split.npz", "split.json", "metadata.json")
_SECTIONS = ("input", "acquisition", "normalization", "sensor_split")
_INPUT_KEYS = {"signal", "sensors", "signal_key", "sensors_key", "signal_layout",
               "sensor_layout", "coordinate_unit", "frame", "signal_delimiter",
               "sensors_delimiter", "signal_skiprows", "sensors_skiprows"}


@dataclass(frozen=True)
class Dataset:
    signal: np.ndarray
    sensors: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    metadata: dict
    directory: Path

    @property
    def gain(self):
        return float(self.metadata["global_gain"])

    @property
    def dt(self):
        return float(self.metadata["sampling"]["dt_seconds"])

    @property
    def sound_speed(self):
        return float(self.metadata["sampling"]["sound_speed_m_s"])

    @property
    def num_times(self):
        return int(self.signal.shape[1])

    @property
    def num_sensors(self):
        return int(self.signal.shape[0])

    @property
    def forward_kwargs(self):
        return {"dt": self.dt, "num_times": self.num_times, "sound_speed": self.sound_speed}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive(value, name):
    if isinstance(value, (bool, str)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite positive number") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _real_matrix(array, name):
    value = np.asarray(array)
    if value.ndim != 2 or not all(value.shape) or value.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a nonempty two-dimensional real numeric array; got {value.shape}, {value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return value


def _float32(array, name):
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(converted).all():
        raise ValueError(f"{name} exceeds the supported float32 range")
    return converted


def _read_matrix(path, *, key=None, delimiter=None, skiprows=0):
    """Read a single numeric matrix; ambiguous containers require a key."""
    path = Path(path)
    suffix = path.suffix.lower()
    skiprows = _integer(skiprows, "skiprows")
    if key is not None and (not isinstance(key, str) or not key):
        raise ValueError("Array keys must be nonempty strings or null")
    if delimiter is not None and (not isinstance(delimiter, str) or len(delimiter) != 1):
        raise ValueError("Text delimiter must be one character or null for whitespace")
    if suffix not in (".txt", ".csv") and (delimiter is not None or skiprows):
        raise ValueError("delimiter/skiprows apply only to TXT and CSV inputs")
    if suffix == ".npy":
        if key is not None:
            raise ValueError("NPY inputs do not use an array key")
        value = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            names = archive.files
            if key is None and len(names) != 1:
                raise ValueError(f"{path.name} contains {names}; specify its array key")
            selected = key if key is not None else names[0]
            if selected not in names:
                raise ValueError(f"Array key {selected!r} is absent from {path.name}; available: {names}")
            value = archive[selected]
    elif suffix == ".mat":
        try:
            archive = loadmat(path, squeeze_me=False, struct_as_record=True)
        except NotImplementedError as error:
            raise ValueError("MAT v7.3/HDF5 is not supported by this reader; export the chosen matrix as NPY or a MATLAB v7-or-earlier MAT file") from error
        names = [name for name in archive if not name.startswith("__")]
        if key is None and len(names) != 1:
            raise ValueError(f"{path.name} contains {names}; specify its array key")
        selected = key if key is not None else names[0]
        if selected not in names:
            raise ValueError(f"Array key {selected!r} is absent from {path.name}; available: {names}")
        value = archive[selected]
    elif suffix in (".txt", ".csv"):
        if key is not None:
            raise ValueError("TXT/CSV inputs do not use an array key")
        if suffix == ".csv" and delimiter is None:
            delimiter = ","
        value = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows, ndmin=2)
    else:
        raise ValueError(f"Unsupported input format {suffix!r}; use NPY, NPZ, TXT, CSV or MAT")
    return _real_matrix(value, path.name)


def _rms(array):
    squared = 0.
    for start in range(0, len(array), 64):
        values = np.asarray(array[start:start + 64], dtype=np.float64)
        squared += float(np.square(values).sum())
    return math.sqrt(squared / array.size)


def _stats(array):
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "min": float(array.min()), "max": float(array.max()),
            "mean": float(np.mean(array, dtype=np.float64)), "rms": _rms(array)}


def _validate_split(train, validation, count):
    values = []
    for name, array in (("train", train), ("validation", validation)):
        array = np.asarray(array)
        if array.ndim != 1 or array.dtype.kind not in "iu" or not len(array):
            raise ValueError(f"{name} indices must be a nonempty one-dimensional integer array")
        if np.any(array < 0) or np.any(array >= count) or len(np.unique(array)) != len(array):
            raise ValueError(f"{name} indices contain duplicates or out-of-range values")
        values.append(np.ascontiguousarray(array, dtype=np.int64))
    if not np.array_equal(np.sort(np.concatenate(values)), np.arange(count)):
        raise ValueError("Train and validation must be disjoint and partition all sensor indices")
    return tuple(values)


def _read_split(path, count):
    if Path(path).suffix.lower() != ".npz":
        raise ValueError("sensor_split.file must be an NPZ with train and validation arrays")
    with np.load(path, allow_pickle=False) as split:
        if not {"train", "validation"} <= set(split.files):
            raise ValueError("Split NPZ needs train and validation arrays")
        return _validate_split(split["train"], split["validation"], count)


def _sampling(metadata, samples):
    try:
        sampling = metadata["sampling"]
        dt = _positive(sampling["dt_seconds"], "sampling.dt_seconds")
        speed = _positive(sampling["sound_speed_m_s"], "sampling.sound_speed_m_s")
        number = _integer(sampling["samples"], "sampling.samples", 1)
        t0 = sampling.get("t0_seconds", 0.)
        if isinstance(t0, (bool, str)) or not np.isscalar(t0) or not math.isfinite(float(t0)) or float(t0) != 0:
            raise ValueError("t0_seconds must be zero; this release does not shift or resample time origins")
        if number != samples:
            raise ValueError("sampling.samples must match the RF time-axis length; no automatic truncation or resampling")
        if "sampling_frequency_hz" in sampling:
            frequency = _positive(sampling["sampling_frequency_hz"], "sampling_frequency_hz")
            if not math.isclose(frequency * dt, 1., rel_tol=1e-10, abs_tol=0):
                raise ValueError("sampling_frequency_hz and dt_seconds disagree")
        return {"dt_seconds": dt, "samples": number, "sound_speed_m_s": speed,
                "t0_seconds": 0., "sampling_frequency_hz": frequency if "sampling_frequency_hz" in sampling else 1. / dt}
    except KeyError as error:
        raise ValueError(f"Missing sampling metadata field: {error}") from error


def _fixed_geometry(metadata):
    expected = {"bounds_m": [-.02, .02] * 3, "voxel_origin_xyz_m": [-.01995] * 3,
                "voxel_dims_xyz": [400] * 3, "voxel_spacing_m": .0001}
    try:
        geometry = metadata["geometry"]
        for name, reference in expected.items():
            values = np.asarray(geometry[name])
            if (values.dtype.kind not in "iuf" or values.shape != np.shape(reference)
                    or not np.isfinite(values).all()
                    or not np.allclose(values, reference, rtol=1e-10, atol=1e-12)):
                raise ValueError(f"Unsupported geometry.{name}; this release requires {reference}")
    except (KeyError, TypeError) as error:
        raise ValueError("Metadata must describe the supported ±20 mm field of view and 400³ grid at 0.1 mm") from error


def load_dataset(directory=None):
    """Load a canonical dataset, including legacy metadata containing only dt."""
    folder = resolve_input(directory) if directory is not None else dataset_root().resolve()
    metadata = json.loads((folder / "metadata.json").read_text())
    arrays = {}
    for name in ("signal_original", "signal_normalized", "sensor_locations"):
        value = _real_matrix(np.load(folder / (name + ".npy"), allow_pickle=False), name)
        if value.dtype != np.float32:
            raise ValueError(f"Canonical {name} must be float32; use prepare_dataset for intentional conversion")
        arrays[name] = np.ascontiguousarray(value)
    original, signal, sensors = (arrays[k] for k in ("signal_original", "signal_normalized", "sensor_locations"))
    if signal.shape != original.shape:
        raise ValueError("Original and normalized RF shapes disagree")
    if sensors.shape != (signal.shape[0], 3):
        raise ValueError("Sensor coordinates must have shape [RF sensor count,3]")
    if signal.shape[0] < 2:
        raise ValueError("At least two sensors are needed for nonempty training and validation splits")
    gain = _positive(metadata.get("global_gain"), "global_gain")
    original_rms, signal_rms = _rms(original), _rms(signal)
    if original_rms == 0 or signal_rms == 0:
        raise ValueError("RF signal has zero energy")
    error, expected_energy = 0., 0.
    for start in range(0, len(signal), 64):
        expected = original[start:start + 64].astype(np.float64) * gain
        actual = signal[start:start + 64].astype(np.float64)
        error += float(np.square(actual - expected).sum())
        expected_energy += float(np.square(expected).sum())
    if not math.isfinite(expected_energy) or expected_energy <= 0 or math.sqrt(error / expected_energy) > 1e-6:
        raise ValueError("Normalized RF must equal original RF times one positive global_gain")
    unit = metadata.get("coordinate_unit")
    if unit is not None:
        if unit != "m":
            raise ValueError("Canonical sensor coordinates must declare coordinate_unit='m'")
    elif not any(word in str(metadata.get("coordinate_order", "")).lower()
                 for word in ("metres", "meters", "metre", "meter")):
        raise ValueError("Metadata must declare sensor coordinate units in metres")
    metadata = dict(metadata)
    metadata["sampling"] = _sampling(metadata, signal.shape[1])
    _fixed_geometry(metadata)
    train, validation = _read_split(folder / "split.npz", len(signal))
    target_rms = metadata.get("normalization", {}).get("target_rms")
    if target_rms is not None and not math.isclose(signal_rms, _positive(target_rms, "normalization.target_rms"), rel_tol=2e-6):
        raise ValueError("Normalized RF RMS disagrees with normalization.target_rms")
    return Dataset(signal=signal, sensors=sensors, train=train, validation=validation,
                   metadata=metadata, directory=folder)


def _preparation_config(config):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")
    try:
        selected = json.loads(json.dumps({name: config[name] for name in _SECTIONS}, allow_nan=False))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Input configuration needs JSON-compatible {_SECTIONS}: {error}") from error
    allowed = {"input": _INPUT_KEYS,
               "acquisition": {"sound_speed_m_s", "sampling_frequency_hz", "samples", "t0_seconds"},
               "normalization": {"target_rms"},
               "sensor_split": {"file", "validation_fraction", "seed"}}
    for name in _SECTIONS:
        if not isinstance(selected[name], dict) or set(selected[name]) - allowed[name]:
            raise ValueError(f"Unknown or invalid keys in {name}; allowed: {sorted(allowed[name])}")
    return selected


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _reuse(folder, fingerprint):
    path = folder / "preparation.json"
    if not path.is_file():
        raise ValueError("Destination already exists without a complete preparation.json; choose a new output directory")
    record = json.loads(path.read_text())
    if record.get("dataset_fingerprint") != fingerprint:
        raise ValueError("Prepared dataset configuration or sources changed; choose a new output directory")
    files = record.get("files", {})
    if set(files) != set(_DATA_FILES):
        raise ValueError("Prepared dataset file manifest is incomplete")
    for name in _DATA_FILES:
        target = folder / name
        if (not target.is_file() or not target.resolve().is_relative_to(folder) or
                target.stat().st_size != files[name].get("size_bytes") or _sha256(target) != files[name].get("sha256")):
            raise ValueError(f"Prepared dataset file changed or is missing: {name}; refusing to overwrite")
    return load_dataset(folder).metadata


def prepare_dataset(config: dict, destination: Path) -> dict:
    """Prepare one frame atomically; reuse only a matching, intact preparation.

    Relative source/destination paths resolve from the repository root. Only
    the four data-related config sections enter the fingerprint, so changing
    reconstruction settings does not invalidate an otherwise identical frame.
    """
    selected = _preparation_config(config)
    folder = resolve_input(destination)
    output_root = PROJECT_ROOT / "outputs"
    if not folder.is_relative_to(output_root) or folder == output_root:
        raise ValueError("Prepared datasets must be written to a subdirectory of this project's outputs/")
    source = selected["input"]
    paths = {}
    for key in ("signal", "sensors"):
        if not isinstance(source.get(key), str) or not source[key]:
            raise ValueError(f"input.{key} must identify an input file")
        paths[key] = resolve_input(source[key])
        if not paths[key].is_file():
            raise ValueError(f"Input file does not exist: {paths[key]}")
    split_options = selected["sensor_split"]
    if split_options.get("file") is not None:
        if not isinstance(split_options["file"], str) or not split_options["file"]:
            raise ValueError("sensor_split.file must be an NPZ path or null")
        paths["split"] = resolve_input(split_options["file"])
        if not paths["split"].is_file():
            raise ValueError(f"Split file does not exist: {paths['split']}")
    sources = {name: {"path": display_path(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
               for name, path in paths.items()}
    identity = {"dataset_config": selected, "source_files": sources}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if folder.exists():
        return _reuse(folder, fingerprint)

    matrices = {}
    for name in ("signal", "sensors"):
        matrices[name] = _read_matrix(paths[name], key=source.get(name + "_key"),
                                     delimiter=source.get(name + "_delimiter"),
                                     skiprows=source.get(name + "_skiprows", 0))
    signal_layout = source.get("signal_layout", "sensor_time")
    sensor_layout = source.get("sensor_layout", "sensor_xyz")
    if signal_layout not in ("sensor_time", "time_sensor"):
        raise ValueError("input.signal_layout must be sensor_time or time_sensor")
    if sensor_layout not in ("sensor_xyz", "xyz_sensor"):
        raise ValueError("input.sensor_layout must be sensor_xyz or xyz_sensor")
    signal = matrices["signal"].T if signal_layout == "time_sensor" else matrices["signal"]
    sensors = matrices["sensors"].T if sensor_layout == "xyz_sensor" else matrices["sensors"]
    unit = source.get("coordinate_unit", "m")
    if unit not in ("m", "mm", "cm"):
        raise ValueError("input.coordinate_unit must be m, mm or cm")
    if sensors.shape != (signal.shape[0], 3):
        raise ValueError(f"RF/sensor shape mismatch after explicit layouts: signal={signal.shape}, sensors={sensors.shape}")
    if signal.shape[0] < 2:
        raise ValueError("At least two sensors are needed for the training/validation split")
    original = _float32(signal, "RF signal")
    sensors = _float32(sensors.astype(np.float64) * {"m": 1., "mm": .001, "cm": .01}[unit], "Sensor coordinates")
    raw_rms = _rms(original)
    if not math.isfinite(raw_rms) or raw_rms <= 0:
        raise ValueError("RF signal has zero energy after canonical float32 conversion")
    target_rms = _positive(selected["normalization"].get("target_rms"), "normalization.target_rms")
    gain = target_rms / raw_rms
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("RMS normalization gain is not finite and positive")
    normalized = _float32(original.astype(np.float64) * gain, "Normalized RF")
    if _rms(normalized) == 0:
        raise ValueError("Requested normalized RF underflows the float32 range")
    acquisition = selected["acquisition"]
    frequency = _positive(acquisition.get("sampling_frequency_hz"), "acquisition.sampling_frequency_hz")
    speed = _positive(acquisition.get("sound_speed_m_s"), "acquisition.sound_speed_m_s")
    samples = acquisition.get("samples")
    if samples is not None and _integer(samples, "acquisition.samples", 1) != signal.shape[1]:
        raise ValueError("acquisition.samples does not match the RF time-axis length; no automatic truncation or resampling")
    sampling = _sampling({"sampling": {"dt_seconds": 1 / frequency, "sampling_frequency_hz": frequency,
                          "samples": int(signal.shape[1]), "sound_speed_m_s": speed,
                          "t0_seconds": acquisition.get("t0_seconds", 0.)}}, signal.shape[1])
    if "split" in paths:
        train, validation = _read_split(paths["split"], len(signal))
        split_rule = "Existing split supplied by file; source indices preserved without reassignment"
        split_seed = None
    else:
        fraction = _positive(split_options.get("validation_fraction", .125), "sensor_split.validation_fraction")
        if fraction >= 1:
            raise ValueError("sensor_split.validation_fraction must be strictly between 0 and 1")
        split_seed = _integer(split_options.get("seed", 20261008), "sensor_split.seed")
        count = min(max(int(round(len(signal) * fraction)), 1), len(signal) - 1)
        permutation = np.random.default_rng(split_seed).permutation(len(signal))
        validation, train = np.sort(permutation[:count]), np.sort(permutation[count:])
        train, validation = _validate_split(train, validation, len(signal))
        split_rule = "default_rng(seed).permutation(S): first clamp(round(S*fraction),1,S-1) validation; remaining train; sort each"
    frame = source.get("frame", paths["signal"].stem)
    if not isinstance(frame, str) or not frame:
        raise ValueError("input.frame must be a descriptive nonempty string, not a multi-frame array index")
    metadata = {
        "frame": frame, "global_gain": gain,
        "rule": "signal_normalized = canonical_float32_signal_original * target_rms / measured_global_rms",
        "normalization": {"target_rms": target_rms, "source_rms": raw_rms,
                          "achieved_rms": _rms(normalized), "scope": "single scalar over the full frame; all sensors and time samples",
                          "channel_relative_amplitudes_preserved": True},
        "processed_original": _stats(original), "normalized": _stats(normalized),
        "sampling": sampling,
        "geometry": {"bounds_m": [-.02, .02] * 3, "voxel_origin_xyz_m": [-.01995] * 3,
                     "voxel_dims_xyz": [400] * 3, "voxel_spacing_m": .0001},
        "coordinate_unit": "m", "source_coordinate_unit": unit,
        "coordinate_order": "Sensor row i corresponds to RF row i; XYZ in metres; only declared layout and unit conversion applied",
        "signal_original_semantics": "Original supplied signed RF amplitudes, stored as float32; raw ADC status and upstream preprocessing are not inferred",
        "preprocessing": "Explicit layout conversion, coordinate-unit conversion and one positive global RF scale; no filtering, abs, sign flip, crop, resampling or per-channel normalization",
        "source_files": sources,
        "dataset_fingerprint": fingerprint,
    }
    split_record = {"seed": split_seed, "rule": split_rule,
                    "training_sensors": train.tolist(), "validation_sensors": validation.tolist(),
                    "counts": [len(train), len(validation)],
                    "validation_purpose": "hyperparameter/model selection; not an independent test; both subsets are training data in an all-sensor refit"}
    # Reject a source that changed during loading before recording provenance.
    if any(_sha256(path) != sources[name]["sha256"] for name, path in paths.items()):
        raise ValueError("An input file changed during preparation; retry after input writing completes")
    folder.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".prepare_dataset_", dir=folder.parent))
    try:
        np.save(stage / "signal_original.npy", original)
        np.save(stage / "signal_normalized.npy", normalized)
        np.save(stage / "sensor_locations.npy", sensors)
        np.savez(stage / "split.npz", train=train, validation=validation)
        _write_json(stage / "split.json", split_record)
        _write_json(stage / "metadata.json", metadata)
        files = {name: {"sha256": _sha256(stage / name), "size_bytes": (stage / name).stat().st_size}
                 for name in _DATA_FILES}
        _write_json(stage / "preparation.json", {"schema_version": 1, "dataset_fingerprint": fingerprint,
                    **identity, "files": files, "writes": "Complete canonical dataset committed by same-filesystem directory rename",
                    "normalization_recomputed_from_source": True})
        load_dataset(stage)
        if folder.exists():
            raise ValueError("Preparation destination appeared while writing; refusing to replace it")
        stage.rename(folder)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return metadata
