#!/usr/bin/env python3
"""Calibrate a fixed shared FIR after all-sensor point-cloud reconstruction.

Ridge is fixed before this fit (default 10). All sensor rows are fitting data;
neither original split is validation. Repeated runs always use the immutable
acoustic prediction and acoustic checkpoint, never an already filtered signal.
No acoustic forward pass, GPU operation, point update or volume update occurs.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
from scipy import fft
import torch

from .fit_random_response import (check_normal_equations, describe_filter,
                                 normal_equations, sha256, signal_metrics)

from .paths import ROOT, PACKAGE_ROOT, data_path, resolve_output, display_path
from .dataset import load_dataset
HALF_WIDTH = 20


def local_path(path: str | Path, *, exists=False) -> Path:
    return resolve_output(path).resolve(strict=exists)


def metric_groups(prediction, target, train, original_validation):
    groups = {"full": np.arange(len(target)), "original_train_subset": train,
              "original_validation_subset": original_validation}
    roles = {"full": f"all_{len(target)}_training_and_response_fit_sensors",
             "original_train_subset": f"original_{len(train)}_subset_also_used_for_training_and_response_fit",
             "original_validation_subset": f"original_{len(original_validation)}_subset_also_used_for_training_and_response_fit_NOT_validation"}
    out = {}
    for name, ids in groups.items():
        values = signal_metrics(prediction, target, ids)
        values["correlation"] = values.pop("pearson_correlation")
        values["role"] = roles[name]
        out[name] = values
    return out


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _validate_acoustic_configuration(checkpoint, metrics):
    config = checkpoint["configuration"]
    if config.get("all_sensors") is not True or not metrics.get("all_sensors_trained"):
        raise ValueError("Only an all_sensors=true reconstruction can receive final all-sensor calibration")
    if config.get("random_only_ancestry") is not True:
        raise ValueError("Expected random-only point-cloud ancestry")
    if config.get("response_kernel_coefficients") is not None or config.get("response_kernel") is not None:
        raise ValueError("The acoustic checkpoint must describe point training without a response kernel")
    if config != metrics["configuration"]:
        raise ValueError("Acoustic checkpoint and metrics disagree about configuration")
    if checkpoint.get("record") != metrics.get("record"):
        raise ValueError("Acoustic checkpoint and metrics disagree about the training record")


def _validate_partial_backup_recovery(folder, backup_pairs, checkpoint, metrics,
                                      acoustic, old_prediction, old_original, gain):
    """Approve recovery only while every current input is still uncalibrated.

    This helper validates without writing. Missing files are completed later,
    after all ordinary signal/metric checks and staged outputs have succeeded.
    An existing backup is never overwritten to hide a conflicting state.
    """
    existing = [name for name in backup_pairs if (folder / name).exists()]
    if not existing or len(existing) == len(backup_pairs):
        raise ValueError("Partial-backup recovery requires a nonempty incomplete backup set")
    _validate_acoustic_configuration(checkpoint, metrics)
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("Global RF normalization gain must be finite and positive")
    if not np.isfinite(acoustic).all():
        raise ValueError("Acoustic prediction must be finite")
    np.testing.assert_array_equal(old_prediction, acoustic)
    np.testing.assert_allclose(old_original, old_prediction / gain, rtol=2e-7, atol=1e-6)
    for backup_name in existing:
        original_name = backup_pairs[backup_name]
        if sha256(folder / backup_name) != sha256(folder / original_name):
            raise RuntimeError(f"Incomplete acoustic backups conflict with the current export: {backup_name}")
    return existing


def _complete_acoustic_backups(folder, backup_pairs, stage):
    """Keep verified existing backups and atomically install each missing one."""
    # Check the whole existing set before writing any missing backup.
    for backup_name, original_name in backup_pairs.items():
        if (folder / backup_name).exists() and sha256(folder / backup_name) != sha256(folder / original_name):
            raise RuntimeError(f"Acoustic backup conflicts with the current export: {backup_name}")
    for backup_name, original_name in backup_pairs.items():
        if (folder / backup_name).exists():
            continue
        staged_backup = stage / backup_name
        shutil.copyfile(folder / original_name, staged_backup)
        if sha256(staged_backup) != sha256(folder / original_name):
            raise RuntimeError("Acoustic backup did not preserve source bytes")
        os.replace(staged_backup, folder / backup_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", default="outputs/final")
    parser.add_argument("--ridge", type=float, default=10.,
                        help="Positive fixed ridge; this final fit does not search hyperparameters")
    parser.add_argument("--half-width", type=int, default=HALF_WIDTH,
                        help="FIR lags run from -half-width to +half-width samples")
    args = parser.parse_args()
    if not np.isfinite(args.ridge) or args.ridge <= 0 or args.half_width < 0:
        raise ValueError("Ridge must be finite/positive and FIR half width nonnegative")
    dataset = load_dataset()
    if args.half_width >= dataset.num_times:
        raise ValueError("FIR half width must be shorter than the signal")
    started = time.perf_counter()
    folder = local_path(args.export, exists=True)
    if not folder.is_dir():
        raise ValueError("--export must identify a completed export directory")
    required = ("metrics.json", "checkpoint.pt", "acoustic_signal_normalized.npy",
                "predicted_signal_normalized.npy", "predicted_signal_original_units.npy",
                "point_cloud.ply", "volume_zyx.npy")
    for name in required:
        local_path(folder / name, exists=True)
    backup_pairs = {
        "metrics_acoustic.json": "metrics.json",
        "checkpoint_acoustic.pt": "checkpoint.pt",
        "predicted_signal_normalized_acoustic.npy": "predicted_signal_normalized.npy",
        "predicted_signal_original_units_acoustic.npy": "predicted_signal_original_units.npy",
    }
    for backup_name in backup_pairs:
        local_path(folder / backup_name)
    backup_flags = [(folder / name).exists() for name in backup_pairs]
    partial_backups = any(backup_flags) and not all(backup_flags)
    rerun = all(backup_flags)
    checkpoint_path = folder / ("checkpoint_acoustic.pt" if rerun else "checkpoint.pt")
    metrics_path = folder / ("metrics_acoustic.json" if rerun else "metrics.json")
    old_prediction_path = folder / ("predicted_signal_normalized_acoustic.npy" if rerun else "predicted_signal_normalized.npy")
    old_original_path = folder / ("predicted_signal_original_units_acoustic.npy" if rerun else "predicted_signal_original_units.npy")
    for path in (checkpoint_path, metrics_path, old_prediction_path, old_original_path):
        local_path(path, exists=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    old_metrics = json.loads(metrics_path.read_text())
    config = checkpoint["configuration"]
    _validate_acoustic_configuration(checkpoint, old_metrics)
    from .export_random import validate_checkpoint_dataset
    validate_checkpoint_dataset(config, dataset)

    acoustic_path = folder / "acoustic_signal_normalized.npy"
    target_path = data_path("signal_normalized.npy")
    metadata_path = data_path("metadata.json")
    split_path = data_path("split.npz")
    acoustic = np.load(acoustic_path, allow_pickle=False).astype(np.float64)
    target = dataset.signal.astype(np.float64)
    if acoustic.shape != target.shape:
        raise ValueError(f"Expected acoustic prediction and target with shape [{dataset.num_sensors},{dataset.num_times}]")
    if not np.isfinite(acoustic).all() or not np.isfinite(target).all():
        raise ValueError("Signals must be finite")
    gain = dataset.gain
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("Global RF normalization gain must be finite and positive")
    original_train, original_validation = dataset.train, dataset.validation
    old_prediction = np.load(old_prediction_path, allow_pickle=False)
    old_original = np.load(old_original_path, allow_pickle=False)
    np.testing.assert_array_equal(old_prediction, acoustic)
    np.testing.assert_allclose(old_original, old_prediction / gain, rtol=2e-7, atol=1e-6)
    recovered_backups = _validate_partial_backup_recovery(
        folder, backup_pairs, checkpoint, old_metrics, acoustic, old_prediction, old_original, gain
    ) if partial_backups else []
    before = metric_groups(acoustic, target, original_train, original_validation)
    for name in before:
        np.testing.assert_allclose(before[name]["mse"], old_metrics[name]["mse"], rtol=2e-7, atol=1e-7)
    immutable_names = ("acoustic_signal_normalized.npy", "point_cloud.ply", "volume_zyx.npy")
    if (folder / "volume_zyx.mat").exists():
        local_path(folder / "volume_zyx.mat", exists=True)
        immutable_names += ("volume_zyx.mat",)
    immutable_hashes = {name: sha256(folder / name) for name in immutable_names}
    if rerun:
        previous_report_path = folder / "response_calibration.json"
        if previous_report_path.exists():
            previous_report = json.loads(previous_report_path.read_text())
            if previous_report["unchanged_file_sha256"] != immutable_hashes:
                raise ValueError("The acoustic point/volume export changed since the previous calibration")

    numerical_checks = check_normal_equations()
    half_width = args.half_width
    lags = np.arange(-half_width, half_width + 1)
    identity = (lags == 0).astype(np.float64)
    gram, cross, method = normal_equations(acoustic, target, lags)
    observations = acoustic.size
    gram /= observations
    cross /= observations
    diagonal_mean = float(np.trace(gram) / len(lags))
    if diagonal_mean <= 0:
        raise ValueError("Acoustic prediction has zero energy")
    strength = args.ridge * diagonal_mean
    matrix = gram + strength * np.eye(len(lags))
    fitted_kernel = np.linalg.solve(matrix, cross + strength * identity)
    # Save coefficients exactly representable in the float32 acoustic exporter.
    kernel = fitted_kernel.astype(np.float32)
    fft_size = fft.next_fast_len(acoustic.shape[1] + len(lags) - 1)
    full = fft.irfft(fft.rfft(acoustic, fft_size, axis=1, workers=4) *
                     fft.rfft(kernel.astype(np.float64), fft_size)[None],
                     fft_size, axis=1, workers=4)
    predicted = full[:, half_width:half_width + acoustic.shape[1]].astype(np.float32)
    predicted_original = predicted / gain
    # Verify lag direction and finite-window cropping with direct convolution.
    direct = np.convolve(acoustic[0], kernel.astype(np.float64), mode="full")[half_width:half_width + acoustic.shape[1]]
    np.testing.assert_allclose(predicted[0], direct, rtol=2e-6, atol=2e-6)
    after = metric_groups(predicted.astype(np.float64), target, original_train, original_validation)
    procedure = f"Point-cloud training without FIR, followed by fixed-ridge shared response calibration on all {dataset.num_sensors} sensors; sequential, not joint optimization"
    kernel_relative_path = str((folder / "response_kernel.npy").relative_to(ROOT))
    calibration = {
        "procedure": procedure, "ridge_relative_strength": args.ridge,
        "ridge_absolute_strength": strength, "ridge_center": "identity delta",
        "ridge_selection": f"Fixed at {args.ridge:g} by run configuration before this all-sensor fit; no search or selection here",
        "sensor_count": dataset.num_sensors, "independent_validation_sensor_count": 0,
        "original_validation_subset_role": "trained and response-fitted subset, not validation",
        "time_sample_seconds": dataset.dt, "lags_samples": lags.tolist(),
        "acquisition": {**dataset.forward_kwargs, "num_sensors": dataset.num_sensors,
                        "t0_seconds": 0., "global_gain": dataset.gain},
        "response_kernel": kernel_relative_path,
        "response_kernel_coefficients": kernel.tolist(),
        "coefficient_storage_dtype": "float32; fit solved in float64",
        "point_parameters_changed": False, "volume_changed": False,
    }
    new_config = copy.deepcopy(config)
    new_config.update({"point_training_response_kernel": None,
                       "point_training_response_kernel_coefficients": None,
                       "response_kernel": kernel_relative_path,
                       "response_kernel_coefficients": kernel.tolist(),
                       "response_calibration": calibration,
                       "acquisition": calibration["acquisition"],
                       "optimization_procedure": procedure})
    acoustic_record = copy.deepcopy(checkpoint["record"])
    new_record = {**acoustic_record,
                  "acoustic_training_record": acoustic_record,
                  "metric_stage": "after fixed-ridge all-sensor response calibration",
                  "train_full_mse": after["full"]["mse"],
                  "reserved_or_original_subset_mse": after["original_validation_subset"]["mse"],
                  "original_train_subset_mse": after["original_train_subset"]["mse"],
                  "subset_is_trained": True,
                  "response_kernel_coefficients": kernel.tolist(),
                  "response_calibration": calibration}
    calibrated_checkpoint = {**checkpoint, "configuration": new_config, "record": new_record}
    for key in ("xyz", "pressure", "radius"):
        assert torch.equal(calibrated_checkpoint[key], checkpoint[key])
    metrics = copy.deepcopy(old_metrics)
    metrics.update({"configuration": new_config, "record": new_record, **after,
                    "acoustic_metrics_before_calibration": before,
                    "all_sensors_trained": True,
                    "scope": f"All {dataset.num_sensors} sensors were used for point training and response estimation. Both original split subsets are trained; neither is validation or test.",
                    "response_calibration": calibration,
                    "response_calibration_report": str(folder / "response_calibration.json")})
    inputs = {display_path(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
              for path in (acoustic_path, target_path, split_path, metadata_path, checkpoint_path, metrics_path,
                           old_prediction_path, old_original_path)}
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "procedure": procedure,
        "all_metrics_are_in_sample": True, "configuration": calibration,
        "before": before, "after": after,
        "full_mse_reduction_fraction": 1 - after["full"]["mse"] / before["full"]["mse"],
        "filter": describe_filter(kernel.astype(np.float64), lags, dataset.dt),
        "float32_coefficient_max_rounding_error": float(np.max(np.abs(kernel.astype(np.float64) - fitted_kernel))),
        "normal_equations": {**method, "fitting_sensor_rows": f"all rows 0..{dataset.num_sensors - 1}, without validation-based selection",
                             "diagonal_mean": diagonal_mean,
                             "condition_number": float(np.linalg.cond(matrix)),
                             "relative_residual": float(np.linalg.norm(matrix @ fitted_kernel - cross - strength * identity) /
                                                         np.linalg.norm(cross + strength * identity))},
        "numerical_checks": {"dense_design": numerical_checks, "direct_convolution": "passed",
                             "old_acoustic_prediction_and_metrics": "passed", "checkpoint_point_parameters": "bitwise unchanged"},
        "inputs": inputs, "unchanged_file_sha256": immutable_hashes,
        "backups": backup_pairs, "rerun_from_acoustic_backups": rerun,
        "recovered_partial_acoustic_backups": recovered_backups,
        "implementation": {"device": "CPU", "torch_cuda_initialized": torch.cuda.is_initialized(),
                           "script_sha256": sha256(Path(__file__)),
                           "fir_math_script_sha256": sha256(PACKAGE_ROOT / "fit_random_response.py")},
        "interpretation": ["The fitted shared FIR is empirical, not a measured instrument impulse response.",
                           "All final fit metrics are in-sample. Earlier reserved-sensor tuning results must be reported separately.",
                           "No point-cloud or voxel-volume quality claim follows from this response-fit MSE reduction."],
    }
    # Stage complete replacements before making any change to exported outputs.
    with tempfile.TemporaryDirectory(prefix=".response_stage_", dir=folder) as temporary:
        stage = Path(temporary)
        np.save(stage / "response_kernel.npy", kernel)
        np.save(stage / "predicted_signal_normalized.npy", predicted)
        np.save(stage / "predicted_signal_original_units.npy", predicted_original)
        torch.save(calibrated_checkpoint, stage / "checkpoint.pt")
        write_json(stage / "config.json", new_config)
        write_json(stage / "record.json", new_record)
        write_json(stage / "metrics.json", metrics)
        reread = torch.load(stage / "checkpoint.pt", map_location="cpu", weights_only=False)
        for key in ("xyz", "pressure", "radius"):
            assert torch.equal(reread[key], checkpoint[key])
        if not rerun:
            _complete_acoustic_backups(folder, backup_pairs, stage)
        output_names = ("response_kernel.npy", "predicted_signal_normalized.npy",
                        "predicted_signal_original_units.npy", "checkpoint.pt", "config.json", "record.json", "metrics.json")
        report["outputs"] = {name: {"sha256": sha256(stage / name)} for name in output_names}
        report["acoustic_backup_sha256"] = {name: sha256(folder / name) for name in backup_pairs}
        for name in output_names:
            local_path(folder / name)
            os.replace(stage / name, folder / name)
        after_hashes = {name: sha256(folder / name) for name in immutable_names}
        if immutable_hashes != after_hashes:
            raise RuntimeError("An acoustic, point-cloud or volume file changed during calibration")
        report["numerical_checks"]["immutable_file_hashes"] = "unchanged"
        report["elapsed_cpu_wall_seconds"] = time.perf_counter() - started
        write_json(stage / "response_calibration.json", report)
        os.replace(stage / "response_calibration.json", local_path(folder / "response_calibration.json"))
    print(json.dumps({"report": str(folder / "response_calibration.json"),
                      "ridge": args.ridge, "all_sensors_used_for_fit": True,
                      "before_full_mse": before["full"]["mse"], "after_full_mse": after["full"]["mse"],
                      "original_train_subset_mse": after["original_train_subset"]["mse"],
                      "original_validation_subset_mse": after["original_validation_subset"]["mse"],
                      "point_and_volume_hashes_unchanged": True, "rerun_from_acoustic_backups": rerun}))


if __name__ == "__main__":
    main()
