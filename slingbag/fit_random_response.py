#!/usr/bin/env python3
"""CPU-only shared FIR diagnostic fitted independently from a random reconstruction.

Fits a small zero-padded, two-sided time convolution to an existing prediction.
The filter is shared across all sensors. Ridge weights are selected on the
existing validation sensors, which consequently are not an independent test.
Only this project's JSON report and selected small FIR array are written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import scipy
from scipy import fft

from .paths import ROOT, data_path, resolve_input, response_path, is_artifact_path, display_path
from .dataset import load_dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_path(value: str) -> Path:
    path = resolve_input(value).resolve(strict=True)
    if not path.is_relative_to(ROOT):
        raise ValueError("Inputs must remain inside the random-only project")
    return path


def output_path(value: str) -> Path:
    return response_path(value)


def normal_equations(prediction: np.ndarray, target: np.ndarray, lags: np.ndarray,
                     sensor_batch: int = 64) -> tuple[np.ndarray, np.ndarray, dict]:
    """Exact finite-window Gram/cross product without constructing [S*T,K].

    For A[s,t,k]=prediction[s,t-lags[k]], first use full zero-extended
    autocorrelation for the Toeplitz Gram matrix, then subtract the short
    contributions outside the observed output window. The target is also
    zero-extended, so its cross correlation needs no boundary correction.
    """
    sensors, samples = prediction.shape
    transform_size = fft.next_fast_len(2 * samples - 1)
    auto_spectrum = np.zeros(transform_size // 2 + 1, dtype=np.float64)
    cross_spectrum = np.zeros(transform_size // 2 + 1, dtype=np.complex128)
    for start in range(0, sensors, sensor_batch):
        a = fft.rfft(prediction[start:start + sensor_batch], transform_size, axis=1, workers=4)
        b = fft.rfft(target[start:start + sensor_batch], transform_size, axis=1, workers=4)
        auto_spectrum += np.sum(np.abs(a) ** 2, axis=0)
        cross_spectrum += np.sum(np.conjugate(a) * b, axis=0)
    autocorrelation = fft.irfft(auto_spectrum, transform_size)
    cross_correlation = fft.irfft(cross_spectrum, transform_size)
    differences = lags[:, None] - lags[None, :]
    # Correlation outside [-(T-1), T-1] is exactly zero. Explicit masking
    # avoids cyclic FFT aliasing for a CLI half-width close to T.
    gram = np.where(np.abs(differences) < samples,
                    autocorrelation[differences % transform_size], 0.)
    cross = cross_correlation[lags % transform_size].copy()
    edge_times = np.concatenate((np.arange(min(0, int(lags.min())), 0),
                                 np.arange(samples, samples + max(0, int(lags.max())))))
    edge_gram = np.zeros_like(gram)
    if len(edge_times):
        indices = edge_times[:, None] - lags[None, :]
        valid = (indices >= 0) & (indices < samples)
        for start in range(0, sensors, sensor_batch):
            edge_design = prediction[start:start + sensor_batch, indices.clip(0, samples - 1)] * valid
            edge_design = edge_design.reshape(-1, len(lags))
            edge_gram += edge_design.T @ edge_design
        gram -= edge_gram
    gram = (gram + gram.T) * .5
    return gram, cross, {
        "fft_size": transform_size,
        "sensor_batch": sensor_batch,
        "finite_window_boundary_correction_frobenius": float(np.linalg.norm(edge_gram)),
        "method": "FFT auto/cross correlation plus exact finite-output-window Gram correction; no [S*T,41] matrix is constructed.",
    }


def check_normal_equations() -> dict:
    rng = np.random.default_rng(1013)
    checks = []
    for samples, half_width in ((57, 3), (31, 20), (31, 0)):
        x, y = rng.normal(size=(2, 3, samples))
        lags = np.arange(-half_width, half_width + 1)
        gram, cross, _ = normal_equations(x, y, lags)
        indices = np.arange(samples)[:, None] - lags[None, :]
        valid = (indices >= 0) & (indices < samples)
        design = (x[:, indices.clip(0, samples - 1)] * valid).reshape(-1, len(lags))
        expected_gram, expected_cross = design.T @ design, design.T @ y.ravel()
        np.testing.assert_allclose(gram, expected_gram, rtol=1e-12, atol=1e-11)
        np.testing.assert_allclose(cross, expected_cross, rtol=1e-12, atol=1e-11)
        checks.append({"sensors": 3, "time_points": samples, "half_width": half_width,
                       "gram_max_absolute_difference": float(np.max(np.abs(gram - expected_gram))),
                       "cross_max_absolute_difference": float(np.max(np.abs(cross - expected_cross)))})
    return {"status": "passed", "fixture": "Dense design matrix with nonzero boundaries, long FIR and identity-only cases",
            "cases": checks}


def signal_metrics(prediction: np.ndarray, target: np.ndarray, indices: np.ndarray) -> dict:
    p, y = prediction[indices], target[indices]
    residual = p - y
    error = float(np.mean(residual ** 2))
    energy = float(np.mean(y ** 2))
    p_centered, y_centered = p - p.mean(), y - y.mean()
    denominator = float(np.sqrt(np.sum(p_centered ** 2) * np.sum(y_centered ** 2)))
    correlation = float(np.sum(p_centered * y_centered) / denominator) if denominator else None
    return {"sensor_count": len(indices), "sample_count": p.size, "mse": error,
            "target_energy": energy, "relative_l2": float(np.sqrt(error / energy)),
            "pearson_correlation": correlation,
            "prediction_rms": float(np.sqrt(np.mean(p ** 2))),
            "target_rms": float(np.sqrt(energy))}


def describe_filter(h: np.ndarray, lags: np.ndarray, dt: float) -> dict:
    peak = int(np.argmax(np.abs(h)))
    # This frequency grid describes the kernel, independently of RF length.
    # Retain the original grid for ordinary kernels, never truncate long FIRs.
    response = fft.rfft(h, n=max(4096, fft.next_fast_len(len(h))))
    h0 = (lags == 0).astype(np.float64)
    return {"coefficients_in_increasing_lag_order": h.tolist(), "coefficient_sum_dc_gain": float(h.sum()),
            "coefficient_l2_norm": float(np.linalg.norm(h)), "distance_from_identity_l2": float(np.linalg.norm(h - h0)),
            "peak_absolute_coefficient": float(np.max(np.abs(h))), "peak_coefficient_lag_samples": int(lags[peak]),
            "peak_coefficient_lag_ns": float(lags[peak] * dt * 1e9),
            "maximum_frequency_response_magnitude": float(np.max(np.abs(response)))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", default="outputs/coarse/export/acoustic_signal_normalized.npy")
    parser.add_argument("--output-prefix", default="outputs/validation/random_response")
    parser.add_argument("--half-width", type=int, default=20)
    parser.add_argument("--ridge", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, .1, 1., 10.])
    args = parser.parse_args()
    if args.half_width < 0 or not all(np.isfinite(a) and a > 0 for a in args.ridge):
        raise ValueError("Half width must be nonnegative and ridge strengths finite/positive")
    started = time.perf_counter()
    dataset = load_dataset()
    numerical_check = check_normal_equations()
    prediction_path = input_path(args.prediction)
    if not is_artifact_path(prediction_path):
        raise ValueError("Prediction must reside in this project's outputs or examples")
    prefix = output_path(args.output_prefix)
    report_path = output_path(str(prefix) + ".json")
    filter_path = output_path(str(prefix) + ".npy")
    target_path = data_path("signal_normalized.npy")
    split_path = data_path("split.npz")
    metadata_path = data_path("metadata.json")
    prediction = np.load(prediction_path, allow_pickle=False).astype(np.float64)
    target = dataset.signal.astype(np.float64)
    train, validation = dataset.train, dataset.validation
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Prediction and target must share a [sensor,time] shape")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Prediction and target must be finite")
    if len(np.unique(np.concatenate((train, validation)))) != len(train) + len(validation):
        raise ValueError("Training/validation rows overlap or contain duplicates")
    if not np.array_equal(np.sort(np.concatenate((train, validation))), np.arange(len(target))):
        raise ValueError("Split does not partition all signal rows")
    if args.half_width >= target.shape[1]:
        raise ValueError("FIR half width must be shorter than the signal")
    lags = np.arange(-args.half_width, args.half_width + 1)
    h0 = (lags == 0).astype(np.float64)
    gram, cross, method = normal_equations(prediction[train], target[train], lags)
    observations = len(train) * target.shape[1]
    gram /= observations
    cross /= observations
    gram_scale = float(np.trace(gram) / len(lags))
    if gram_scale <= 0:
        raise ValueError("Training predictions have zero energy")
    eigenvalues = np.linalg.eigvalsh(gram)
    convolution_size = fft.next_fast_len(target.shape[1] + len(lags) - 1)
    prediction_fft = fft.rfft(prediction, convolution_size, axis=1, workers=4)

    def apply(h):
        full = fft.irfft(prediction_fft * fft.rfft(h, convolution_size)[None, :],
                         convolution_size, axis=1, workers=4)
        return full[:, args.half_width:args.half_width + target.shape[1]]

    np.testing.assert_allclose(apply(h0), prediction, rtol=1e-12, atol=1e-11)
    variants = []

    def evaluate(name, h, extra):
        filtered = apply(h)
        variants.append({"name": name, **extra,
                         "train": signal_metrics(filtered, target, train),
                         "validation": signal_metrics(filtered, target, validation),
                         "filter": describe_filter(h, lags, dataset.dt)})

    evaluate("identity_baseline", h0, {"fit": "No fitted response parameters"})
    center = args.half_width
    scalar_gain = max(0., float(cross[center] / gram[center, center]))
    evaluate("training_scalar_gain_only", scalar_gain * h0,
             {"fit": "One nonnegative scalar fitted only on training sensors", "scalar_gain": scalar_gain})
    for alpha in args.ridge:
        strength = float(alpha * gram_scale)
        matrix = gram + strength * np.eye(len(lags))
        h = np.linalg.solve(matrix, cross + strength * h0)
        evaluate(f"shared_FIR_ridge_{alpha:g}", h, {
            "fit": "41-tap shared FIR fitted only on training sensors" if len(lags) == 41 else f"{len(lags)}-tap shared FIR fitted only on training sensors",
            "ridge_relative_strength": alpha, "ridge_absolute_strength_in_mean_objective": strength,
            "ridge_center": "Identity delta response",
            "regularized_gram_condition_number": float(np.linalg.cond(matrix)),
            "effective_linear_degrees_of_freedom": float(np.sum(eigenvalues / (eigenvalues + strength))),
            "normal_equation_relative_residual": float(np.linalg.norm(matrix @ h - cross - strength * h0) /
                                                        np.linalg.norm(cross + strength * h0)),
        })
    baseline = variants[0]
    for candidate in variants:
        for group in ("train", "validation"):
            original_mse = baseline[group]["mse"]
            candidate[group]["mse_reduction_from_identity_fraction"] = 1 - candidate[group]["mse"] / original_mse
    best = min(variants, key=lambda candidate: candidate["validation"]["mse"])
    for path in (report_path, filter_path):
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError("Output escapes this project")
        path.parent.mkdir(parents=True, exist_ok=True)
    np.save(filter_path, np.asarray(best["filter"]["coefficients_in_increasing_lag_order"], dtype=np.float64))
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Empirical shared temporal-response diagnostic using only this project's random-reconstruction acoustic prediction, measured RF and fixed sensor split. No GPU, teacher, DAS, previous FIR, per-sensor fit, coordinate change or RF polarity change.",
        "inputs": {display_path(path): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
                   for path in (prediction_path, target_path, split_path, metadata_path)},
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                    "device": "CPU", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset")},
        "training_rows": train.tolist(), "validation_rows": validation.tolist(),
        "structure_references_or_previous_fir_read": False,
        "signal_shape": list(target.shape), "training_sensor_count": len(train), "validation_sensor_count": len(validation),
        "time_sample_seconds": dataset.dt, "lags_samples": lags.tolist(),
        "acquisition": {**dataset.forward_kwargs, "num_sensors": dataset.num_sensors,
                        "t0_seconds": 0., "global_gain": dataset.gain},
        "lag_support_ns": [float(lags[0] * dataset.dt * 1e9), float(lags[-1] * dataset.dt * 1e9)],
        "convolution_definition": "filtered[s,t] = sum_k h[k] * prediction[s,t-lags[k]], with zero padding outside the recorded signal. Positive lag delays the prediction. A two-sided diagnostic FIR need not be causal.",
        "fit_objective": "mean_train((A*h-target)^2) + alpha*mean(diag(A.T*A)/N)*||h-delta||_2^2; N=train_sensors*time_samples. All normal equations and amplitudes use training rows only.",
        "normal_equations": {**method, "diagonal_mean": gram_scale,
                             "minimum_gram_eigenvalue": float(eigenvalues.min()), "maximum_gram_eigenvalue": float(eigenvalues.max())},
        "numerical_checks": {**numerical_check, "identity_convolution": "passed"},
        "variants": variants,
        "selected": {"name": best["name"], "selection": "Lowest validation MSE among identity, train scalar gain and predeclared ridge candidates; validation is used for model selection, not an independent test.",
                     "train": best["train"], "validation": best["validation"], "filter": best["filter"],
                     "ridge_relative_strength": best.get("ridge_relative_strength"),
                     "filter_npy": str(filter_path), "filter_sha256": sha256(filter_path)},
        "interpretation": [
            "This fitted FIR is not a measured instrument impulse response. It may absorb shared timing, bandwidth or model/initialization errors and does not identify their unique physical cause.",
            "A validation improvement indicates that a common time-response correction transfers across the reserved sensors for this fixed prediction; it does not establish improved reconstructed image quality.",
            "If validation MSE does not improve, retain the baseline and do not infer a useful shared response correction.",
            f"FIR normal equations and the optional scalar gain use only the {len(train)} training sensors. The input reconstruction checkpoint may itself already have been selected with validation; validation is used again here for ridge/model selection and is not an independent test.",
            "This diagnostic fits signed FIR coefficients, not signed Gaussian source pressures. No channel-specific response or normalization is fitted.",
        ],
        "recommendation": {
            "eligible_for_controlled_fine_ablation": best["name"].startswith("shared_FIR_") and best["validation"]["mse"] < baseline["validation"]["mse"],
            "criterion": "A shared FIR selected over identity/scalar provides lower reserved-sensor MSE; this supports an ablation, not guaranteed image improvement.",
            "transfer_limit": "A filter estimated for this fixed broad-width prediction may compensate for source-width/position errors and need not transfer unchanged to narrower or optimized sources. Keep an unfiltered fine-stage control.",
        },
        "elapsed_cpu_wall_seconds": time.perf_counter() - started,
        "script_sha256": sha256(Path(__file__)),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(report_path), "selected": best["name"],
                      "baseline_train_mse": baseline["train"]["mse"], "selected_train_mse": best["train"]["mse"],
                      "baseline_validation_mse": baseline["validation"]["mse"], "selected_validation_mse": best["validation"]["mse"],
                      "filter": str(filter_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
