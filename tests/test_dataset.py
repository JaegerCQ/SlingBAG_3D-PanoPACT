"""CPU-only acquisition, input-format and transactional-preparation checks."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.io import savemat

from slingbag.dataset import load_dataset, prepare_dataset
from slingbag.paths import PROJECT_ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DatasetChecks(unittest.TestCase):
    def setUp(self):
        outputs = PROJECT_ROOT / "outputs"
        outputs.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="dataset_test_", dir=outputs)
        self.folder = Path(self.temporary.name)
        self.signal = np.asarray([[1., -2., 3., -4., 5., -6., 7., -8.]]) * np.arange(1., 5.)[:, None]
        self.sensors = np.asarray([[100., 0., 0.], [0., 100., 0.], [-100., 0., 0.], [0., -100., 0.]])
        self.signal_path, self.sensor_path = self.folder / "signal.npy", self.folder / "sensors.npy"
        np.save(self.signal_path, self.signal)
        np.save(self.sensor_path, self.sensors)
        self.destination = self.folder / "prepared"
        self.config = {
            "schema_version": 2, "seed": 1013,
            "input": {"signal": str(self.signal_path), "sensors": str(self.sensor_path),
                      "signal_key": None, "sensors_key": None, "signal_layout": "sensor_time",
                      "sensor_layout": "sensor_xyz", "coordinate_unit": "mm", "frame": "small_cpu_frame"},
            "acquisition": {"sound_speed_m_s": 1480., "sampling_frequency_hz": 80e6,
                            "samples": None, "t0_seconds": 0.},
            "normalization": {"target_rms": 2.5},
            "sensor_split": {"file": None, "validation_fraction": .25, "seed": 20261008},
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_dynamic_sampling_global_gain_sign_and_coordinate_units(self):
        before = [digest(self.signal_path), digest(self.sensor_path)]
        metadata = prepare_dataset(self.config, self.destination)
        data = load_dataset(self.destination)
        self.assertEqual(data.signal.dtype, np.float32)
        self.assertEqual(data.signal.shape, (4, 8))
        self.assertEqual(data.num_sensors, 4)
        self.assertEqual(data.num_times, 8)
        self.assertEqual(data.forward_kwargs, {"dt": 1 / 80e6, "num_times": 8, "sound_speed": 1480.})
        np.testing.assert_allclose(data.sensors, self.sensors * .001, rtol=1e-7)
        np.testing.assert_allclose(data.signal, self.signal * data.gain, rtol=1e-7)
        np.testing.assert_array_equal(np.signbit(data.signal), np.signbit(self.signal))
        # A single global gain preserves ratios between sensor rows.
        np.testing.assert_allclose(data.signal[3], data.signal[0] * 4, rtol=1e-7)
        self.assertAlmostEqual(float(np.sqrt(np.mean(data.signal.astype(np.float64) ** 2))), 2.5, places=6)
        self.assertEqual(metadata["sampling"]["sampling_frequency_hz"], 80e6)
        self.assertEqual(before, [digest(self.signal_path), digest(self.sensor_path)])

    def test_supported_numpy_text_csv_and_mat_inputs(self):
        for extension in ("npy", "npz", "txt", "csv", "mat"):
            with self.subTest(format=extension):
                config = copy.deepcopy(self.config)
                for name, values in (("signal", self.signal), ("sensors", self.sensors)):
                    path = self.folder / f"{name}_format.{extension}"
                    config["input"][name] = str(path)
                    if extension == "npy":
                        np.save(path, values)
                    elif extension == "npz":
                        np.savez(path, matrix=values)
                    elif extension == "mat":
                        savemat(path, {"matrix": values})
                    elif extension == "csv":
                        np.savetxt(path, values, delimiter=",", header=",".join(f"v{i}" for i in range(values.shape[1])), comments="")
                        config["input"][name + "_skiprows"] = 1
                    else:
                        np.savetxt(path, values, delimiter=";", header="numeric values", comments="")
                        config["input"][name + "_delimiter"] = ";"
                        config["input"][name + "_skiprows"] = 1
                destination = self.folder / ("prepared_" + extension)
                prepare_dataset(config, destination)
                np.testing.assert_array_equal(np.load(destination / "signal_original.npy"), self.signal.astype(np.float32))
                np.testing.assert_allclose(load_dataset(destination).sensors, self.sensors * .001, rtol=1e-7)

    def test_multi_array_containers_require_keys_and_missing_keys_fail(self):
        for extension in ("npz", "mat"):
            with self.subTest(format=extension):
                path = self.folder / ("both." + extension)
                arrays = {"rf": self.signal, "positions": self.sensors}
                np.savez(path, **arrays) if extension == "npz" else savemat(path, arrays)
                config = copy.deepcopy(self.config)
                config["input"].update(signal=str(path), sensors=str(path))
                with self.assertRaisesRegex(ValueError, "array key"):
                    prepare_dataset(config, self.folder / (extension + "_ambiguous"))
                config["input"].update(signal_key="rf", sensors_key="positions")
                destination = self.folder / (extension + "_keyed")
                prepare_dataset(config, destination)
                self.assertEqual(load_dataset(destination).signal.shape, self.signal.shape)
                config["input"]["signal_key"] = "missing"
                with self.assertRaisesRegex(ValueError, "absent"):
                    prepare_dataset(config, self.folder / (extension + "_missing"))

    def test_explicit_transpose_and_all_coordinate_units(self):
        for unit, scale in (("m", 1.), ("mm", .001), ("cm", .01)):
            with self.subTest(unit=unit):
                np.save(self.signal_path, self.signal.T)
                np.save(self.sensor_path, (self.sensors * .001 / scale).T)
                config = copy.deepcopy(self.config)
                config["input"].update(signal_layout="time_sensor", sensor_layout="xyz_sensor", coordinate_unit=unit)
                destination = self.folder / ("prepared_" + unit)
                prepare_dataset(config, destination)
                data = load_dataset(destination)
                np.testing.assert_array_equal(np.load(destination / "signal_original.npy"), self.signal.astype(np.float32))
                np.testing.assert_allclose(data.sensors, self.sensors * .001, rtol=1e-7)

    def test_invalid_arrays_fail_before_destination_is_committed(self):
        arrays = {
            "nan": self.signal.copy(), "inf": self.signal.copy(),
            "complex": self.signal.astype(np.complex128), "zero": np.zeros_like(self.signal),
            "three_dimensional": self.signal[None], "empty": np.empty((4, 0)),
            "overflow": self.signal * 1e100, "object": self.signal.astype(object),
        }
        arrays["nan"][0, 0] = np.nan
        arrays["inf"][0, 0] = np.inf
        for name, values in arrays.items():
            with self.subTest(case=name):
                np.save(self.signal_path, values)
                destination = self.folder / ("invalid_" + name)
                with self.assertRaises(ValueError):
                    prepare_dataset(self.config, destination)
                self.assertFalse(destination.exists())

    def test_invalid_options_and_sample_mismatch_are_not_silently_fixed(self):
        cases = [("acquisition", "samples", 7), ("acquisition", "samples", True),
                 ("acquisition", "sampling_frequency_hz", 0), ("acquisition", "sound_speed_m_s", None),
                 ("acquisition", "t0_seconds", 1e-6), ("acquisition", "t0_seconds", None),
                 ("normalization", "target_rms", None),
                 ("normalization", "target_rms", -1), ("input", "coordinate_unit", "inches"),
                 ("input", "signal_layout", "auto"), ("input", "frame", 3),
                 ("input", "signal_skiprows", -1), ("input", "signal_layuot", "sensor_time"),
                 ("sensor_split", "validation_fraction", 1.), ("sensor_split", "seed", -1)]
        for index, (section, key, value) in enumerate(cases):
            with self.subTest(section=section, key=key):
                config = copy.deepcopy(self.config)
                config[section][key] = value
                with self.assertRaises(ValueError):
                    prepare_dataset(config, self.folder / ("bad_option_" + str(index)))
        np.save(self.sensor_path, np.zeros((5, 3)))
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            prepare_dataset(self.config, self.destination)

    def test_supplied_split_preserves_membership_and_order(self):
        path = self.folder / "split.npz"
        np.savez(path, train=np.asarray([3, 0, 2]), validation=np.asarray([1]))
        self.config["sensor_split"]["file"] = str(path)
        prepare_dataset(self.config, self.destination)
        data = load_dataset(self.destination)
        np.testing.assert_array_equal(data.train, [3, 0, 2])
        np.testing.assert_array_equal(data.validation, [1])
        np.savez(path, train=np.asarray([0, 1, 2]), validation=np.asarray([2]))
        with self.assertRaisesRegex(ValueError, "disjoint"):
            prepare_dataset(self.config, self.folder / "overlap")

    def test_automatic_split_reproduces_pose000_and_clamps_small_counts(self):
        np.save(self.signal_path, np.tile(np.asarray([[1., -1.]], dtype=np.float32), (1024, 1)))
        np.save(self.sensor_path, np.tile(np.asarray([[100., 0., 0.]]), (1024, 1)))
        self.config["sensor_split"]["validation_fraction"] = .125
        prepare_dataset(self.config, self.destination)
        data = load_dataset(self.destination)
        with np.load(PROJECT_ROOT / "data/pose000/split.npz") as historical:
            np.testing.assert_array_equal(data.train, historical["train"])
            np.testing.assert_array_equal(data.validation, historical["validation"])
        np.save(self.signal_path, self.signal[:2])
        np.save(self.sensor_path, self.sensors[:2])
        self.config["sensor_split"]["validation_fraction"] = .001
        other = self.folder / "small_split"
        prepare_dataset(self.config, other)
        self.assertEqual((len(load_dataset(other).train), len(load_dataset(other).validation)), (1, 1))

    def test_exact_reuse_ignores_reconstruction_settings_without_rewriting(self):
        prepare_dataset(self.config, self.destination)
        before = {p.name: (digest(p), p.stat().st_mtime_ns) for p in self.destination.iterdir()}
        config = copy.deepcopy(self.config)
        config["seed"] = 2026
        config["training"] = {"arbitrary_reconstruction_option": 123}
        prepare_dataset(config, self.destination)
        after = {p.name: (digest(p), p.stat().st_mtime_ns) for p in self.destination.iterdir()}
        self.assertEqual(before, after)
        config["normalization"]["target_rms"] = 3.
        with self.assertRaisesRegex(ValueError, "configuration or sources changed"):
            prepare_dataset(config, self.destination)
        self.assertEqual(before, {p.name: (digest(p), p.stat().st_mtime_ns) for p in self.destination.iterdir()})

    def test_changed_source_or_prepared_content_is_not_overwritten(self):
        prepare_dataset(self.config, self.destination)
        np.save(self.signal_path, self.signal + 1.)
        with self.assertRaisesRegex(ValueError, "configuration or sources changed"):
            prepare_dataset(self.config, self.destination)
        np.save(self.signal_path, self.signal)
        normalized_path = self.destination / "signal_normalized.npy"
        altered = np.load(normalized_path)
        altered[0, 0] += 1.
        np.save(normalized_path, altered)
        changed = digest(normalized_path)
        with self.assertRaisesRegex(ValueError, "changed or is missing"):
            prepare_dataset(self.config, self.destination)
        self.assertEqual(digest(normalized_path), changed)

    def test_failed_staging_is_removed_and_retry_succeeds(self):
        from slingbag import dataset
        original_writer = dataset._write_json

        def fail_metadata(path, values):
            if path.name == "metadata.json":
                raise OSError("Simulated interrupted staging write")
            return original_writer(path, values)

        with mock.patch.object(dataset, "_write_json", side_effect=fail_metadata):
            with self.assertRaises(OSError):
                prepare_dataset(self.config, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertFalse(list(self.folder.glob(".prepare_dataset_*")))
        prepare_dataset(self.config, self.destination)
        self.assertTrue((self.destination / "preparation.json").is_file())

    def test_empty_existing_destination_and_non_outputs_paths_are_rejected(self):
        self.destination.mkdir()
        with self.assertRaisesRegex(ValueError, "without a complete"):
            prepare_dataset(self.config, self.destination)
        for path in (PROJECT_ROOT / "data", PROJECT_ROOT / "examples", PROJECT_ROOT / "outputs"):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "outputs"):
                    prepare_dataset(self.config, path)
        link = self.folder / "link_into_data"
        link.symlink_to(PROJECT_ROOT / "data", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "outputs"):
            prepare_dataset(self.config, link / "must_not_be_created")

    def test_legacy_dt_only_metadata_and_dynamic_default_directory(self):
        prepare_dataset(self.config, self.destination)
        path = self.destination / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["sampling"].pop("sampling_frequency_hz")
        metadata.pop("coordinate_unit")
        path.write_text(json.dumps(metadata))
        with mock.patch("slingbag.dataset.dataset_root", return_value=self.destination):
            data = load_dataset()
        self.assertEqual(data.dt, 1 / 80e6)
        self.assertEqual(data.metadata["sampling"]["sampling_frequency_hz"], 80e6)

    def test_loader_rejects_bad_units_geometry_time_origin_and_double_normalization(self):
        prepare_dataset(self.config, self.destination)
        path = self.destination / "metadata.json"
        original = json.loads(path.read_text())
        variants = []
        for name, value in (("bounds_m", [-.03, .03] * 3), ("voxel_dims_xyz", [200] * 3),
                            ("voxel_spacing_m", .0002), ("voxel_origin_xyz_m", [0.] * 3)):
            metadata = copy.deepcopy(original)
            metadata["geometry"][name] = value
            variants.append(metadata)
        for parent, key, value in ((None, "coordinate_unit", "mm"), ("sampling", "t0_seconds", 1e-6),
                                   ("sampling", "sampling_frequency_hz", 40e6), (None, "global_gain", -1)):
            metadata = copy.deepcopy(original)
            (metadata if parent is None else metadata[parent])[key] = value
            variants.append(metadata)
        for metadata in variants:
            path.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                load_dataset(self.destination)
        path.write_text(json.dumps(original))
        np.save(self.destination / "signal_normalized.npy", np.load(self.destination / "signal_normalized.npy") * 2)
        with self.assertRaisesRegex(ValueError, "global_gain"):
            load_dataset(self.destination)

    def test_mat73_error_is_actionable_without_optional_hdf5_dependency(self):
        path = self.folder / "signal.mat"
        path.write_bytes(b"fixture")
        self.config["input"]["signal"] = str(path)
        with mock.patch("slingbag.dataset.loadmat", side_effect=NotImplementedError("v7.3")):
            with self.assertRaisesRegex(ValueError, "MAT v7.3/HDF5"):
                prepare_dataset(self.config, self.destination)


if __name__ == "__main__":
    unittest.main()
