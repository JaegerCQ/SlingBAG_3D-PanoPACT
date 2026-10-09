"""CPU checks for prepared datasets with dynamic acquisition specifications."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_environment import check_data


class PreparedDatasetChecks(unittest.TestCase):
    def setUp(self):
        cache = ROOT / ".cache" / "environment_data_tests"
        cache.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=cache)
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, sensors=5, samples=37, frequency=20e6, speed=1485., target_rms=3.2):
        rng = np.random.default_rng(938)
        raw = rng.normal(size=(sensors, samples)).astype(np.float32) * 17
        raw_rms = float(np.sqrt(np.mean(raw.astype(np.float64)**2)))
        gain = target_rms / raw_rms
        normalized = (raw.astype(np.float64) * gain).astype(np.float32)
        xyz = np.zeros((sensors, 3), dtype=np.float32)
        angle = np.arange(sensors) * (2 * np.pi / sensors)
        xyz[:, 0], xyz[:, 1], xyz[:, 2] = .1 * np.cos(angle), .1 * np.sin(angle), .03
        np.save(self.directory / "signal_original.npy", raw)
        np.save(self.directory / "signal_normalized.npy", normalized)
        np.save(self.directory / "sensor_locations.npy", xyz)
        np.savez(self.directory / "split.npz", train=np.arange(sensors - 1), validation=np.asarray([sensors - 1]))
        self.metadata = {"global_gain": gain, "normalization": {"target_rms": target_rms},
            "processed_original": {"shape": [sensors, samples], "rms": raw_rms},
            "normalized": {"shape": [sensors, samples], "rms": float(np.sqrt(np.mean(normalized.astype(np.float64)**2)))},
            "sampling": {"samples": samples, "dt_seconds": 1 / frequency, "sampling_frequency_hz": frequency,
                         "sound_speed_m_s": speed, "t0_seconds": 0},
            "geometry": {"voxel_origin_xyz_m": [-.01995] * 3, "voxel_dims_xyz": [400] * 3,
                         "voxel_spacing_m": .0001, "bounds_m": [-.02, .02] * 3},
            "coordinate_order": "sensor XYZ in metres; RF rows retain sensor order"}
        self.write_metadata()
        return raw

    def write_metadata(self):
        (self.directory / "metadata.json").write_text(json.dumps(self.metadata))

    def checks(self):
        checks = []
        check_data(self.directory, checks)
        return checks

    def assert_failure(self, name):
        checks = self.checks()
        self.assertTrue(any(c["check"] == name and c["status"] == "FAIL" for c in checks), checks)

    def test_dynamic_acquisitions_and_auto_gain(self):
        for n, t, fs, c in ((5, 37, 20e6, 1485.), (11, 65, 64e6, 1550.), (2, 1, 12.5e6, 1400.)):
            with self.subTest(sensors=n, samples=t, frequency=fs, speed=c):
                self.fixture(n, t, fs, c)
                checks = self.checks()
                self.assertFalse(any(c["status"] == "FAIL" for c in checks), checks)
                self.assertTrue(any(c["check"] == "automatic RMS normalization" and c["status"] == "PASS" for c in checks))

    def test_legacy_sampling_and_rms_metadata_remain_valid(self):
        self.fixture()
        del self.metadata["sampling"]["sampling_frequency_hz"]
        del self.metadata["normalization"]
        self.metadata["raw"] = self.metadata.pop("processed_original")
        self.metadata["simulation"] = {"rms": self.metadata["raw"]["rms"] * self.metadata["global_gain"]}
        self.write_metadata()
        self.assertFalse(any(c["status"] == "FAIL" for c in self.checks()))

    def test_frequency_and_dt_must_agree(self):
        self.fixture()
        self.metadata["sampling"]["sampling_frequency_hz"] *= 2
        self.write_metadata()
        self.assert_failure("sampling and time units")

    def test_declared_samples_and_nonzero_t0_are_rejected(self):
        self.fixture()
        original = copy.deepcopy(self.metadata)
        for field, value in (("samples", 38), ("samples", 37.5), ("t0_seconds", 1e-6), ("sound_speed_m_s", -1500)):
            with self.subTest(field=field, value=value):
                self.metadata = copy.deepcopy(original)
                self.metadata["sampling"][field] = value
                self.write_metadata()
                self.assert_failure("sampling and time units")

    def test_old_frame_gain_is_rejected(self):
        self.fixture()
        self.metadata["global_gain"] = .2612356995077034
        self.write_metadata()
        self.assert_failure("global_gain relationship")
        self.assert_failure("automatic RMS normalization")

    def test_target_rms_and_recorded_rms_are_checked(self):
        self.fixture()
        self.metadata["normalization"]["target_rms"] *= 1.1
        self.metadata["normalized"]["rms"] *= 1.2
        self.write_metadata()
        self.assert_failure("automatic RMS normalization")
        self.assert_failure("normalized metadata RMS")

    def test_prepared_dtype_and_coordinate_units_are_strict(self):
        raw = self.fixture()
        np.save(self.directory / "signal_original.npy", raw.astype(np.float64))
        self.metadata["coordinate_unit"] = "mm"
        self.write_metadata()
        self.assert_failure("signal_original")
        self.assert_failure("coordinate units declaration")

    def test_recorded_source_and_achieved_rms_are_checked(self):
        self.fixture()
        self.metadata["normalization"].update(source_rms=12345., achieved_rms=23456.)
        self.write_metadata()
        self.assert_failure("normalization source_rms")
        self.assert_failure("normalization achieved_rms")

    def test_fixed_field_of_view_is_retained(self):
        self.fixture()
        self.metadata["geometry"].update(voxel_origin_xyz_m=[-.00995] * 3, voxel_dims_xyz=[200] * 3, bounds_m=[-.01, .01] * 3)
        self.write_metadata()
        self.assert_failure("native physical grid")

    def test_bad_sensor_rows_and_overlapping_split(self):
        self.fixture()
        np.save(self.directory / "sensor_locations.npy", np.zeros((4, 3), dtype=np.float32))
        np.savez(self.directory / "split.npz", train=np.asarray([0, 1, 2, 3]), validation=np.asarray([3]))
        self.assert_failure("sensor/RF row correspondence")
        self.assert_failure("sensor split")

    def test_nonfinite_and_zero_signals(self):
        raw = self.fixture()
        raw[0, 0] = np.nan
        np.save(self.directory / "signal_original.npy", raw)
        self.assert_failure("signal_original")
        np.save(self.directory / "signal_original.npy", np.zeros_like(raw))
        np.save(self.directory / "signal_normalized.npy", np.zeros_like(raw))
        self.assert_failure("RF signal energy")


if __name__ == "__main__":
    unittest.main()
