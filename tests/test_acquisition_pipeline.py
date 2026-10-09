"""CPU integration checks for metadata-driven sampling and arbitrary splits."""
from contextlib import ExitStack, redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from slingbag import paths, fit_random_response, apply_final_response
from slingbag.dataset import load_dataset
from slingbag.export_random import predict_acoustic, validate_checkpoint_dataset
from slingbag.rapid_forward import gaussian_forward
from slingbag.train_random import predict_signal


class AcquisitionPipeline(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        self.stack = ExitStack()
        self.stack.enter_context(patch.dict(os.environ, {'SLINGBAG_DATA_DIR': str(self.data)}))
        self.stack.enter_context(patch.object(paths, 'PROJECT_ROOT', self.root))
        self.stack.enter_context(patch.object(paths, 'ROOT', self.root))
        self.stack.enter_context(patch.object(fit_random_response, 'ROOT', self.root))
        self.stack.enter_context(patch.object(apply_final_response, 'ROOT', self.root))
        self.write_dataset()

    def tearDown(self):
        self.stack.close()
        self.temporary.cleanup()

    def write_dataset(self, samples=64, dt=50e-9, sound_speed=1480., include_fs=True):
        rng = np.random.default_rng(71)
        original = rng.normal(size=(6, samples)).astype(np.float32)
        normalized = original * np.float32(.25)
        sensors = rng.uniform(-.003, .003, size=(6, 3)).astype(np.float32)
        np.save(self.data / 'signal_original.npy', original)
        np.save(self.data / 'signal_normalized.npy', normalized)
        np.save(self.data / 'sensor_locations.npy', sensors)
        np.savez(self.data / 'split.npz', train=np.array([0, 1, 3, 4]), validation=np.array([2, 5]))
        sampling = {'dt_seconds': dt, 'samples': samples, 'sound_speed_m_s': sound_speed, 't0_seconds': 0.}
        if include_fs:
            sampling['sampling_frequency_hz'] = 1. / dt
        self.metadata = {
            'global_gain': .25, 'coordinate_unit': 'm', 'sampling': sampling,
            'normalization': {'target_rms': float(np.sqrt(np.mean(normalized.astype(np.float64) ** 2)))},
            'geometry': {'bounds_m': [-.02, .02] * 3, 'voxel_dims_xyz': [400] * 3,
                         'voxel_spacing_m': .0001, 'voxel_origin_xyz_m': [-.01995] * 3},
        }
        (self.data / 'metadata.json').write_text(json.dumps(self.metadata))

    @staticmethod
    def parameters():
        return (torch.tensor([[.00025, -.00015, .00020]], dtype=torch.float64),
                torch.tensor([1.2], dtype=torch.float64),
                torch.tensor([.00035], dtype=torch.float64))

    def test_train_and_export_use_nondefault_acquisition(self):
        dataset = load_dataset()
        sensors = torch.from_numpy(dataset.sensors).double()
        parameters = self.parameters()
        expected = gaussian_forward(sensors, *parameters, dt=50e-9, num_times=64,
                                    sound_speed=1480., backend='torch')
        training = predict_signal(sensors, *parameters, dataset.forward_kwargs)
        exported = predict_acoustic(sensors, *parameters, dataset.forward_kwargs)
        self.assertEqual(training.shape, (6, 64))
        torch.testing.assert_close(training, expected, atol=0., rtol=0.)
        torch.testing.assert_close(exported, expected, atol=0., rtol=0.)
        wrong_speed = gaussian_forward(sensors, *parameters, dt=50e-9, num_times=64,
                                       sound_speed=1500., backend='torch')
        self.assertFalse(torch.equal(training, wrong_speed))

    def test_legacy_sampling_matches_original_forward_defaults(self):
        self.write_dataset(samples=4096, dt=25e-9, sound_speed=1500., include_fs=False)
        dataset = load_dataset()
        sensors = torch.from_numpy(dataset.sensors[:2]).double()
        parameters = self.parameters()
        expected = gaussian_forward(sensors, *parameters, backend='torch')
        actual = predict_signal(sensors, *parameters, dataset.forward_kwargs)
        torch.testing.assert_close(actual, expected, atol=0., rtol=0.)

    def test_checkpoint_sampling_and_data_consistency(self):
        dataset = load_dataset()
        acquisition = {**dataset.forward_kwargs, 'num_sensors': dataset.num_sensors,
                       't0_seconds': 0., 'global_gain': dataset.gain}
        configuration = {'acquisition': acquisition, 'input_files': {
            'old/location/sensor_locations.npy': hashlib.sha256((self.data / 'sensor_locations.npy').read_bytes()).hexdigest()}}
        validate_checkpoint_dataset(configuration, dataset)
        validate_checkpoint_dataset({}, dataset)  # Old checkpoints have no sampling record.
        changed = copy.deepcopy(configuration)
        changed['acquisition']['sound_speed'] = 1500.
        with self.assertRaisesRegex(ValueError, 'sound_speed'):
            validate_checkpoint_dataset(changed, dataset)
        changed['acquisition'] = acquisition
        changed['input_files']['old/location/sensor_locations.npy'] = 'wrong hash'
        with self.assertRaisesRegex(ValueError, 'sensor_locations'):
            validate_checkpoint_dataset(changed, dataset)

    def test_fir_fit_uses_dynamic_split_and_time_units(self):
        dataset = load_dataset()
        folder = self.root / 'outputs' / 'fir'
        folder.mkdir(parents=True)
        prediction_path = folder / 'acoustic.npy'
        np.save(prediction_path, dataset.signal * np.float32(.9))
        prefix = folder / 'kernel'
        argv = ['fit_random_response', '--prediction', str(prediction_path),
                '--output-prefix', str(prefix), '--half-width', '3', '--ridge', '1']
        with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
            fit_random_response.main()
        report = json.loads(prefix.with_suffix('.json').read_text())
        self.assertEqual(report['signal_shape'], [6, 64])
        self.assertEqual((report['training_sensor_count'], report['validation_sensor_count']), (4, 2))
        self.assertEqual(report['time_sample_seconds'], 50e-9)
        np.testing.assert_allclose(report['lag_support_ns'], [-150., 150.], rtol=0., atol=1e-12)
        self.assertEqual(np.load(prefix.with_suffix('.npy')).shape, (7,))

    def test_final_calibration_uses_dynamic_acquisition_and_is_idempotent(self):
        dataset = load_dataset()
        folder = self.root / 'outputs' / 'final'
        folder.mkdir(parents=True)
        acoustic = dataset.signal * np.float32(.9)
        configuration = {'all_sensors': True, 'random_only_ancestry': True,
                         'response_kernel': None, 'response_kernel_coefficients': None}
        groups = apply_final_response.metric_groups(acoustic.astype(np.float64),
            dataset.signal.astype(np.float64), dataset.train, dataset.validation)
        record = {'step': 1, 'train_full_mse': groups['full']['mse'], 'subset_is_trained': True}
        checkpoint = {'configuration': configuration, 'record': record,
                      'xyz': torch.zeros((1, 3)), 'pressure': torch.ones(1), 'radius': torch.full((1,), .00035)}
        torch.save(checkpoint, folder / 'checkpoint.pt')
        (folder / 'metrics.json').write_text(json.dumps({
            'configuration': configuration, 'record': record, 'all_sensors_trained': True, **groups}))
        np.save(folder / 'acoustic_signal_normalized.npy', acoustic)
        np.save(folder / 'predicted_signal_normalized.npy', acoustic)
        np.save(folder / 'predicted_signal_original_units.npy', acoustic / dataset.gain)
        (folder / 'point_cloud.ply').write_bytes(b'unchanged fixture cloud')
        np.save(folder / 'volume_zyx.npy', np.ones((2, 2, 2), dtype=np.float32))
        immutable = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                     for name in ('point_cloud.ply', 'volume_zyx.npy')}
        argv = ['apply_final_response', '--export', str(folder), '--ridge', '2.5', '--half-width', '3']
        with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
            apply_final_response.main()
        first = np.load(folder / 'predicted_signal_normalized.npy')
        metrics = json.loads((folder / 'metrics.json').read_text())
        calibration = metrics['response_calibration']
        self.assertEqual(calibration['sensor_count'], 6)
        self.assertEqual(calibration['time_sample_seconds'], 50e-9)
        self.assertEqual(calibration['ridge_relative_strength'], 2.5)
        self.assertEqual(calibration['lags_samples'], list(range(-3, 4)))
        self.assertEqual(metrics['configuration']['acquisition']['num_times'], 64)
        self.assertIn('NOT_validation', metrics['original_validation_subset']['role'])
        with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
            apply_final_response.main()
        np.testing.assert_array_equal(first, np.load(folder / 'predicted_signal_normalized.npy'))
        for name, digest in immutable.items():
            self.assertEqual(digest, hashlib.sha256((folder / name).read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
