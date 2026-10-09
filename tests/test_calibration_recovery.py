"""CPU fixtures for safe completion of interrupted acoustic backups."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from slingbag.apply_final_response import (_complete_acoustic_backups,
                                           _validate_partial_backup_recovery)
from slingbag.fit_random_response import sha256


class CalibrationBackupRecovery(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.stage = self.folder / 'stage'
        self.stage.mkdir()
        self.pairs = {
            'metrics_acoustic.json': 'metrics.json',
            'checkpoint_acoustic.pt': 'checkpoint.pt',
            'predicted_signal_normalized_acoustic.npy': 'predicted_signal_normalized.npy',
            'predicted_signal_original_units_acoustic.npy': 'predicted_signal_original_units.npy',
        }
        config = {'all_sensors': True, 'random_only_ancestry': True,
                  'response_kernel': None, 'response_kernel_coefficients': None}
        record = {'step': 3, 'train_full_mse': 1.25, 'subset_is_trained': True}
        self.checkpoint = {'configuration': config, 'record': record,
                           'xyz': torch.zeros((1, 3)), 'pressure': torch.ones(1), 'radius': torch.ones(1)}
        self.metrics = {'configuration': copy.deepcopy(config), 'record': copy.deepcopy(record),
                        'all_sensors_trained': True}
        self.gain = .25
        self.acoustic = np.arange(12, dtype=np.float32).reshape(3, 4) / 3
        self.prediction = self.acoustic.copy()
        self.original = self.prediction / self.gain
        self._write_originals()
        shutil.copyfile(self.folder / 'metrics.json', self.folder / 'metrics_acoustic.json')

    def tearDown(self):
        self.temporary.cleanup()

    def _write_originals(self):
        torch.save(self.checkpoint, self.folder / 'checkpoint.pt')
        (self.folder / 'metrics.json').write_text(json.dumps(self.metrics))
        np.save(self.folder / 'predicted_signal_normalized.npy', self.prediction)
        np.save(self.folder / 'predicted_signal_original_units.npy', self.original)

    def _validate(self):
        return _validate_partial_backup_recovery(self.folder, self.pairs,
            self.checkpoint, self.metrics, self.acoustic, self.prediction, self.original, self.gain)

    def test_matching_partial_backups_are_completed_without_rewriting_existing(self):
        saved = self.folder / 'metrics_acoustic.json'
        before_hash, before_mtime = sha256(saved), saved.stat().st_mtime_ns
        self.assertEqual(self._validate(), ['metrics_acoustic.json'])
        _complete_acoustic_backups(self.folder, self.pairs, self.stage)
        for backup, original in self.pairs.items():
            self.assertEqual(sha256(self.folder / backup), sha256(self.folder / original))
        self.assertEqual(sha256(saved), before_hash)
        self.assertEqual(saved.stat().st_mtime_ns, before_mtime)

    def test_conflicting_backup_is_rejected_without_filling_missing_files(self):
        (self.folder / 'metrics_acoustic.json').write_text('conflicting partial backup')
        with self.assertRaises(RuntimeError):
            self._validate()
        with self.assertRaises(RuntimeError):
            _complete_acoustic_backups(self.folder, self.pairs, self.stage)
        self.assertFalse((self.folder / 'checkpoint_acoustic.pt').exists())

    def test_calibrated_checkpoint_is_rejected_even_with_matching_backup_bytes(self):
        self.checkpoint['configuration']['response_kernel_coefficients'] = [0., 1., 0.]
        self.metrics['configuration'] = copy.deepcopy(self.checkpoint['configuration'])
        self._write_originals()
        shutil.copyfile(self.folder / 'metrics.json', self.folder / 'metrics_acoustic.json')
        with self.assertRaisesRegex(ValueError, 'without a response kernel'):
            self._validate()

    def test_checkpoint_metrics_mismatch_is_rejected(self):
        self.metrics['record']['step'] = 4
        with self.assertRaisesRegex(ValueError, 'training record'):
            self._validate()

    def test_filtered_prediction_is_rejected(self):
        self.prediction = self.acoustic + .25
        self.original = self.prediction / self.gain
        with self.assertRaises(AssertionError):
            self._validate()

    def test_original_unit_mismatch_is_rejected(self):
        self.original = self.original + .25
        with self.assertRaises(AssertionError):
            self._validate()


if __name__ == '__main__':
    unittest.main()
