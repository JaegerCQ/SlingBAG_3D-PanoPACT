"""Regression checks for the user-facing acquisition configuration and pipeline."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruct import ROOT, apply_overrides, dataset_directory, make_plan, validate_config, verify_run_config


class RunConfiguration(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'configs/smoke.json').read_text())
        self.base = ROOT / 'outputs/config_test'

    def test_preparation_and_all_stages_use_run_dataset(self):
        validate_config(self.config)
        plan = make_plan(self.config, self.base)
        self.assertEqual(len(plan), 14)
        self.assertEqual(plan[0][0], 'prepare_data')
        self.assertIn(str(self.base / 'data'), plan[0][1])
        self.assertEqual(dataset_directory(self.config, self.base), self.base / 'data')
        self.assertIn(str(self.base / 'data/metadata.json'), plan[-1][1])
        self.assertIn(self.base / 'data/preparation.json', plan[0][2])

    def test_cli_overrides_sampling_and_automatic_sample_count(self):
        args = argparse.Namespace(signal='inputs/frame_a.npz', sensors='inputs/probes.csv',
            sound_speed=1480., sampling_frequency=20e6, samples='auto', coordinate_unit='mm',
            signal_layout='time_sensor', target_rms=12., seed=100)
        apply_overrides(self.config, args)
        validate_config(self.config)
        self.assertEqual(self.config['input']['frame'], 'frame_a')
        self.assertEqual(self.config['input']['coordinate_unit'], 'mm')
        self.assertIsNone(self.config['acquisition']['samples'])
        self.assertEqual(self.config['acquisition']['sampling_frequency_hz'], 20e6)
        self.assertEqual(self.config['acquisition']['sound_speed_m_s'], 1480.)
        self.assertEqual(self.config['normalization']['target_rms'], 12.)

    def test_response_width_is_consistent_in_validation_and_final(self):
        self.config['response_fit']['half_width'] = 7
        self.config['final_response_ridge'] = 3.
        validate_config(self.config)
        commands = {name: command for name, command, _ in make_plan(self.config, self.base)}
        for stage in ('fit_response', 'calibrate_final'):
            command = commands[stage]
            self.assertEqual(command[command.index('--half-width') + 1], '7')
        self.assertEqual(commands['calibrate_final'][commands['calibrate_final'].index('--ridge') + 1], '3.0')

    def test_legacy_prepared_dataset_has_no_preparation_stage(self):
        legacy = copy.deepcopy(self.config)
        legacy['schema_version'] = 1
        for key in ('input', 'acquisition', 'normalization', 'sensor_split'):
            del legacy[key]
        legacy['data_dir'] = 'data/pose000'
        validate_config(legacy)
        self.assertEqual(dataset_directory(legacy, self.base), ROOT / 'data/pose000')
        self.assertEqual(make_plan(legacy, self.base)[0][0], 'coarse')

    def test_resuming_rejects_changed_or_missing_preparation_config(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            with self.assertRaises(ValueError):
                verify_run_config(base, self.config)
            (base / 'config.json').write_text(json.dumps(self.config))
            verify_run_config(base, self.config)
            changed = copy.deepcopy(self.config)
            changed['acquisition']['sound_speed_m_s'] += 10
            (base / 'config.json').write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                verify_run_config(base, self.config)


if __name__ == '__main__':
    unittest.main()
