#!/usr/bin/env python3
"""Run the random-initialized reconstruction from a portable JSON configuration."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from slingbag.paths import PROJECT_ROOT as ROOT, resolve_input, resolve_output


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def flags(options):
    result = []
    for name, value in options.items():
        flag = '--' + name.replace('_', '-')
        if isinstance(value, bool):
            if value:
                result.append(flag)
        elif value is not None:
            result.append(flag)
            result.extend(map(str, value if isinstance(value, list) else [value]))
    return result


def make_plan(config, base):
    """Return ordered (stage name, argv, stable completion artifacts)."""
    stages = []
    data = dataset_directory(config, base)
    if config['schema_version'] == 2:
        stages.append(('prepare_data', [sys.executable, str(ROOT / 'scripts/prepare_data.py'),
            '--config', str(base / 'config.json'), '--output', str(data)],
            [data / name for name in ('signal_original.npy', 'signal_normalized.npy',
                'sensor_locations.npy', 'split.npz', 'split.json', 'metadata.json', 'preparation.json')]))

    def add(name, module, options, products):
        command = [sys.executable, '-u', '-m', 'slingbag.' + module, *flags(options)]
        stages.append((name, command, [base / p for p in products]))

    def train(name, initial=None, *, all_sensors=False):
        options = {**config['shared_training'], **config['training'][name],
                   'output': base / name, 'seed': config['seed']}
        if initial is None:
            options.update(config['initialization'])
        else:
            options['initial_checkpoint'] = base / initial
        if all_sensors:
            options['all_sensors'] = True
        if name == 'response':
            options['response_kernel'] = base / 'response_fit/kernel.npy'
        add(name, 'train_random', options, [name + '/selected.pt', name + '/summary.json'])

    def export(name, checkpoint, folder):
        add(name, 'export_random', {'checkpoint': base / checkpoint,
            'output': base / folder, 'mat': config.get('export_mat', False)},
            [folder + '/' + name for name in
             ('acoustic_signal_normalized.npy', 'point_cloud.ply', 'volume_zyx.npy')])

    train('coarse')
    add('split', 'split_random', {'checkpoint': base / 'coarse/selected.pt',
        'output': base / 'split/initial.pt', **config['split']}, ['split/initial.pt', 'split/initial.json'])
    train('fine', 'split/initial.pt')
    add('anneal', 'anneal_width', {'checkpoint': base / 'fine/selected.pt',
        'output': base / 'narrow_initial/initial.pt', **config['anneal']},
        ['narrow_initial/initial.pt', 'narrow_initial/initial.json'])
    train('narrow', 'narrow_initial/initial.pt')
    export('export_narrow', 'narrow/selected.pt', 'narrow_export')
    add('fit_response', 'fit_random_response', {
        'prediction': base / 'narrow_export/acoustic_signal_normalized.npy',
        'output_prefix': base / 'response_fit/kernel', **config['response_fit']},
        ['response_fit/kernel.npy', 'response_fit/kernel.json'])
    if config.get('response_diagnostic', True):
        train('response', 'narrow/selected.pt')
        export('export_validated', 'response/selected.pt', 'validated')
    # The final branch deliberately starts at the width-annealed proposal;
    # it trains all sensors without a response kernel, as in the source run.
    train('refit_all', 'narrow_initial/initial.pt', all_sensors=True)
    export('export_final', 'refit_all/selected.pt', 'final')
    add('calibrate_final', 'apply_final_response', {
        'export': base / 'final', 'ridge': config['final_response_ridge'],
        'half_width': config['response_fit']['half_width']},
        ['final/response_kernel.npy', 'final/checkpoint.pt', 'final/metrics.json',
         'final/predicted_signal_normalized.npy', 'final/response_calibration.json'])
    stages.append(('preview', [sys.executable, str(ROOT / 'scripts/preview.py'),
        '--volume', str(base / 'final/volume_zyx.npy'), '--metadata',
        str(data / 'metadata.json'),
        '--output', str(base / 'final/preview.png')], [base / 'final/preview.png']))
    return stages


def validate_config(config):
    required = {'schema_version', 'description', 'default_output', 'seed',
        'initialization', 'shared_training', 'training', 'split', 'anneal',
        'response_fit', 'final_response_ridge', 'response_diagnostic', 'export_mat'}
    if config.get('schema_version') == 2:
        required |= {'input', 'acquisition', 'normalization', 'sensor_split'}
    elif config.get('schema_version') == 1:
        required.add('data_dir')
    else:
        raise ValueError('Use schema_version=2 as in configs/pose000.json; legacy version 1 is also supported')
    if set(config) != required:
        raise ValueError(f'Config fields differ from the schema: missing={required - set(config)}, unknown={set(config) - required}')
    if isinstance(config['final_response_ridge'], bool) or not math.isfinite(config['final_response_ridge']) or config['final_response_ridge'] <= 0:
        raise ValueError('final_response_ridge must be finite and positive')
    if set(config['training']) != {'coarse', 'fine', 'narrow', 'response', 'refit_all'}:
        raise ValueError('Config must define the five named training stages')
    allowed = {'iterations', 'batch_sensors', 'field_grid', 'prior_weight', 'sparsity_weight',
        'curvature_weight', 'pressure_lr', 'adam_eps', 'position_lr', 'geometry_start',
        'max_shift_mm', 'eval_every'}
    for group in [config['shared_training'], *config['training'].values()]:
        if not set(group) <= allowed:
            raise ValueError(f'Unknown or reserved training keys: {set(group) - allowed}')
    for key, allowed in [('initialization', {'balls', 'sigma_mm', 'initial_pressure'}),
                         ('split', {'top', 'ratio'}),
                         ('anneal', {'fine_cutoff_mm', 'new_sigma_mm'}),
                         ('response_fit', {'half_width', 'ridge'})]:
        if set(config[key]) != allowed:
            raise ValueError(f'Invalid keys in {key}: expected {sorted(allowed)}')


def dataset_directory(config, base):
    return base / 'data' if config['schema_version'] == 2 else resolve_input(config['data_dir'])


def apply_overrides(config, args):
    changes = [(args.signal, 'input', 'signal'), (args.sensors, 'input', 'sensors'),
               (args.sound_speed, 'acquisition', 'sound_speed_m_s'),
               (args.sampling_frequency, 'acquisition', 'sampling_frequency_hz'),
               (args.coordinate_unit, 'input', 'coordinate_unit'),
               (args.signal_layout, 'input', 'signal_layout'),
               (args.target_rms, 'normalization', 'target_rms')]
    if config['schema_version'] != 2 and (args.samples is not None or any(v is not None for v, _, _ in changes)):
        raise ValueError('Input/acquisition overrides require a version 2 config')
    for value, section, key in changes:
        if value is not None:
            config[section][key] = value
    if args.signal is not None:
        config['input']['frame'] = Path(args.signal).stem
    if args.samples is not None:
        config['acquisition']['samples'] = None if args.samples.lower() == 'auto' else int(args.samples)
    if args.seed is not None:
        config['seed'] = args.seed


def verify_run_config(base, expected):
    """Preparation reads this snapshot, so it must match the resumed identity."""
    path = base / 'config.json'
    if not path.is_file() or not path.resolve().is_relative_to(base):
        raise ValueError('The run config.json snapshot is missing or outside the run directory')
    if json.loads(path.read_text()) != expected:
        raise ValueError('The saved run config.json differs from the requested config. Choose a new output directory.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pose000.json', help='Relative paths resolve from the repository root')
    parser.add_argument('--output', help='New directory under outputs/; defaults to the config value')
    parser.add_argument('--seed', type=int, help='Override the random seed for a new experiment')
    parser.add_argument('--signal', help='Input signal file; version 2 config override')
    parser.add_argument('--sensors', help='Input sensor-coordinate file; version 2 config override')
    parser.add_argument('--sound-speed', type=float, help='Sound speed in metres/second')
    parser.add_argument('--sampling-frequency', type=float, help='Sampling frequency in Hz, e.g. 40000000')
    parser.add_argument('--samples', help='Expected time samples, or auto to infer from the signal; does not crop/resample')
    parser.add_argument('--coordinate-unit', choices=('m', 'mm', 'cm'), help='Unit of the input sensor coordinates')
    parser.add_argument('--signal-layout', choices=('sensor_time', 'time_sensor'), help='Axis order in the input RF matrix')
    parser.add_argument('--target-rms', type=float, help='Target global RF RMS; gain is recomputed from this input frame')
    parser.add_argument('--prepare-only', action='store_true', help='Prepare and validate data without starting GPU training')
    parser.add_argument('--resume', action='store_true', help='Skip verified complete stages; restart an interrupted stage')
    parser.add_argument('--dry-run', action='store_true', help='Print all stage commands without writing or running them')
    args = parser.parse_args()
    config = json.loads(resolve_input(args.config).read_text())
    validate_config(config)
    apply_overrides(config, args)
    validate_config(config)
    base = resolve_output(args.output or config['default_output'])
    if not base.is_relative_to(ROOT / 'outputs') or base == ROOT / 'outputs':
        raise ValueError('Use a run subdirectory inside outputs/')
    data = dataset_directory(config, base)
    if args.prepare_only and config['schema_version'] != 2:
        raise ValueError('--prepare-only requires schema_version=2')
    if config['schema_version'] == 1:
        from slingbag.dataset import load_dataset
        load_dataset(data)
    plan = make_plan(config, base)
    if args.prepare_only:
        plan = plan[:1]
    if args.dry_run:
        print(config['description'])
        for name, command, _ in plan:
            print(f'[{name}] {shlex.join(command)}')
        return

    env = os.environ.copy()
    env['SLINGBAG_DATA_DIR'] = str(data)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.setdefault('MAX_JOBS', '2')
    env.setdefault('OMP_NUM_THREADS', '4')
    env.setdefault('OPENBLAS_NUM_THREADS', '4')
    for key, directory in {'TORCH_EXTENSIONS_DIR': '.build', 'CUDA_CACHE_PATH': '.cache/cuda',
                           'MPLCONFIGDIR': '.cache/matplotlib', 'TRITON_CACHE_DIR': '.cache/triton'}.items():
        env[key] = str(ROOT / directory)
    code_files = [Path(__file__), ROOT / 'scripts/preview.py', ROOT / 'scripts/_script_common.py', ROOT / 'scripts/prepare_data.py',
                  *sorted((ROOT / 'slingbag').rglob('*.py')), *sorted((ROOT / 'slingbag/csrc').glob('*'))]
    if config['schema_version'] == 2:
        input_paths = {'signal': resolve_input(config['input']['signal']),
                       'sensors': resolve_input(config['input']['sensors'])}
        if config['sensor_split'].get('file'):
            input_paths['split'] = resolve_input(config['sensor_split']['file'])
        inputs = {name: sha256(path) for name, path in input_paths.items()}
    else:
        inputs = {name: sha256(data / name) for name in
                  ('signal_original.npy', 'signal_normalized.npy', 'sensor_locations.npy', 'split.npz', 'metadata.json')}
    identity = {'config': config, 'data_sha256': inputs,
                'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in code_files if p.is_file()}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    state_path = base / 'run.json'
    if args.resume:
        if not state_path.is_file():
            raise ValueError('--resume requires an existing run.json; omit it for a new run')
        state = json.loads(state_path.read_text())
        if state['fingerprint'] != fingerprint:
            raise ValueError('Config, code or input data changed. Choose a new --output directory.')
        verify_run_config(base, config)
        for name, _, products in plan:
            if name not in state['completed']:
                continue
            if any(not p.is_file() or sha256(p) != state['completed'][name]['sha256'].get(str(p.relative_to(base))) for p in products):
                raise ValueError(f'Completed stage {name} has missing/changed files. Choose a new output directory.')
    else:
        if base.exists() and any(base.iterdir()):
            raise ValueError('Output directory is not empty. Use --resume or a different --output.')
        base.mkdir(parents=True, exist_ok=True)
        state = {**identity, 'fingerprint': fingerprint, 'completed': {},
                 'created_utc': datetime.now(timezone.utc).isoformat(),
                 'resume_scope': 'completed stages only; an interrupted stage restarts from its initial input'}
        write_json(state_path, state)
        write_json(base / 'config.json', config)
    (base / 'logs').mkdir(exist_ok=True)
    print(config['description'], flush=True)
    for name, command, products in plan:
        if name in state['completed']:
            print(f'[{name}] verified; skipping', flush=True)
            continue
        print(f'[{name}] {shlex.join(command)}', flush=True)
        started = time.perf_counter()
        with (base / 'logs' / f'{name}.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                returncode = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
        if returncode:
            raise RuntimeError(f'{name} failed (exit {returncode}). See {base / "logs" / (name + ".log")}')
        if name == 'prepare_data':
            preparation = json.loads((data / 'preparation.json').read_text())
            prepared_sources = {key: value['sha256'] for key, value in preparation['source_files'].items()}
            if prepared_sources != inputs:
                raise RuntimeError('Input files changed between run creation and preparation. Choose a new output directory.')
        state['completed'][name] = {'seconds': time.perf_counter() - started,
            'sha256': {str(p.relative_to(base)): sha256(p) for p in products}}
        write_json(state_path, state)
    if args.prepare_only:
        print(f'Data prepared: {data}. Continue this run with --resume and without --prepare-only.', flush=True)
    else:
        print(f'Complete: {base / "final"}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(1)
