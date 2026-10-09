#!/usr/bin/env python3
"""Prepare a single input frame and automatically normalize its global RF RMS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slingbag.paths import resolve_input, resolve_output, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pose000.json', help='Version 2 reconstruction config')
    parser.add_argument('--output', default='outputs/prepared', help='New or matching prepared directory inside outputs/')
    args = parser.parse_args()
    config = json.loads(resolve_input(args.config).read_text())
    if config.get('schema_version') != 2:
        raise ValueError('Data preparation requires schema_version=2')
    destination = resolve_output(args.output)
    if not destination.is_relative_to(PROJECT_ROOT / 'outputs') or destination == PROJECT_ROOT / 'outputs':
        raise ValueError('Data must be prepared in a subdirectory of outputs/')
    from slingbag.dataset import prepare_dataset
    metadata = prepare_dataset(config, destination)
    print(json.dumps({'data_directory': str(destination), 'frame': metadata.get('frame'),
        'sampling': metadata['sampling'], 'global_gain': metadata['global_gain'],
        'normalization': metadata.get('normalization'),
        'signal_shape': metadata['normalized']['shape'], 'signal_rms': metadata['normalized']['rms']},
        ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(1)
