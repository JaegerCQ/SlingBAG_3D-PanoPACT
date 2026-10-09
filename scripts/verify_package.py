#!/usr/bin/env python3
"""Verify distributed files against the payload and source manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', action='append', help='Optional manifest; repeat to check several')
    args = parser.parse_args()
    names = args.manifest or ['provenance/portable_manifest.json', 'provenance/source_manifest.json']
    errors = []
    checked = 0
    for name in names:
        manifest = json.loads((ROOT / name).read_text())
        for entry in manifest['files']:
            checked += 1
            path = (ROOT / entry['path']).resolve()
            if not path.is_relative_to(ROOT):
                errors.append(f"Path escapes repository: {entry['path']}")
            elif not path.is_file():
                errors.append(f"Missing file: {entry['path']} (if cloned with Git LFS, run git lfs pull)")
            elif path.stat().st_size != entry['size_bytes'] or sha256(path) != entry['sha256']:
                errors.append(f"Size/SHA256 mismatch: {entry['path']} (check for a Git LFS pointer)")
    if errors:
        for error in errors:
            print('FAIL:', error, file=sys.stderr)
        return 1
    print(f"PASS: {checked} files verified without accessing historical source folders.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
