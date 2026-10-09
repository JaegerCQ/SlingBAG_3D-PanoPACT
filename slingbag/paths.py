"""Portable project, dataset, and generated-artifact paths.

Data may live outside the repository when SLINGBAG_DATA_DIR explicitly names
that directory. Generated artifacts remain inside outputs/ or examples/.
Relative CLI paths resolve against the project root, independent of cwd.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT
PACKAGE_ROOT = Path(__file__).resolve().parent


def resolve_input(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def dataset_root():
    """Evaluate the environment at call time so launchers may set it first."""
    value = os.environ.get("SLINGBAG_DATA_DIR")
    return resolve_input(value) if value else PROJECT_ROOT / "data" / "pose000"


def data_path(name=""):
    root = dataset_root().resolve()
    path = (root / Path(name)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Dataset path escapes {root}: {path}")
    return path


def is_artifact_path(path):
    path = resolve_input(path)
    # Keep these roots lexical: a symlink replacing outputs/ or examples/
    # must not silently authorize writes to a directory outside the project.
    return any(path.is_relative_to(PROJECT_ROOT / name) for name in ("outputs", "examples"))


def resolve_output(path):
    path = resolve_input(path)
    if not is_artifact_path(path):
        raise ValueError(f"Generated files must remain in {PROJECT_ROOT}/outputs or examples: {path}")
    return path


def response_path(path):
    """Location for a fitted FIR or FIR report; never an old-project kernel."""
    return resolve_output(path)


def display_path(path):
    """Project-relative provenance label, or explicit external dataset path."""
    path = Path(path).resolve()
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
