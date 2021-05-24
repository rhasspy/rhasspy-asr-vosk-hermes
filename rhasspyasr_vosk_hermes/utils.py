"""Utility methods"""
import typing
from pathlib import Path


def find_model_dir(base_dir: Path, key_dir="graph") -> typing.Optional[Path]:
    """Recursively search a directory for a Vosk model"""
    if not base_dir.is_dir():
        return None

    for sub_dir in base_dir.iterdir():
        if sub_dir.name == key_dir:
            return base_dir

        maybe_model_dir = find_model_dir(sub_dir)
        if maybe_model_dir is not None:
            return maybe_model_dir

    return None
