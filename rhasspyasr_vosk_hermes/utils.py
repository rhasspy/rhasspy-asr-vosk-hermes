"""Utility methods"""
import typing
from pathlib import Path


def find_model_dir(
    base_dir: Path, key_dirs=("graph", "ivector"), key_files=("AUTHORS", "LICENSE")
) -> typing.Optional[Path]:
    """Recursively search a directory for a Vosk model"""
    if not base_dir.is_dir():
        return None

    for item in base_dir.iterdir():
        if item.is_dir():
            if item.name in key_dirs:
                return base_dir
            else:
                maybe_model_dir = find_model_dir(item)
                if maybe_model_dir is not None:
                    return maybe_model_dir
        elif item.is_file():
            if item.name in key_files:
                return base_dir

    return None
