import argparse
import os
from pathlib import Path


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def zarr_path(string):
    path = Path(string)
    if path.is_dir() and (path / ".zarray").exists() or any(path.glob("*.zgroup")):
        return path
    raise argparse.ArgumentTypeError(f"{string} is not a valid Zarr directory")
