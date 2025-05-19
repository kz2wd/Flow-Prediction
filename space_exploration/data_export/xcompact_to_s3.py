import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tqdm

def get_shape_from_xdmf(folder, snapshot_index):
    snapshot_path = folder / f"snapshot-{snapshot_index}.xdmf"
    tree = ET.parse(snapshot_path)
    root = tree.getroot()
    for elem in root.iter():
        if 'Dimensions' in elem.attrib:
            dims = tuple(map(int, elem.attrib['Dimensions'].split()))
            return dims  # (nz, ny, nx)
    raise ValueError("Could not find grid dimensions in XDMF.")

import argparse, os

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def load_velocity_snapshot(snapshot_index, dims, folder, dtype=np.float64):
    nx, ny, nz = dims[::-1]  # because dims = (nz, ny, nx)
    shape = (nx, ny, nz)

    components = []
    for comp in ['ux', 'uy', 'uz']:
        filename = folder / f"{comp}-{snapshot_index}.bin"
        data = np.fromfile(filename, dtype=dtype).reshape(shape, order='F')
        components.append(data)

    return np.stack(components, axis=0)  # Shape: [3, nx, ny, nz]


def get_ids(folder: Path):
    matching_ids = set()
    regex = re.compile(r'snapshot-(\d+).xdmf')
    for file_name in os.listdir(folder):
        match = regex.match(file_name)
        if match:
            file_id = int(match.group(1))
            matching_ids.add(file_id)
    return sorted(list(matching_ids))


def load_all_snapshots(folder: Path, dtype=np.float64):
    data = []
    indices = get_ids(folder)
    dims = None

    for idx in tqdm.tqdm(indices):

        if dims is None:
            # dims_F for Fortran format (z, y, x)
            dims = get_shape_from_xdmf(folder, idx)

        snapshot = load_velocity_snapshot(idx, dims, folder, dtype)
        data.append(snapshot)
    return np.stack(data, axis=0)  # Shape: [N, 3, nx, ny, nz]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=dir_path)
    args = parser.parse_args(namespace=parser)
    main_folder = Path(args.path)
    dataset = load_all_snapshots(main_folder, dtype=np.float64)
    print(dataset.shape)
