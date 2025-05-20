import re
import xml.etree.ElementTree as ET
from pathlib import Path
import os
import argparse

import numpy as np
import tqdm
import dask.array as da
from dask import delayed
from dask.diagnostics import ProgressBar
import s3fs

from space_exploration.data_export import s3_access


def get_shape_from_xdmf(folder, snapshot_index):
    snapshot_path = folder / f"snapshot-{snapshot_index}.xdmf"
    tree = ET.parse(snapshot_path)
    root = tree.getroot()
    for elem in root.iter():
        if 'Dimensions' in elem.attrib:
            dims = tuple(map(int, elem.attrib['Dimensions'].split()))
            return dims  # (nz, ny, nx)
    raise ValueError("Could not find grid dimensions in XDMF.")

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

def load_all_snapshots_zarr(folder: Path, zarr_store_path: str, s3=False):
    indices = get_ids(folder)
    dims = get_shape_from_xdmf(folder, indices[0])
    nx, ny, nz = dims[::-1]

    delayed_arrays = []
    for idx in tqdm.tqdm(indices):
        arr = delayed(load_velocity_snapshot)(idx, dims, folder)
        darr = da.from_delayed(arr, shape=(3, nx, ny, nz), dtype=np.float64)
        delayed_arrays.append(darr)

    dset = da.stack(delayed_arrays, axis=0)  # Shape: [N, 3, nx, ny, nz]

    store = zarr_store_path if not s3 else s3_access.get_s3_map(zarr_store_path)
    dset.to_zarr(store, overwrite=True)
    return dset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=dir_path, required=True)
    parser.add_argument('--zarr_store', type=str, required=True, help='Local path or s3://bucket/key')
    parser.add_argument('--use_s3', action='store_true')
    args = parser.parse_args()

    main_folder = Path(args.path)

    with ProgressBar():
        dset = load_all_snapshots_zarr(main_folder, args.zarr_store, s3=args.use_s3)

    print("Saved dataset to Zarr:", args.zarr_store)
    print("Shape:", dset.shape)
    print("Mean:", dset.mean().compute())  # print mean as a way to show user that it is not filled with NaN.
