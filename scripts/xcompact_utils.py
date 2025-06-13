import os
import re
from pathlib import Path

import xml.etree.ElementTree as ET

import numpy as np
from dask import delayed
import dask.array as da
import tqdm

from scripts.database_add import add_dataset, add_channel
from space_exploration.dataset.dataset_stat import DatasetStats


def get_shape_from_xdmf(folder, snapshot_index):
    snapshot_path = folder / f"snapshot-{snapshot_index}.xdmf"
    tree = ET.parse(snapshot_path)
    root = tree.getroot()
    for elem in root.iter():
        if 'Dimensions' in elem.attrib:
            dims = tuple(map(int, elem.attrib['Dimensions'].split()))
            return dims  # (nz, ny, nx)
    raise ValueError("Could not find grid dimensions in XDMF.")


def load_snapshot(snapshot_index, dims, folder):
    nx, ny, nz = dims[::-1]  # because dims = (nz, ny, nx)
    shape = (nx, ny, nz)
    components = []
    for comp in ['ux', 'uy', 'uz']:
        filename = folder / f"{comp}-{snapshot_index}.bin"
        data = np.fromfile(filename, dtype=np.float64).reshape(shape, order='F')
        data = np.float32(data) # Convert to f32
        components.append(data)
    return None, np.stack(components, axis=0)  # Shape: [3, nx, ny, nz]

def get_ids(folder: Path):
    matching_ids = set()
    regex = re.compile(r'snapshot-(\d+).xdmf')
    for file_name in os.listdir(folder):
        match = regex.match(file_name)
        if match:
            file_id = int(match.group(1))
            matching_ids.add(file_id)
    return sorted(list(matching_ids))

def get_snapshot_xy(simulation_folder: Path):
    folder = simulation_folder / "data"
    indices = get_ids(folder)
    dims = get_shape_from_xdmf(folder, indices[0])
    nx, ny, nz = dims[::-1]

    y_das = []
    x_das = []
    for idx in tqdm.tqdm(indices):
        x, y = delayed(load_snapshot)(idx, dims, folder)
        y_da = da.from_delayed(y, shape=(3, nx, ny, nz))
        y_das.append(y_da)
        x_da = da.from_delayed(x, shape=(3, nx, 1, nz))
        x_das.append(x)

    y = da.stack(y_das, axis=0)  # Shape: [N, 3, nx, ny, nz]
    x = da.stack(x_das, axis=0)  # Shape: [N, 3, nx, 1,  nz]
    return None, y


def build_export_metadata(session, ds, s3_file_name, dataset_name, scaling, channel):

    stats = DatasetStats.from_ds(ds)

    add_dataset(
        session=session,
        name=dataset_name,
        s3_storage_name=s3_file_name,
        scaling=scaling,
        channel=channel,
        stats=stats,
    )


def read_ypi(folder):
    with open(folder / "ypi.dat", 'r') as f:
        return [float(line.strip()) for line in f.readlines()]

def get_channel_data(filepath):
    channel_data = {}
    with open(filepath, 'r') as file:
        for line in file:
            # Remove everything after '!' (comments)
            line = line.split('!')[0].strip()
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Try converting value to int, float, or keep as string
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                channel_data[key] = value
    return channel_data


def add_channel_from_simulation(session, simulation_folder, channel_name, channel_scale, input_file_path=None):

    if input_file_path:
        input_file = Path(input_file_path)
    else:
        input_file = simulation_folder / "input.i3d"


    channel_data = get_channel_data(input_file)
    y_dim = read_ypi(simulation_folder)

    channel = add_channel(session,
                channel_name,
                channel_data['nx'], channel_data['xlx'],
                y_dim,
                channel_data['nz'], channel_data['zlz'],
                channel_scale)
    return channel
