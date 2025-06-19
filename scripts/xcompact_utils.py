import os
import re
from pathlib import Path

import xml.etree.ElementTree as ET

import numpy as np
from dask import delayed
import dask.array as da
import tqdm

from scripts.database_add import add_channel


def get_shape_from_xdmf(folder, snapshot_index):
    snapshot_path = folder / f"snapshot-{snapshot_index}.xdmf"
    tree = ET.parse(snapshot_path)
    root = tree.getroot()
    for elem in root.iter():
        if 'Dimensions' in elem.attrib:
            dims = tuple(map(int, elem.attrib['Dimensions'].split()))
            return dims  # (nz, ny, nx)
    raise ValueError("Could not find grid dimensions in XDMF.")


def load_snapshot(snapshot_index, dims, folder, mu_viscosity, y_0_1_distance, y_lim=None):
    nx, ny, nz = dims[::-1]  # because dims = (nz, ny, nx)
    shape = (nx, ny, nz)
    components = []
    for comp in ['ux', 'uy', 'uz']:
        filename = folder / f"{comp}-{snapshot_index}.bin"
        data = np.fromfile(filename, dtype=np.float64).reshape(shape, order='F')
        data = np.float32(data) # Convert to f32
        components.append(data)
    y_sample = np.stack(components, axis=0)  # Shape: [3, nx, ny, nz]

    if y_lim is not None:
        y_sample = y_sample[:, :, :y_lim, :]

    pressure_file = folder / f"pp-{snapshot_index}.bin"
    pressure_data = np.fromfile(pressure_file, dtype=np.float64).reshape(shape, order='F')
    pressure_data = np.float32(pressure_data)[:, 1, :]

    tau_x = mu_viscosity * (y_sample[0, :, 1, :] - y_sample[0, :, 0, :]) / y_0_1_distance
    tau_z = mu_viscosity * (y_sample[2, :, 1, :] - y_sample[2, :, 0, :]) / y_0_1_distance

    x_sample = np.stack([
        pressure_data[:, np.newaxis, :],
        tau_x[:, np.newaxis, :],
        tau_z[:, np.newaxis, :]
    ], axis=0)  # Final shape: (3, nx, 1, nz)

    return x_sample, y_sample

def get_ids(folder: Path):
    matching_ids = set()
    regex = re.compile(r'snapshot-(\d+).xdmf')
    for file_name in os.listdir(folder):
        match = regex.match(file_name)
        if match:
            file_id = int(match.group(1))
            matching_ids.add(file_id)
    return sorted(list(matching_ids))

def get_snapshot_xy(simulation_folder: Path, input_file_path=None, y_lim=None):
    folder = simulation_folder / "data"
    indices = get_ids(folder)
    dims = get_shape_from_xdmf(folder, indices[0])
    nx, ny, nz = dims[::-1]

    if input_file_path:
        input_file = Path(input_file_path)
    else:
        input_file = simulation_folder / "input.i3d"
    sim_data = get_simulation_data(input_file)
    mu_viscosity = sim_data["nu0nu"]
    ypi = read_ypi(simulation_folder)
    y_0_1_distance = ypi[1] - ypi[0]

    y_das = []
    x_das = []
    for idx in tqdm.tqdm(indices):
        delayed_yx = delayed(load_snapshot)(idx, dims, folder, mu_viscosity, y_0_1_distance, y_lim)
        x_da = da.from_delayed(delayed_yx[0], shape=(3, nx, 1, nz), dtype=np.float32)
        y_da = da.from_delayed(delayed_yx[1], shape=(3, nx, y_lim, nz), dtype=np.float32)
        y_das.append(y_da)
        x_das.append(x_da)

    y = da.stack(y_das, axis=0)  # Shape: [N, 3, nx, ny, nz]
    x = da.stack(x_das, axis=0)  # Shape: [N, 3, nx, 1,  nz]
    return x, y


def get_chunk_size(inner_shape, target_chunk_MB = 200):
    dtype = np.float32
    bytes_per_sample = np.prod(inner_shape) * np.dtype(dtype).itemsize
    target_chunk_bytes = target_chunk_MB * 1024 ** 2
    samples_per_chunk = target_chunk_bytes // bytes_per_sample

    return samples_per_chunk, *inner_shape


def read_ypi(folder):
    with open(folder / "ypi.dat", 'r') as f:
        return [float(line.strip()) for line in f.readlines()]

def get_simulation_data(filepath):
    simulation_data = {}
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
                simulation_data[key] = value
    return simulation_data


def add_channel_from_simulation(simulation_folder, channel_name, channel_scale, input_file_path=None):

    if input_file_path:
        input_file = Path(input_file_path)
    else:
        input_file = simulation_folder / "input.i3d"


    channel_data = get_simulation_data(input_file)
    y_dim = read_ypi(simulation_folder)

    channel = add_channel(
            channel_name,
            channel_data['nx'], channel_data['xlx'],
            y_dim,
            channel_data['nz'], channel_data['zlz'],
            channel_scale)
    return channel
