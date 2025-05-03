import os

import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from FolderManager import FolderManager


def tf_parser(rec, root_folder):
    features = {
        'i_sample': tf.io.FixedLenFeature([], tf.int64),
        'nx': tf.io.FixedLenFeature([], tf.int64),
        'ny': tf.io.FixedLenFeature([], tf.int64),
        'nz': tf.io.FixedLenFeature([], tf.int64),
        'x': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'y': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'z': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'raw_u': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'raw_v': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'raw_w': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'raw_b_p': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'raw_b_tx': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'raw_b_tz': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'raw_t_p': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'raw_t_tx': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'raw_t_tz': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    }

    parsed_rec = tf.io.parse_single_example(rec, features)

    i_smp = tf.cast(parsed_rec['i_sample'], tf.int32)

    nx = tf.cast(parsed_rec['nx'], tf.int32)
    ny = tf.cast(parsed_rec['ny'], tf.int32)
    nz = tf.cast(parsed_rec['nz'], tf.int32)

    filename = f"{root_folder}tfrecords/scaling.npz"

    # Load mean velocity values in the streamwise and wall-normal directions for low- and high-resolution data

    U_mean = np.expand_dims(np.load(filename)['U_mean'], axis=-1)
    V_mean = np.expand_dims(np.load(filename)['V_mean'], axis=-1)
    W_mean = np.expand_dims(np.load(filename)['W_mean'], axis=-1)
    PB_mean = np.expand_dims(np.load(filename)['PB_mean'], axis=-1)
    PT_mean = np.expand_dims(np.load(filename)['PT_mean'], axis=-1)
    TBX_mean = np.expand_dims(np.load(filename)['TBX_mean'], axis=-1)
    TBZ_mean = np.expand_dims(np.load(filename)['TBZ_mean'], axis=-1)
    TTX_mean = np.expand_dims(np.load(filename)['TTX_mean'], axis=-1)
    TTZ_mean = np.expand_dims(np.load(filename)['TTZ_mean'], axis=-1)

    # Load standard deviation velocity values in the streamwise and wall-normal directions for low- and high-resolution data

    U_std = np.expand_dims(np.load(filename)['U_std'], axis=-1)
    V_std = np.expand_dims(np.load(filename)['V_std'], axis=-1)
    W_std = np.expand_dims(np.load(filename)['W_std'], axis=-1)
    PB_std = np.expand_dims(np.load(filename)['PB_std'], axis=-1)
    PT_std = np.expand_dims(np.load(filename)['PT_std'], axis=-1)
    TBX_std = np.expand_dims(np.load(filename)['TBX_std'], axis=-1)
    TBZ_std = np.expand_dims(np.load(filename)['TBZ_std'], axis=-1)
    TTX_std = np.expand_dims(np.load(filename)['TTX_std'], axis=-1)
    TTZ_std = np.expand_dims(np.load(filename)['TTZ_std'], axis=-1)

    # Reshape data into 2-dimensional matrix, substract mean value and divide by the standard deviation. Concatenate the streamwise and wall-normal velocities along the third dimension

    flow = (tf.reshape(parsed_rec['raw_u'], (nx, ny, nz, 1)) - U_mean) / U_std
    flow = tf.concat((flow, (tf.reshape(parsed_rec['raw_v'], (nx, ny, nz, 1)) - V_mean) / V_std), -1)
    flow = tf.concat((flow, (tf.reshape(parsed_rec['raw_w'], (nx, ny, nz, 1)) - W_mean) / W_std), -1)

    flow = tf.where(tf.math.is_nan(flow), tf.zeros_like(flow), flow)

    wall = (tf.reshape(parsed_rec['raw_b_p'], (nx, 1, nz, 1)) - PB_mean) / PB_std
    wall = tf.concat((wall, (tf.reshape(parsed_rec['raw_b_tx'], (nx, 1, nz, 1)) - TBX_mean) / TBX_std), -1)
    wall = tf.concat((wall, (tf.reshape(parsed_rec['raw_b_tz'], (nx, 1, nz, 1)) - TBZ_mean) / TBZ_std), -1)

    return wall, flow[:, 0:64, :, :]


def to_h5(target_folder, output_path):
    folder_path = FolderManager.tfrecords / target_folder
    record_files = sorted([
        str(os.path.join(folder_path, file))
        for file in os.listdir(folder_path)
        if file.endswith(".tfrecords")
    ])

    def build_dataset(file_list):
        if len(file_list) == 0:
            return None
        ds = tf.data.Dataset.from_tensor_slices(file_list)
        ds = ds.interleave(
            lambda x: tf.data.TFRecordDataset(x),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        ds = ds.map(
            lambda record: tf_parser(record, "./"),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
    dataset = build_dataset(record_files)
    x_data, y_data = [], []

    for i, (x, y) in tqdm(enumerate(dataset), total=100000):
        print(i)
        x_data.append(x.numpy())
        y_data.append(y.numpy())

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("x", data=x_data, compression="gzip")
        f.create_dataset("y", data=y_data, compression="gzip")

    print(f"Saved {len(x_data)} samples to {output_path}")


if "__main__" == __name__:
    to_h5("test", "dataset/test.hdf5")
