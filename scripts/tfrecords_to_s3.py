import os
import re

import numpy as np
import dask.array as da

from space_exploration.FolderManager import FolderManager

import tensorflow as tf

from space_exploration.beans.dataset_bean import Dataset
from space_exploration.dataset import s3_access, db_access
from space_exploration.dataset.ds_helper import y_along_component_denormalize


def main():
    # dataset_train, dataset_valid = generate_pipeline_training(root_folder, batch_size=1)
    dataset_valid = generate_dataset()

    nx = 64
    ny = 64
    nz = 64
    # dataset_y is of shape N, x, y, z, 3
    dataset_y = []
    dataset_x = []
    for x, y in dataset_valid:
        dataset_y.append(y)
        dataset_x.append(x)

    # Switch to shape N, 3, x, y, z
    Y = np.float32(np.array(dataset_y))

    Y = np.transpose(Y, (0, 4, 1, 2, 3))

    chunk_size = 50

    y_ds = da.from_array(Y, chunks=(chunk_size, 3, nx, ny, nz))

    X = np.float32(np.array(dataset_x))
    X = np.transpose(X, (0, 4, 1, 2, 3))
    x_ds = da.from_array(X, chunks=(chunk_size, 3, nx, ny, nz))

    session = db_access.get_session()
    dataset = Dataset.get_dataset_or_fail(session, "paper-train")
    stats = dataset.get_stats()
    y_ds = y_along_component_denormalize(y_ds, stats)

    s3_access.store_xy(x_ds, y_ds, "simulations/paper-train.zarr")
    print("✅ Exported paper dataset")


def generate_dataset():
    # Step 1: Gather all TFRecord files
    target_folder = "train"
    folder_path = FolderManager.tfrecords / target_folder

    record_files = sorted([

        str(os.path.join(folder_path, file))

        for file in os.listdir(folder_path)

        if file.endswith(".tfrecords")

    ])
    # for i, file in enumerate(record_files):
    #     try:
    #         filenames = [file]
    #         raw_dataset = tf.data.TFRecordDataset(filenames)
    #         raw_dataset = raw_dataset.map(tf_parser)
    #         for record in raw_dataset.take(10):
    #             print(record)
    #             input(">>>>")
    #         print(f"{i} ok")
    #     except Exception as e:
    #         print(f"{i} failed")
    #         print(e)


    if not record_files:
        raise FileNotFoundError(f"No TFRecord files found in: {folder_path}")

    # Step 2: Split the file list

    np.random.seed(0)

    def build_dataset(file_list):
        if len(file_list) == 0:
            return None


        def is_valid(x):
            return tf.size(x) > 0


        ds = tf.data.Dataset.from_tensor_slices(file_list)

        ds = ds.interleave(
            lambda x: tf.data.TFRecordDataset(x),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE

        )

        ds = ds.filter(is_valid)

        ds = ds.map(
            lambda record: tf_parser(record),
            num_parallel_calls=tf.data.AUTOTUNE
        )


        return ds

    return build_dataset(record_files)


def tf_parser(rec):
    try:
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

        filename = FolderManager.tfrecords / "scaling.npz"

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
    except Exception as e:
        print("Issue with record", rec)


if __name__ == '__main__':

    main()
