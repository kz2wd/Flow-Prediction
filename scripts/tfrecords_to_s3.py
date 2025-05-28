import os
import re

import numpy as np

from space_exploration.FolderManager import FolderManager

import tensorflow as tf

def main():
    # dataset_train, dataset_valid = generate_pipeline_training(root_folder, batch_size=1)
    dataset_valid = generate_pipeline_training(batch_size=1)

    itr = iter(dataset_valid)

    nx = 64
    ny = 64
    nz = 64

    n_samp = 100
    x_target = np.zeros((n_samp, nx, 1, nz, 3), np.float32)
    y_target = np.zeros((n_samp, nx, ny, nz, 3), np.float32)
    y_predic = np.zeros((n_samp, nx, ny, nz, 3), np.float32)

    for idx in range(n_samp):
        if float.is_integer(idx / 50):
            print(idx)
            # print(idx)
        x, y = next(itr)

        x_target[idx] = x.numpy()
        y_target[idx] = y.numpy()


    # dataset_y = [y for x, y in dataset_valid]
    # print(dataset_y)
    # print(len(dataset_y))



def generate_pipeline_training(validation_split=0.2, shuffle_buffer=200, batch_size=4, n_prefetch=8):
    tfr_path = str(FolderManager.tfrecords / "test")
    tfr_files = sorted(
        [os.path.join(tfr_path, f) for f in os.listdir(tfr_path) if os.path.isfile(os.path.join(tfr_path, f))])
    regex = re.compile(f'.tfrecords')
    tfr_files = ([string for string in tfr_files if re.search(regex, string)])

    n_samples_per_tfr = np.array([int(s.split('.')[-2][-3:]) for s in tfr_files])
    n_samples_per_tfr = n_samples_per_tfr[np.argsort(-n_samples_per_tfr)]
    cumulative_samples_per_tfr = np.cumsum(np.array(n_samples_per_tfr))
    tot_samples_per_ds = sum(n_samples_per_tfr)
    n_tfr_loaded_per_ds = int(tfr_files[0].split('_')[-3][-3:])

    tfr_files = [string for string in tfr_files if int(string.split('_')[-3][:3]) <= n_tfr_loaded_per_ds]

    n_samp_train = int(sum(n_samples_per_tfr) * (1 - validation_split))
    n_samp_valid = sum(n_samples_per_tfr) - n_samp_train

    (n_files_train, samples_train_left) = np.divmod(n_samp_train, n_samples_per_tfr[0])

    if samples_train_left > 0:
        n_files_train += 1

    tfr_files_train = [string for string in tfr_files if int(string.split('_')[-3][:3]) <= n_files_train]
    n_tfr_left = np.sum(np.where(cumulative_samples_per_tfr < samples_train_left, 1, 0)) + 1

    if sum([int(s.split('.')[-2][-2:]) for s in tfr_files_train]) != n_samp_train:

        shared_tfr = tfr_files_train[-1]
        tfr_files_valid = [shared_tfr]
    else:

        shared_tfr = ''
        tfr_files_valid = list()

    tfr_files_valid.extend([string for string in tfr_files if string not in tfr_files_train])
    tfr_files_valid = sorted(tfr_files_valid)

    shared_tfr_out = tf.constant(shared_tfr)
    n_tfr_per_ds = tf.constant(n_tfr_loaded_per_ds)
    n_samples_loaded_per_tfr = list()

    if n_tfr_loaded_per_ds > 1:

        n_samples_loaded_per_tfr.extend(n_samples_per_tfr[:n_tfr_loaded_per_ds - 1])
        n_samples_loaded_per_tfr.append(tot_samples_per_ds - cumulative_samples_per_tfr[n_tfr_loaded_per_ds - 2])

    else:

        n_samples_loaded_per_tfr.append(tot_samples_per_ds)

    n_samples_loaded_per_tfr = np.array(n_samples_loaded_per_tfr)

    tfr_files_train_ds = tf.data.Dataset.list_files(tfr_files_train, seed=666)
    tfr_files_val_ds = tf.data.Dataset.list_files(tfr_files_valid, seed=686)

    if n_tfr_left > 1:

        samples_train_shared = samples_train_left - cumulative_samples_per_tfr[n_tfr_left - 2]
        n_samples_tfr_shared = n_samples_loaded_per_tfr[n_tfr_left - 1]

    else:

        samples_train_shared = samples_train_left
        n_samples_tfr_shared = n_samples_loaded_per_tfr[0]

    tfr_files_train_ds = tfr_files_train_ds.interleave(
        lambda x: tf.data.TFRecordDataset(x).take(samples_train_shared) if tf.math.equal(x,
                                                                                         shared_tfr_out) else tf.data.TFRecordDataset(
            x).take(tf.gather(n_samples_loaded_per_tfr,
                              tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-3], sep='-')[0],
                                                   tf.int32) - 1)),
        cycle_length=16,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    tfr_files_val_ds = tfr_files_val_ds.interleave(
        lambda x: tf.data.TFRecordDataset(x).skip(samples_train_shared).take(
            n_samples_tfr_shared - samples_train_shared) if tf.math.equal(x,
                                                                          shared_tfr_out) else tf.data.TFRecordDataset(
            x).take(tf.gather(n_samples_loaded_per_tfr,
                              tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-3], sep='-')[0],
                                                   tf.int32) - 1)),
        cycle_length=16,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dataset_train = tfr_files_train_ds.map(lambda x: tf_parser(x),
                                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_train = dataset_train.shuffle(shuffle_buffer)
    dataset_train = dataset_train.batch(batch_size=batch_size)
    dataset_train = dataset_train.prefetch(n_prefetch)

    dataset_valid = tfr_files_val_ds.map(lambda x: tf_parser(x),
                                         num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_valid = dataset_valid.shuffle(shuffle_buffer)
    dataset_valid = dataset_valid.batch(batch_size=batch_size)
    dataset_valid = dataset_valid.prefetch(n_prefetch)

    return dataset_train, dataset_valid


def tf_parser(rec):

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

    try:
        flow = (tf.reshape(parsed_rec['raw_u'], (nx, ny, nz, 1)) - U_mean) / U_std
        flow = tf.concat((flow, (tf.reshape(parsed_rec['raw_v'], (nx, ny, nz, 1)) - V_mean) / V_std), -1)
        flow = tf.concat((flow, (tf.reshape(parsed_rec['raw_w'], (nx, ny, nz, 1)) - W_mean) / W_std), -1)

        flow = tf.where(tf.math.is_nan(flow), tf.zeros_like(flow), flow)

        wall = (tf.reshape(parsed_rec['raw_b_p'], (nx, 1, nz, 1)) - PB_mean) / PB_std
        wall = tf.concat((wall, (tf.reshape(parsed_rec['raw_b_tx'], (nx, 1, nz, 1)) - TBX_mean) / TBX_std), -1)
        wall = tf.concat((wall, (tf.reshape(parsed_rec['raw_b_tz'], (nx, 1, nz, 1)) - TBZ_mean) / TBZ_std), -1)

        return wall, flow[:, 0:64, :, :]
    except Exception as e:
        print("Issue shaping record", rec)


if __name__ == '__main__':

    main()
