import bisect
import os
import re
import sys
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import vtk
from vtk.util import numpy_support

from folder_locations import *
from visualization.saving_file_names import *


class GEN3D(ABC):
    def __init__(self, model_name, checkpoint, prediction_area_x, prediction_area_y_start, prediction_area_y_end,
                 prediction_area_z, channel_x_resolution, channel_y_resolution, channel_z_resolution, channel_x_length,
                 channel_z_length, learning_rate, input_channels=3, output_channels=3,
                 n_residual_blocks=32,
                 COORDINATES_FOLDER_PATH=CHANNEL_COORDINATES_FOLDER,
                 generated_data_folder=GENERATED_DATA_FOLDER, checkpoint_folder=CHECKPOINTS_FOLDER,
                 tfrecords_folder=TFRECORDS_FOLDER, in_legend_name=None):

        if in_legend_name is None:
            self.in_legend_name = model_name
        else:
            self.in_legend_name = in_legend_name
        self._y_target_original = None
        self._y_predict_original = None
        self.TBZ_std = None
        self.TBX_std = None
        self.PB_std = None
        self.W_std = None
        self.V_std = None
        self.U_std = None
        self.TBZ_mean = None
        self.TBX_mean = None
        self.PB_mean = None
        self.W_mean = None
        self.V_mean = None
        self.U_mean = None
        self.tfrecords_folder = tfrecords_folder
        self.checkpoint_folder = checkpoint_folder / model_name
        self.checkpoint_folder.mkdir(parents=True, exist_ok=True)
        self.generated_data_folder = generated_data_folder / model_name
        self.generated_data_folder.mkdir(parents=True, exist_ok=True)
        self.x_target_normalized = None
        self.y_predict_normalized = None
        self.y_target_normalized = None
        self.network_error_w = None
        self.network_error_v = None
        self.network_error_u = None
        self.real_error_u = None
        self.real_error_v = None
        self.real_error_w = None
        self.channel_Z = None
        self.channel_Y = None
        self.channel_X = None
        self.checkpoint = checkpoint
        self.coordinates_folder = COORDINATES_FOLDER_PATH
        self.prediction_area_x = prediction_area_x
        self.prediction_area_y_start = prediction_area_y_start
        self.prediction_area_y_end = prediction_area_y_end
        self.prediction_area_z = prediction_area_z
        self.channel_x_resolution = channel_x_resolution
        self.channel_y_resolution = channel_y_resolution
        self.channel_z_resolution = channel_z_resolution
        self.channel_x_length = channel_x_length
        self.channel_z_length = channel_z_length
        self.learning_rate = learning_rate
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.model_name = model_name
        self.n_residual_blocks = n_residual_blocks
        self.data_scaling = 100 / 3

        self.records_features = {
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
            'raw_p': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            # 'raw_t_p': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'raw_tx': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'raw_tz': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            # 'raw_t_tx': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            # 'raw_t_tz': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        }

        self.read_scaling()

    def _to_original(self, target):
        # This function was made by hand and refactored by chatgpt because of my skill issues regarding numpy knowledge
        # enjoy this nice form

        _, _, Y, _, _ = target.shape

        # Reshape means and stds to broadcast correctly over the Y axis
        U_std = self.U_std.reshape(1, 1, Y, 1, 1)
        V_std = self.V_std.reshape(1, 1, Y, 1, 1)
        W_std = self.W_std.reshape(1, 1, Y, 1, 1)

        U_mean = self.U_mean.reshape(1, 1, Y, 1, 1)
        V_mean = self.V_mean.reshape(1, 1, Y, 1, 1)
        W_mean = self.W_mean.reshape(1, 1, Y, 1, 1)

        original = target.copy()

        # Apply transformation vectorially
        original[..., 0] = (target[..., 0] * U_std[..., 0] + U_mean[..., 0]) * self.data_scaling
        original[..., 1] = (target[..., 1] * V_std[..., 0] + V_mean[..., 0]) * self.data_scaling
        original[..., 2] = (target[..., 2] * W_std[..., 0] + W_mean[..., 0]) * self.data_scaling

        return original

    @property
    def y_target_original(self):
        if self.y_target_normalized is None:
            return None
        if self._y_target_original is None:
            self._y_target_original = self._to_original(self.y_target_normalized)
        return self._y_target_original

    @property
    def y_predict_original(self):
        if self.y_predict_normalized is None:
            return None
        if self._y_predict_original is None:
            self._y_predict_original = self._to_original(self.y_predict_normalized)
        return self._y_predict_original

    def read_scaling(self):
        filename = self.tfrecords_folder / "scaling.npz"

        # Load mean velocity values in the streamwise and wall-normal directions for low- and high-resolution data
        with np.load(filename) as f:
            self.U_mean = np.expand_dims(f['U_mean'], axis=-1)[:, :64, :, :]
            self.V_mean = np.expand_dims(f['V_mean'], axis=-1)[:, :64, :, :]
            self.W_mean = np.expand_dims(f['W_mean'], axis=-1)[:, :64, :, :]
            self.PB_mean = np.expand_dims(f['PB_mean'], axis=-1)

            # PT_mean = np.expand_dims(np.load(filename)['PT_mean'], axis=-1)
            self.TBX_mean = np.expand_dims(f['TBX_mean'], axis=-1)
            self.TBZ_mean = np.expand_dims(f['TBZ_mean'], axis=-1)
            # TTX_mean = np.expand_dims(np.load(filename)['TTX_mean'], axis=-1)
            # TTZ_mean = np.expand_dims(np.load(filename)['TTZ_mean'], axis=-1)

            # Load standard deviation velocity values in the streamwise and wall-normal directions for low- and high-resolution data

            self.U_std = np.expand_dims(f['U_std'], axis=-1)[:, :64, :, :]
            self.V_std = np.expand_dims(f['V_std'], axis=-1)[:, :64, :, :]
            self.W_std = np.expand_dims(f['W_std'], axis=-1)[:, :64, :, :]
            self.PB_std = np.expand_dims(f['PB_std'], axis=-1)
            # PT_std = np.expand_dims(np.load(filename)['PT_std'], axis=-1)
            self.TBX_std = np.expand_dims(f['TBX_std'], axis=-1)
            self.TBZ_std = np.expand_dims(f['TBZ_std'], axis=-1)
            # TTX_std = np.expand_dims(np.load(filename)['TTX_std'], axis=-1)
            # TTZ_std = np.expand_dims(np.load(filename)['TTZ_std'], axis=-1)

    class NoPredictionsException(Exception):
        def __init__(self):
            super().__init__("No predictions available, run at least once")

    @abstractmethod
    def generator(self):
        pass

    @abstractmethod
    def discriminator(self):
        pass

    @abstractmethod
    def generator_loss(self):
        pass

    @abstractmethod
    def discriminator_loss(self):
        pass

    def generator_optimizer(self):
        return keras.optimizers.Adam(learning_rate=self.learning_rate)

    def discriminator_optimizer(self):
        return keras.optimizers.Adam(learning_rate=self.learning_rate)

    @property
    def prediction_area_y(self):
        return self.prediction_area_y_end - self.prediction_area_y_start

    def read_channel_mesh_bin(self):
        self.channel_X = np.arange(self.channel_x_resolution) * self.channel_x_length / self.channel_x_resolution
        self.channel_Z = np.arange(self.channel_z_resolution) * self.channel_z_length / self.channel_z_resolution
        try:
            self.channel_Y = np.load(self.coordinates_folder / "coordY.npy")
        except FileNotFoundError:
            print("CoordY.npy not found, visualizations might be disabled.", file=sys.stderr)

    @tf.function
    def tf_parser(self, rec):
        parsed_rec = tf.io.parse_single_example(rec, self.records_features)
        i_smp = tf.cast(parsed_rec['i_sample'], tf.int32)
        nx = tf.cast(parsed_rec['nx'], tf.int32)
        ny = tf.cast(parsed_rec['ny'], tf.int32)
        nz = tf.cast(parsed_rec['nz'], tf.int32)

        # Reshape data into 2-dimensional matrix, substract mean value and divide by the standard deviation. Concatenate the streamwise and wall-normal velocities along the third dimension

        flow = (tf.reshape(parsed_rec['raw_u'], (nx, ny, nz, 1)) - self.U_mean) / self.U_std
        flow = tf.concat((flow, (tf.reshape(parsed_rec['raw_v'], (nx, ny, nz, 1)) - self.V_mean) / self.V_std), -1)
        flow = tf.concat((flow, (tf.reshape(parsed_rec['raw_w'], (nx, ny, nz, 1)) - self.W_mean) / self.W_std), -1)

        flow = tf.where(tf.math.is_nan(flow), tf.zeros_like(flow), flow)

        wall = (tf.reshape(parsed_rec['raw_p'], (nx, 1, nz, 1)) - self.PB_mean) / self.PB_std
        wall = tf.concat((wall, (tf.reshape(parsed_rec['raw_tx'], (nx, 1, nz, 1)) - self.TBX_mean) / self.TBX_std), -1)
        wall = tf.concat((wall, (tf.reshape(parsed_rec['raw_tz'], (nx, 1, nz, 1)) - self.TBZ_mean) / self.TBZ_std), -1)

        return wall, flow[:, self.prediction_area_y_start:self.prediction_area_y_end, :, :]

    def generate_pipeline_training(self, validation_split=1, shuffle_buffer=200, batch_size=4,
                                   n_prefetch=4):

        tfr_path = self.tfrecords_folder / "test"
        tfr_files = sorted(
            [os.path.join(tfr_path, f) for f in os.listdir(tfr_path) if os.path.isfile(os.path.join(tfr_path, f))])
        regex = re.compile(f'.tfrecords')
        tfr_files = ([string for string in tfr_files if re.search(regex, string)])

        n_samples_per_tfr = np.array([int(s.split('.')[-2][-3:]) for s in tfr_files])
        n_samples_per_tfr = n_samples_per_tfr[np.argsort(-n_samples_per_tfr)]
        cumulative_samples_per_tfr = np.cumsum(np.array(n_samples_per_tfr))
        tot_samples_per_ds = sum(n_samples_per_tfr)
        # n_tfr_loaded_per_ds = int(tfr_files[0].split('_')[-3][-3:])
        n_tfr_loaded_per_ds = 119  # some were corrupted
        # n_tfr_loaded_per_ds = int(tfr_files[0].split('_')[-3][-4:])
        tfr_files = [string for string in tfr_files if int(string.split('_')[-3][:3]) <= n_tfr_loaded_per_ds]
        # tfr_files = [string for string in tfr_files if int(string.split('_')[-3][:4]) <= n_tfr_loaded_per_ds]

        n_samp_train = int(sum(n_samples_per_tfr) * (1 - validation_split))
        n_samp_valid = sum(n_samples_per_tfr) - n_samp_train

        (n_files_train, samples_train_left) = np.divmod(n_samp_train, n_samples_per_tfr[0])

        if samples_train_left > 0:
            n_files_train += 1

        tfr_files_train = [string for string in tfr_files if int(string.split('_')[-3][:3]) <= n_files_train]
        # tfr_files_train = [string for string in tfr_files if int(string.split('_')[-3][:4]) <= n_files_train]
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

        # tfr_files_train_ds = tf.data.Dataset.list_files(tfr_files_train, seed=666)
        tfr_files_val_ds = tf.data.Dataset.list_files(tfr_files_valid, seed=686)

        if n_tfr_left > 1:

            samples_train_shared = samples_train_left - cumulative_samples_per_tfr[n_tfr_left - 2]
            n_samples_tfr_shared = n_samples_loaded_per_tfr[n_tfr_left - 1]

        else:

            samples_train_shared = samples_train_left
            n_samples_tfr_shared = n_samples_loaded_per_tfr[0]

        # tfr_files_train_ds = tfr_files_train_ds.interleave(
        #    lambda x : tf.data.TFRecordDataset(x).take(samples_train_shared) if tf.math.equal(x, shared_tfr_out) else tf.data.TFRecordDataset(x).take(tf.gather(n_samples_loaded_per_tfr, tf.strings.to_number(tf.strings.split(tf.strings.split(x, sep='_')[-3],sep='-')[0], tf.int32)-1)),
        #    cycle_length=16,
        #    num_parallel_calls=tf.data.experimental.AUTOTUNE
        # )

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

        # dataset_train = tfr_files_train_ds.map(lambda x: tf_parser(x, root_folder), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # dataset_train = dataset_train.shuffle(shuffle_buffer)
        # dataset_train = dataset_train.batch(batch_size=batch_size)
        # dataset_train = dataset_train.prefetch(n_prefetch)

        dataset_valid = tfr_files_val_ds.map(lambda x: self.tf_parser(x),
                                             num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # dataset_valid = dataset_valid.shuffle(shuffle_buffer)
        dataset_valid = dataset_valid.batch(batch_size=batch_size)
        dataset_valid = dataset_valid.prefetch(n_prefetch)

        return dataset_valid  # dataset_train, dataset_valid

    def replace_prediction_with_zeros(self):
        self.y_predict_normalized[...] = 0

    def replace_prediction_with_noise(self):
        self.y_predict_normalized = np.random.default_rng().uniform(-1, 1, self.y_predict_normalized.shape)

    def test(self, test_sample_amount=50):
        physical_devices = tf.config.list_physical_devices('GPU')
        available_GPUs = len(physical_devices)
        print('Using TensorFlow version: ', tf.__version__, ', GPU:', available_GPUs)
        print('Using Keras version: ', tf.keras.__version__)
        if physical_devices:
            try:
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

        dataset_valid = self.generate_pipeline_training(batch_size=1)

        generator = self.generator()
        discriminator = self.discriminator()
        generator_loss = self.generator_loss()
        discriminator_loss = self.discriminator_loss()
        generator_optimizer = self.generator_optimizer()
        discriminator_optimizer = self.discriminator_optimizer()

        # Generate checkpoint object to track the generator and discriminator architectures and optimizers

        checkpoint = tf.train.Checkpoint(
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator=generator,
            discriminator=discriminator
        )

        # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        ckpt_file = self.checkpoint_folder / self.checkpoint
        checkpoint.restore(ckpt_file).expect_partial()

        # samples, stream wise resolution, wall normal wise resolution, span wise resolution, velocities.
        # velocities -> 0, u, stream wise | 1, v, wall normal wise | 2, w, spawn wise
        self.x_target_normalized = np.zeros((test_sample_amount, self.prediction_area_x, 1, self.prediction_area_z, 3),
                                            np.float32)

        self.load_empty_data(test_sample_amount)

        itr = iter(dataset_valid)
        try:
            for i in range(test_sample_amount):
                if float.is_integer(i / 50):
                    print(i)
                # print(idx)
                x, y = next(itr)

                self.x_target_normalized[i] = x.numpy()
                self.y_target_normalized[i] = y.numpy()

                self.y_predict_normalized[i] = generator(np.expand_dims(self.x_target_normalized[i], axis=0),
                                                         training=False)
        except Exception as e:
            print(e)

        self.read_channel_mesh_bin()

        print(f'Target mean: {np.mean(self.y_target_normalized)}, std: {np.std(self.y_target_normalized)}')
        print(f'Predict mean: {np.mean(self.y_predict_normalized)}, std: {np.std(self.y_predict_normalized)}')

    def ensure_prediction(self):
        if self.y_target_normalized is None or self.y_predict_normalized is None or self.x_target_normalized is None:
            print("Target and prediction not computed, please predict something first", file=sys.stderr)
            raise GEN3D.NoPredictionsException()

    def compute_errors(self):
        self.ensure_prediction()
        self.network_error_u = np.mean(
            (self.y_target_normalized[:, :, :, :, 0] - self.y_predict_normalized[:, :, :, :, 0]) ** 2,
            axis=(0, 1, 3))
        self.network_error_v = np.mean(
            (self.y_target_normalized[:, :, :, :, 1] - self.y_predict_normalized[:, :, :, :, 1]) ** 2,
            axis=(0, 1, 3))
        self.network_error_w = np.mean(
            (self.y_target_normalized[:, :, :, :, 2] - self.y_predict_normalized[:, :, :, :, 2]) ** 2,
            axis=(0, 1, 3))

    def save_prediction(self):
        self.ensure_prediction()
        np.save(self.generated_data_folder / "target_x.npy", self.x_target_normalized)
        np.save(self.generated_data_folder / "target_y.npy", self.y_target_normalized)
        np.save(self.generated_data_folder / "predict_y.npy", self.y_predict_normalized)

    def load_prediction(self):
        self.x_target_normalized = np.load(self.generated_data_folder / "target_x.npy")
        self.y_target_normalized = np.load(self.generated_data_folder / "target_y.npy")
        self.y_predict_normalized = np.load(self.generated_data_folder / "predict_y.npy")

    def lazy_predict(self, at_least=1):
        try:
            self.load_prediction()
            if len(self.y_predict_normalized) < at_least:
                raise OSError  # hehehe naughty code
        except OSError as _:
            self.test(at_least)
            self.save_prediction()
        self.read_channel_mesh_bin()

    def plot_compare_normalized_original(self):
        plt.semilogx(self.prediction_channel_y[1:], np.mean(self.y_target_normalized[:, :, 1:, :, 0], axis=(0, 1, 3)),
                     label="Normalized U")
        plt.semilogx(self.prediction_channel_y[1:], np.mean(self.y_target_original[:, :, 1:, :, 0], axis=(0, 1, 3)),
                     label="Original U")
        plt.grid()
        plt.grid(which='minor', linestyle='--')
        plt.legend()
        plt.title("Original vs normalized mean")
        plt.xlabel("y+")
        plt.ylabel("unit")
        plt.show()

    def plot_normalized_means(self):
        plt.semilogx(self.prediction_channel_y[1:], np.mean(self.y_target_normalized[:, :, 1:, :, 0], axis=(0, 1, 3)),
                     label="target U")
        # plt.semilogx(self.prediction_channel_y[1:], np.mean(self.y_target_normalized[:, :, 1:, :, 1], axis=(0, 1, 3)),
        #              label="target V")
        # plt.semilogx(self.prediction_channel_y[1:], np.mean(self.y_target_normalized[:, :, 1:, :, 2], axis=(0, 1, 3)),
        #              label="target W")
        plt.semilogx(self.prediction_channel_y[1:], np.mean(self.y_predict_normalized[:, :, 1:, :, 0], axis=(0, 1, 3)),
                     label="predict U")
        # plt.semilogx(self.prediction_channel_y[1:], np.mean(self.y_predict_normalized[:, :, 1:, :, 1], axis=(0, 1, 3)),
        #              label="predict V")
        # plt.semilogx(self.prediction_channel_y[1:], np.mean(self.y_predict_normalized[:, :, 1:, :, 2], axis=(0, 1, 3)),
        #              label="predict W")
        plt.grid()
        plt.grid(which='minor', linestyle='--')
        plt.legend()
        plt.title("Normalized mean")
        plt.xlabel("y+")
        plt.ylabel("unit")
        plt.show()

        def plot_stds(self):
            plt.semilogx(self.prediction_channel_y[1:],
                         np.std(self.y_target_normalized[:, :, 1:, :, 0], axis=(0, 1, 3)),
                         label="target U")
            plt.semilogx(self.prediction_channel_y[1:],
                         np.std(self.y_target_normalized[:, :, 1:, :, 1], axis=(0, 1, 3)),
                         label="target V")
            plt.semilogx(self.prediction_channel_y[1:],
                         np.std(self.y_target_normalized[:, :, 1:, :, 2], axis=(0, 1, 3)),
                         label="target W")
            plt.semilogx(self.prediction_channel_y[1:],
                         np.std(self.y_predict_normalized[:, :, 1:, :, 0], axis=(0, 1, 3)),
                         label="predict U")
            plt.semilogx(self.prediction_channel_y[1:],
                         np.std(self.y_predict_normalized[:, :, 1:, :, 1], axis=(0, 1, 3)),
                         label="predict V")
            plt.semilogx(self.prediction_channel_y[1:],
                         np.std(self.y_predict_normalized[:, :, 1:, :, 2], axis=(0, 1, 3)),
                         label="predict W")
            plt.grid()
            plt.grid(which='minor', linestyle='--')
            plt.legend()
            plt.title("Standard deviations")
            plt.xlabel("y+")
            plt.ylabel("unit")
            plt.show()


    def plot_empiric_means(self):
        plt.semilogx(self.prediction_channel_y[1:], self.U_mean.reshape(-1)[1:] * self.data_scaling, label="u")
        plt.xlim([0, 200])
        plt.grid()
        plt.grid(which='minor', linestyle='--')
        plt.xlabel("y+")
        plt.ylabel("U+")
        plt.title("Empiric mean")
        plt.legend()
        plt.show()


    def plot_empiric_stds(self):
        plt.plot(self.prediction_channel_y[1:], self.U_std.reshape(-1)[1:] * self.data_scaling)
        plt.plot(self.prediction_channel_y[1:], self.V_std.reshape(-1)[1:] * self.data_scaling)
        plt.plot(self.prediction_channel_y[1:], self.W_std.reshape(-1)[1:] * self.data_scaling)
        plt.grid()
        plt.grid(which='minor', linestyle='--')
        plt.show()


    # Prediction channel Y with correct scale
    @property
    def prediction_channel_y(self):
        if self.channel_Y is None:
            self.read_channel_mesh_bin()
        # 200 so 1 unit of channel = 1 y_plus, comes from the paper ...
        return 200 * self.channel_Y[self.prediction_area_y_start:self.prediction_area_y_end]


    def get_losses(self, y_plus):
        self.compute_errors()
        if self.channel_Y is None or self.channel_X is None or self.channel_Z is None:
            print("Channels not loaded, please read channel mesh bin", file=sys.stderr)

        y_index = bisect.bisect_right(self.prediction_channel_y, y_plus)

        # error_u = np.mean(self.error_u[1: y_index])
        # error_v = np.mean(self.error_v[1: y_index])
        # error_w = np.mean(self.error_w[1: y_index])
        return None  # todo : fix
        # return error_u, error_v, error_w


    def load_empty_data(self, test_sample_amount):
        self.y_target_normalized = np.zeros(
            (test_sample_amount, self.prediction_area_x, self.prediction_area_y, self.prediction_area_z, 3), np.float32)
        self.y_predict_normalized = np.zeros(
            (test_sample_amount, self.prediction_area_x, self.prediction_area_y, self.prediction_area_z, 3), np.float32)


    def plot_wall_normal_profiles(self, figure_name="stream_wise_profiles.png"):
        if self.channel_Y is None or self.channel_X is None or self.channel_Z is None:
            print("Channels not loaded, please read channel mesh bin", file=sys.stderr)
        plt.rc('font', size=15)
        plt.semilogx(self.prediction_channel_y[1:],
                     np.mean(self.y_predict_original[:, :, 1:, :, 0], axis=(0, 1, 3)), label='predicted velocity')
        plt.semilogx(self.prediction_channel_y[1:],
                     np.mean(self.y_target_original[:, :, 1:, :, 0], axis=(0, 1, 3)), label='target velocity')

        plt.semilogx(self.prediction_channel_y[1:], self.U_mean.reshape(-1)[1:] * self.data_scaling,
                     label='average velocity')
        # plt.xlim([1, 200])
        # plt.ylim([0, 1])
        plt.legend()
        plt.title("Averaged U velocities along Y")
        plt.xlabel("y+")
        plt.ylabel("U+")
        plt.grid()
        plt.grid(which='minor', linestyle='--')
        plt.savefig(self.generated_data_folder / figure_name)


    def plot_results(self, error_fig_name="error.png", contour_fig_name="prediction.png"):
        if self.network_error_u is None or self.network_error_v is None or self.network_error_w is None:
            try:
                self.compute_errors()
            except GEN3D.NoPredictionsException:
                return

        if self.channel_Y is None or self.channel_X is None or self.channel_Z is None:
            print("Channels not loaded, please read channel mesh bin", file=sys.stderr)
        plt.rc('font', size=15)
        plt.semilogx(200 * self.channel_Y[1:self.prediction_area_y], self.network_error_u[1:], label='MSE U')
        plt.semilogx(200 * self.channel_Y[1:self.prediction_area_y], self.network_error_v[1:], label='MSE V')
        plt.semilogx(200 * self.channel_Y[1:self.prediction_area_y], self.network_error_w[1:], label='MSE W')
        # plt.xlim([1, 200])
        # plt.ylim([0, 1])
        plt.legend()
        plt.grid()
        plt.grid(which='minor', linestyle='--')
        plt.savefig(self.generated_data_folder / error_fig_name)
        plt.clf()
        rows = 2
        cols = 3
        ratio = 0.5
        inches_per_pt = 1.0 / 72.27
        fig_width_pt = 2000
        fig_width = fig_width_pt * inches_per_pt
        fig_height = fig_width * rows / cols * ratio
        fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
        axs[0, 0].contourf(self.channel_X, self.channel_Z, self.y_target_original[0, :, 22, :, 0].T, vmin=-3, vmax=3,
                           cmap='RdBu_r')
        axs[1, 0].contourf(self.channel_X, self.channel_Z, self.y_predict_original[0, :, 22, :, 0].T, vmin=-3, vmax=3,
                           cmap='RdBu_r')
        axs[0, 1].contourf(self.channel_X, self.channel_Z, self.y_target_original[0, :, 22, :, 1].T, vmin=-3, vmax=3,
                           cmap='PuOr_r')
        axs[1, 1].contourf(self.channel_X, self.channel_Z, self.y_predict_original[0, :, 22, :, 1].T, vmin=-3, vmax=3,
                           cmap='PuOr_r')
        axs[0, 2].contourf(self.channel_X, self.channel_Z, self.y_target_original[0, :, 22, :, 2].T, vmin=-3, vmax=3,
                           cmap='PiYG_r')
        axs[1, 2].contourf(self.channel_X, self.channel_Z, self.y_predict_original[0, :, 22, :, 2].T, vmin=-3, vmax=3,
                           cmap='PiYG_r')
        axs[0, 0].set_xlim([0, np.pi])
        axs[0, 0].set_ylim([0, np.pi / 2])
        axs[0, 1].set_xlim([0, np.pi])
        axs[0, 1].set_ylim([0, np.pi / 2])
        axs[0, 2].set_xlim([0, np.pi])
        axs[0, 2].set_ylim([0, np.pi / 2])
        axs[1, 0].set_xlim([0, np.pi])
        axs[1, 0].set_ylim([0, np.pi / 2])
        axs[1, 1].set_xlim([0, np.pi])
        axs[1, 1].set_ylim([0, np.pi / 2])
        axs[1, 2].set_xlim([0, np.pi])
        axs[1, 2].set_ylim([0, np.pi / 2])
        fig.savefig(self.generated_data_folder / contour_fig_name)


    # WARNING : Correct type here should be rectilinear grid
    # but for some reason my Paraview couldn't display it as a Volume, So I use StructuredGrid
    # If you want to try with rectilinear, add an export_vtr function or something alike.
    def export_vts(self):
        try:
            self.ensure_prediction()
        except GEN3D.NoPredictionsException:
            return

        self._export_array_vts(self.y_target_original[0], TARGET_FILE_NAME, TARGET_ARRAY_NAME)
        self._export_array_vts(self.y_predict_original[0], PREDICTION_FILE_NAME, PREDICTION_ARRAY_NAME)


    # File name with no extension
    def _export_array_vts(self, target, file_name, array_name=None):
        if array_name is None:
            array_name = file_name
        structured_grid = vtk.vtkStructuredGrid()
        points = vtk.vtkPoints()
        for k in range(self.prediction_area_z):
            for j in range(self.prediction_area_y):
                for i in range(self.prediction_area_x):
                    points.InsertNextPoint(self.channel_X[i], self.channel_Y[j], self.channel_Z[k])

        structured_grid.SetPoints(points)
        structured_grid.SetDimensions(self.prediction_area_x, self.prediction_area_y, self.prediction_area_z)

        velocity_array = numpy_support.numpy_to_vtk(num_array=target.reshape(-1, 3), deep=True,
                                                    array_type=vtk.VTK_FLOAT)
        velocity_array.SetName(array_name)

        structured_grid.GetPointData().AddArray(velocity_array)

        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(self.generated_data_folder / file_name)
        writer.SetInputData(structured_grid)
        writer.Write()
