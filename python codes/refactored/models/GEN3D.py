import re
import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import vtk
from vtk.util import numpy_support

from abc import ABC, abstractmethod

class GEN3D(ABC):
    def __init__(self, root_folder, COORDINATES_FOLDER_PATH, model_name, checkpoint, nx, ny, nz, NX, NY, NZ, LX, LZ, learning_rate, flow_range, input_channels=3, output_channels=3,
                 n_residual_blocks=32):
        self.x_target = None
        self.y_predic = None
        self.y_target = None
        self.error_w = None
        self.error_v = None
        self.error_u = None
        self.channel_Z = None
        self.channel_Y = None
        self.channel_X = None
        self.checkpoint = checkpoint
        self.root_folder = root_folder
        self.flow_range = flow_range
        self.COORDINATES_FOLDER_PATH = COORDINATES_FOLDER_PATH
        # N_ is for the channel simulation
        # n_ is for the predicted area
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.NX = NX
        self.NY = NY
        self.NZ = NZ
        self.LX = LX
        self.LZ = LZ
        self.learning_rate = learning_rate
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.model_name = model_name
        self.n_residual_blocks = n_residual_blocks

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

    def read_channel_mesh_bin(self):
        self.channel_X = np.arange(self.NX) * self.LX / self.NX
        self.channel_Z = np.arange(self.NZ) * self.LZ / self.NZ
        try:
            self.channel_Y = np.load(f'{self.COORDINATES_FOLDER_PATH}/coordY.npy')
        except FileNotFoundError:
            print("CoordY.npy not found, visualizations might be disabled.", file=sys.stderr)

    @tf.function
    def tf_parser(self, rec, root_folder):
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
            'raw_p': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            # 'raw_t_p': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'raw_tx': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'raw_tz': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            # 'raw_t_tx': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            # 'raw_t_tz': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        }

        parsed_rec = tf.io.parse_single_example(rec, features)

        i_smp = tf.cast(parsed_rec['i_sample'], tf.int32)

        nx = tf.cast(parsed_rec['nx'], tf.int32)
        ny = tf.cast(parsed_rec['ny'], tf.int32)
        nz = tf.cast(parsed_rec['nz'], tf.int32)

        filename = f"{root_folder}tfrecords/scaling.npz"

        # Load mean velocity values in the streamwise and wall-normal directions for low- and high-resolution data

        U_mean = np.expand_dims(np.load(filename)['U_mean'], axis=-1)[:, :64, :, :]
        V_mean = np.expand_dims(np.load(filename)['V_mean'], axis=-1)[:, :64, :, :]
        W_mean = np.expand_dims(np.load(filename)['W_mean'], axis=-1)[:, :64, :, :]
        PB_mean = np.expand_dims(np.load(filename)['PB_mean'], axis=-1)
        # PT_mean = np.expand_dims(np.load(filename)['PT_mean'], axis=-1)
        TBX_mean = np.expand_dims(np.load(filename)['TBX_mean'], axis=-1)
        TBZ_mean = np.expand_dims(np.load(filename)['TBZ_mean'], axis=-1)
        # TTX_mean = np.expand_dims(np.load(filename)['TTX_mean'], axis=-1)
        # TTZ_mean = np.expand_dims(np.load(filename)['TTZ_mean'], axis=-1)

        # Load standard deviation velocity values in the streamwise and wall-normal directions for low- and high-resolution data

        U_std = np.expand_dims(np.load(filename)['U_std'], axis=-1)[:, :64, :, :]
        V_std = np.expand_dims(np.load(filename)['V_std'], axis=-1)[:, :64, :, :]
        W_std = np.expand_dims(np.load(filename)['W_std'], axis=-1)[:, :64, :, :]
        PB_std = np.expand_dims(np.load(filename)['PB_std'], axis=-1)
        # PT_std = np.expand_dims(np.load(filename)['PT_std'], axis=-1)
        TBX_std = np.expand_dims(np.load(filename)['TBX_std'], axis=-1)
        TBZ_std = np.expand_dims(np.load(filename)['TBZ_std'], axis=-1)
        # TTX_std = np.expand_dims(np.load(filename)['TTX_std'], axis=-1)
        # TTZ_std = np.expand_dims(np.load(filename)['TTZ_std'], axis=-1)
        # Reshape data into 2-dimensional matrix, substract mean value and divide by the standard deviation. Concatenate the streamwise and wall-normal velocities along the third dimension

        flow = (tf.reshape(parsed_rec['raw_u'], (nx, ny, nz, 1)) - U_mean) / U_std
        flow = tf.concat((flow, (tf.reshape(parsed_rec['raw_v'], (nx, ny, nz, 1)) - V_mean) / V_std), -1)
        flow = tf.concat((flow, (tf.reshape(parsed_rec['raw_w'], (nx, ny, nz, 1)) - W_mean) / W_std), -1)

        flow = tf.where(tf.math.is_nan(flow), tf.zeros_like(flow), flow)

        wall = (tf.reshape(parsed_rec['raw_p'], (nx, 1, nz, 1)) - PB_mean) / PB_std
        wall = tf.concat((wall, (tf.reshape(parsed_rec['raw_tx'], (nx, 1, nz, 1)) - TBX_mean) / TBX_std), -1)
        wall = tf.concat((wall, (tf.reshape(parsed_rec['raw_tz'], (nx, 1, nz, 1)) - TBZ_mean) / TBZ_std), -1)

        return wall, flow[:, self.flow_range, :, :]

    def generate_pipeline_training(self, root_folder, validation_split=1, shuffle_buffer=200, batch_size=4,
                                   n_prefetch=4):

        tfr_path = f"{root_folder}tfrecords/test/"
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

        dataset_valid = tfr_files_val_ds.map(lambda x: self.tf_parser(x, root_folder),
                                             num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_valid = dataset_valid.shuffle(shuffle_buffer)
        dataset_valid = dataset_valid.batch(batch_size=batch_size)
        dataset_valid = dataset_valid.prefetch(n_prefetch)

        return dataset_valid  # dataset_train, dataset_valid

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

        dataset_valid = self.generate_pipeline_training(self.root_folder, batch_size=1)

        generator = self.generator()
        discriminator = self.discriminator()
        generator_loss = self.generator_loss()
        discriminator_loss = self.discriminator_loss()
        generator_optimizer = self.generator_optimizer()
        discriminator_optimizer = self.discriminator_optimizer()

        checkpoint_dir = f"{self.root_folder}models/checkpoints_{self.model_name}"

        # Define checkpoint prefix

        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

        # Generate checkpoint object to track the generator and discriminator architectures and optimizers

        checkpoint = tf.train.Checkpoint(
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator=generator,
            discriminator=discriminator
        )

        # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        ckpt_file = f"{self.root_folder}/models/checkpoints_{self.model_name}/{self.checkpoint}"
        checkpoint.restore(ckpt_file).expect_partial()

        # samples, stream wise resolution, wall normal wise resolution, span wise resolution, velocities.
        # velocities -> 0, u, stream wise | 1, v, wall normal wise | 2, w, spawn wise
        self.x_target = np.zeros((test_sample_amount, self.nx, 1, self.nz, 3), np.float32)
        self.y_target = np.zeros((test_sample_amount, self.nx, self.ny, self.nz, 3), np.float32)
        self.y_predic = np.zeros((test_sample_amount, self.nx, self.ny, self.nz, 3), np.float32)

        itr = iter(dataset_valid)
        try:
            for idx in range(test_sample_amount):
                if float.is_integer(idx / 50):
                    print(idx)
                # print(idx)
                x, y = next(itr)

                self.x_target[idx] = x.numpy()
                self.y_target[idx] = y.numpy()

                self.y_predic[idx] = generator(np.expand_dims(self.x_target[idx], axis=0), training=False)
        except Exception as e:
            print(e)

        self.read_channel_mesh_bin()

        print(f'Target mean: {np.mean(self.y_target)}, std: {np.std(self.y_target)}')
        print(f'Predic mean: {np.mean(self.y_predic)}, std: {np.std(self.y_predic)}')

    def ensure_prediction(self):
        if self.y_target is None or self.y_predic is None or self.x_target is None:
            print("Target and prediction not computed, please predict something first", file=sys.stderr)
            raise GEN3D.NoPredictionsException()

    def compute_errors(self):
        self.ensure_prediction()
        self.error_v = np.mean((self.y_target[:, :, :, :, 1] - self.y_predic[:, :, :, :, 1]) ** 2, axis=(0, 1, 3))
        self.error_w = np.mean((self.y_target[:, :, :, :, 2] - self.y_predic[:, :, :, :, 2]) ** 2, axis=(0, 1, 3))
        self.error_u = np.mean((self.y_target[:, :, :, :, 0] - self.y_predic[:, :, :, :, 0]) ** 2, axis=(0, 1, 3))

    def plot_results(self, error_fig_name="error.png", contour_fig_name="prediction.png"):
        if self.error_u is None or self.error_v is None or self.error_w is None:
            try:
                self.compute_errors()
            except GEN3D.NoPredictionsException:
                return

        if self.channel_Y is None or self.channel_X is None or self.channel_Z is None:
            print("Channels not loaded, please read channel mesh bin", file=sys.stderr)
        plt.rc('font', size=15)
        plt.semilogx(200 * self.channel_Y[1:self.ny], self.error_u[1:], label='MSE U')
        plt.semilogx(200 * self.channel_Y[1:self.ny], self.error_v[1:], label='MSE V')
        plt.semilogx(200 * self.channel_Y[1:self.ny], self.error_w[1:], label='MSE W')
        plt.xlim([1, 200])
        plt.ylim([0, 1])
        plt.legend()
        plt.grid()
        plt.grid(which='minor', linestyle='--')
        plt.savefig(error_fig_name)
        plt.clf()
        rows = 2
        cols = 3
        ratio = 0.5
        inches_per_pt = 1.0 / 72.27
        fig_width_pt = 2000
        fig_width = fig_width_pt * inches_per_pt
        fig_height = fig_width * rows / cols * ratio
        fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
        axs[0, 0].contourf(self.channel_X, self.channel_Z, self.y_target[0, :, 22, :, 0].T, vmin=-3, vmax=3, cmap='RdBu_r')
        axs[1, 0].contourf(self.channel_X, self.channel_Z, self.y_predic[0, :, 22, :, 0].T, vmin=-3, vmax=3, cmap='RdBu_r')
        axs[0, 1].contourf(self.channel_X, self.channel_Z, self.y_target[0, :, 22, :, 1].T, vmin=-3, vmax=3, cmap='PuOr_r')
        axs[1, 1].contourf(self.channel_X, self.channel_Z, self.y_predic[0, :, 22, :, 1].T, vmin=-3, vmax=3, cmap='PuOr_r')
        axs[0, 2].contourf(self.channel_X, self.channel_Z, self.y_target[0, :, 22, :, 2].T, vmin=-3, vmax=3, cmap='PiYG_r')
        axs[1, 2].contourf(self.channel_X, self.channel_Z, self.y_predic[0, :, 22, :, 2].T, vmin=-3, vmax=3, cmap='PiYG_r')
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
        fig.savefig(contour_fig_name)

    def export_vti(self):
        try:
            self.ensure_prediction()
        except GEN3D.NoPredictionsException:
            return


        export_target = self.y_target[0]
        points = vtk.vtkPoints()
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    points.InsertNextPoint(self.channel_X[i], self.channel_Y[j], self.channel_Z[k])
        grid = vtk.vtkRectilinearGrid()
        grid.SetDimensions(self.nx, self.ny, self.nz)
        grid.SetXCoordinates(numpy_support.numpy_to_vtk(self.channel_X))
        grid.SetYCoordinates(numpy_support.numpy_to_vtk(self.channel_Y))
        grid.SetZCoordinates(numpy_support.numpy_to_vtk(self.channel_Z))

        velocity_array = numpy_support.numpy_to_vtk(num_array=export_target.reshape(-1, 3), deep=True, array_type=vtk.VTK_FLOAT)
        velocity_array.SetName('velocity')

        grid.GetPointData().AddArray(velocity_array)


        writer = vtk.vtkXMLRectilinearGridWriter()
        writer.SetFileName("target_velocity_field.vti")
        writer.SetInputData(grid)
        writer.Write()


