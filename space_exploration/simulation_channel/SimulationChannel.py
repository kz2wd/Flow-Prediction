from typing_extensions import Unpack

from pathlib import Path

import numpy as np

import tensorflow as tf

from FolderManager import FolderManager
from space_exploration.records_reader import RecordsFeatures
from space_exploration.simulation_channel.ChannelData import ChannelData
from space_exploration.simulation_channel.PredictionSubSpace import PredictionSubSpace


class SimulationChannel:

    def __init__(self, x_length, x_resolution, z_length, z_resolution, prediction_sub_space: PredictionSubSpace, channel_data_file: Path = FolderManager.tfrecords / "scaling.npz"):
        self.x_dimension = np.arange(x_resolution) * x_length / x_resolution
        self.y_dimension = np.load(FolderManager.channel_coordinates / "coordY.npy")
        self.z_dimension = np.arange(z_resolution) * z_length / z_resolution
        self.prediction_sub_space: PredictionSubSpace = prediction_sub_space
        self.channel_y_scaling = 200
        self.data = ChannelData(channel_data_file, self.prediction_sub_space)

    @property
    def y_channel(self):
        return self.channel_y_scaling * self.y_dimension[self.prediction_sub_space.y[0]: self.prediction_sub_space.y[1]]

    @tf.function
    def tf_parser(self, rec):
        parsed_rec = tf.io.parse_single_example(rec, RecordsFeatures.features)
        i_smp = tf.cast(parsed_rec['i_sample'], tf.int32)
        nx = tf.cast(parsed_rec['nx'], tf.int32)
        ny = tf.cast(parsed_rec['ny'], tf.int32)
        nz = tf.cast(parsed_rec['nz'], tf.int32)

        # Reshape data into 2-dimensional matrix, substract mean value and divide by the standard deviation. Concatenate the streamwise and wall-normal velocities along the third dimension

        flow = (tf.reshape(parsed_rec['raw_u'], (nx, ny, nz, 1)) - self.data.U_mean) / self.data.U_std
        flow = tf.concat((flow, (tf.reshape(parsed_rec['raw_v'], (nx, ny, nz, 1)) - self.data.V_mean) / self.data.V_std), -1)
        flow = tf.concat((flow, (tf.reshape(parsed_rec['raw_w'], (nx, ny, nz, 1)) - self.data.W_mean) / self.data.W_std), -1)

        flow = tf.where(tf.math.is_nan(flow), tf.zeros_like(flow), flow)

        wall = (tf.reshape(parsed_rec['raw_p'], (nx, 1, nz, 1)) - self.data.PB_mean) / self.data.PB_std
        wall = tf.concat((wall, (tf.reshape(parsed_rec['raw_tx'], (nx, 1, nz, 1)) - self.data.TBX_mean) / self.data.TBX_std), -1)
        wall = tf.concat((wall, (tf.reshape(parsed_rec['raw_tz'], (nx, 1, nz, 1)) - self.data.TBZ_mean) / self.data.TBZ_std), -1)

        return wall, self.prediction_sub_space.select(flow)