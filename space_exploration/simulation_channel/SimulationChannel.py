
from pathlib import Path

import numpy as np


from space_exploration.FolderManager import FolderManager
from space_exploration.simulation_channel.ChannelData import ChannelData
from space_exploration.simulation_channel.PredictionSubSpace import PredictionSubSpace


class SimulationChannel:

    def __init__(self, x_length, x_resolution, z_length, z_resolution, prediction_sub_space: PredictionSubSpace, channel_data_file: Path = FolderManager.tfrecords / "scaling.npz"):
        self.x_dimension = np.arange(x_resolution) * x_length / x_resolution
        self.y_dimension = np.load(FolderManager.channel_coordinates / "coordY.npy")
        self.z_dimension = np.arange(z_resolution) * z_length / z_resolution
        self.prediction_sub_space: PredictionSubSpace = prediction_sub_space

        self.data = ChannelData(channel_data_file)