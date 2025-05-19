
from abc import ABC

import numpy as np
from torch import nn

from FolderManager import FolderManager
from space_exploration.models.GAN3D import GAN3D
from space_exploration.simulation_channel.PredictionSubSpace import PredictionSubSpace
from space_exploration.simulation_channel.SimulationChannel import SimulationChannel


class Discriminator(nn.Module):
    def __init__(self, channel: SimulationChannel):
        nx = channel.prediction_sub_space.x[1]
        ny = channel.prediction_sub_space.y[1]
        nz = channel.prediction_sub_space.z[1]
        super().__init__()
        total_stride = 4 * 2 * 2 * 2
        flatten_size = nx // total_stride * ny // total_stride * nz // total_stride * 512  # each channel gets divided by
        # the total stride, times 512 because we have 512 filters at the end
        self.model = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 64, kernel_size=3, stride=4, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(flatten_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, X, Y, Z)
        x = self.model(x)
        return x


class PaperBase(GAN3D, ABC):
    """
    A base for models coming from the paper
    """

    def __init__(self, name, checkpoint_number, y_end, up_sampling_indices=None):
        prediction_sub_space = PredictionSubSpace(x_start=0, x_end=64, y_start=0, y_end=y_end, z_start=0,
                                                  z_end=64)
        channel = SimulationChannel(x_length=np.pi, z_length=np.pi / 2, x_resolution=64, z_resolution=64,
                                    prediction_sub_space=prediction_sub_space,
                                    channel_data_file=FolderManager.tfrecords / "scaling.npz")
        super().__init__(name=name, checkpoint_number=checkpoint_number, channel=channel)


    def get_discriminator(self, channel: SimulationChannel):
        return Discriminator(channel)