import time
from abc import ABC, abstractmethod

import h5py
import mlflow
import numpy as np
import tqdm
import vtk
from torch.utils.data import DataLoader, random_split
from vtk.util import numpy_support

import torch
import torch.nn.functional as F

from space_exploration.FolderManager import FolderManager
from space_exploration.data_viz.PlotData import PlotData, save_benchmarks
from space_exploration.models.dataset import HDF5Dataset
from space_exploration.simulation_channel import SimulationChannel
from visualization.saving_file_names import *


# Loss functions
def generator_loss(fake_y, y_pred, y_true):
    adversarial_labels = torch.ones_like(fake_y) - torch.rand_like(fake_y) * 0.2
    adversarial_loss = F.binary_cross_entropy(fake_y, adversarial_labels, reduction='none')

    content_loss = F.mse_loss(y_pred, y_true, reduction='none').mean(dim=(1, 2, 3, 4))
    total_loss = content_loss + 1e-3 * adversarial_loss

    return total_loss.mean()

def discriminator_loss(real_y, fake_y):
    real_labels = torch.ones_like(real_y) - torch.rand_like(real_y) * 0.2
    fake_labels = torch.rand_like(fake_y) * 0.2

    real_loss = F.binary_cross_entropy(real_y, real_labels, reduction='none')
    fake_loss = F.binary_cross_entropy(fake_y, fake_labels, reduction='none')

    total_loss = 0.5 * (real_loss + fake_loss)
    return total_loss.mean()

class GAN3D(ABC):
    def __init__(self, name, channel: SimulationChannel,
                 n_residual_blocks=32, input_channels=3, output_channels=3):

        self.channel: SimulationChannel = channel

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.name = name
        self.n_residual_blocks = n_residual_blocks

        FolderManager.init(self)  # ensure all related folders are created

        self.device = torch.device("cuda")  # if we cannot get cuda, don't even try...

        self.generator = self.get_generator(channel).to(self.device)
        self.discriminator = self.get_discriminator(channel).to(self.device)
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss


    @abstractmethod
    def get_generator(self, channel: SimulationChannel):
        pass

    @abstractmethod
    def get_discriminator(self, channel: SimulationChannel):
        pass

