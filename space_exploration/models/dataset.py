import torch
from torch.utils.data import Dataset
import h5py

from space_exploration.simulation_channel.SimulationChannel import SimulationChannel


class HDF5Dataset(Dataset):
    def __init__(self, file_path, channel: SimulationChannel, amount=-1):
        print("opening dataset file")
        print("using amount:", amount)
        self.file = h5py.File(file_path, "r")
        print(self.file.keys())

        self.x = self.file['x'][:amount, ...]  # [X, x, 1, z, 3]
        self.y = self.file['y'][:amount, :, :channel.prediction_sub_space.y[1], :, :]  # [X, x, y, z, 3]
        print("Loaded data:")
        print(self.x.shape)
        print(self.y.shape)
        print("dataset initialized")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y

    def __del__(self):
        self.file.close()