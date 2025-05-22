import torch
from torch.utils.data import Dataset


from space_exploration.simulation_channel.SimulationChannel import SimulationChannel


class S3Dataset(Dataset):
    def __init__(self, ds, channel: SimulationChannel):

        self.ds = ds

        self.x = self.ds[..., :, :, 1, :]  # [B, 3, x, 1, z]
        self.y = self.ds[..., :, :, channel.prediction_sub_space.y[1], :]  # [B, 3, x, y, z]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y
