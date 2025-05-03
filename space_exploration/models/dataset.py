import torch
from torch.utils.data import Dataset
import h5py

class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        self.file = h5py.File(file_path, "r")
        self.x = self.file["x"]
        self.y = self.file["y"]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y

    def __del__(self):
        self.file.close()