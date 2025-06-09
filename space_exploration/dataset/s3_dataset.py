import torch
from dask.diagnostics import ProgressBar
from torch.utils.data import Dataset

from space_exploration.dataset.normalize.normalizer_base import NormalizerBase


class S3Dataset(Dataset):
    def __init__(self, x_ds, y_ds, max_y, normalizer: NormalizerBase):

        self.y_ds = normalizer.normalize(y_ds)
        print("⌛ Initializing Dataset...")
        with ProgressBar():
            self.y_ds = self.y_ds.compute()

        self.x = x_ds.compute()
        self.y = self.y_ds[..., :, :, :max_y, :]  # [B, 3, x, y, z]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y
