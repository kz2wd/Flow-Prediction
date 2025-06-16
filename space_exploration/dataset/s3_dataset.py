from typing import Type

import torch
from dask.diagnostics import ProgressBar
from torch.utils.data import Dataset

from space_exploration.dataset.transforms.general.default_unchanged import DefaultUnchanged


class S3Dataset(Dataset):
    def __init__(self, ref_bean, x_ds, y_ds, max_y, XTransform: Type = DefaultUnchanged, YTransform: Type = DefaultUnchanged):
        self.ref_bean = ref_bean
        self.x_transform = XTransform(self.ref_bean)
        self.y_transform = YTransform(self.ref_bean)

        self.y_ds = self.y_transform.to_training(y_ds)
        print("⌛ Initializing Dataset...")
        print("X...")
        with ProgressBar():
            self.x = self.x_transform.to_training(x_ds).compute()

        print("Y...")
        with ProgressBar():
            self.y_ds = self.y_ds.compute()

        self.y = self.y_ds[..., :, :, :max_y, :]  # [B, 3, x, y, z]
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y
