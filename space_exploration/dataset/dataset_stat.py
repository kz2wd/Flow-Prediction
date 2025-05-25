from __future__ import annotations

from dataclasses import dataclass

import dask
import numpy as np
from dask.array import Array
from dask.diagnostics import ProgressBar


@dataclass
class DatasetStats:
    u_means: np.array
    v_means: np.array
    w_means: np.array
    u_stds: np.array
    v_stds: np.array
    w_stds: np.array


    @staticmethod
    def from_ds(ds: Array):
        # ds shape: B 3 X Y Z

        means = ds.mean(axis=(0, 2, 4))
        stds = ds.std(axis=(0, 2, 4))

        print("Computing mean & std of dataset ⏳")
        with ProgressBar():
            means, stds = dask.compute(means, stds)

        stats = DatasetStats(
            *means,
            *stds
        )

        return stats