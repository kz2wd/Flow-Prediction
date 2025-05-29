from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar

from space_exploration.dataset import s3_access

if TYPE_CHECKING:
    from space_exploration.beans.dataset_bean import Dataset


def benchmark_dataset(ds_bean: 'Dataset'):
    # ds shape: (Batch, velocity component, x, y, z)
    ds = ds_bean.load_s3() * ds_bean.scaling

    y_start = 1 if ds_bean.channel.discard_first_y else 0
    y_dimension = ds_bean.channel.get_simulation_channel().y_dimension
    max_y = ds.shape[3]
    y_dimension = y_dimension[y_start: max_y]
    y_dimension = y_dimension * ds_bean.channel.y_scale_to_y_plus

    ds = ds[:, :, :, y_start:, :]

    with ProgressBar():
        velocity_mean = ds.mean(axis=(0, 2, 4)).compute()  # (3, y)
        velocity_std = ds.std(axis=(2, 4)).mean(axis=0).compute()  # (3, y)
        fluctuation = ds - velocity_mean[None, :, None, :, None]
        squared_velocity_mean = (fluctuation ** 2).mean(axis=(0, 2, 4)).compute()  # (3, y)
        reynolds_uv = (fluctuation[:, 0] * fluctuation[:, 1]).mean(axis=(0, 1, 3)).compute()  # (y,)

    components = ['u', 'v', 'w']
    num_components = 3
    y_size = len(y_dimension)

    # Flatten into long form
    data = {
        'component': np.repeat(components, y_size),
        'y_dimension': np.tile(y_dimension, num_components),
        'velocity_mean': velocity_mean.flatten(),
        'velocity_std': velocity_std.flatten(),
        'squared_velocity_mean': squared_velocity_mean.flatten(),
        'reynolds_uv': np.tile(reynolds_uv.flatten(), num_components),
        'dataset_id': ds_bean.id,
        'name': str(ds_bean.name),
    }

    df = pd.DataFrame(data)

    s3_access.store_df(df, ds_bean.get_benchmark_storage_name())

    print(f"Saved benchmark for {ds_bean.name} to {ds_bean.get_benchmark_storage_name()}")

