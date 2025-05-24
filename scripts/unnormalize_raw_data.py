from pathlib import Path

import dask.array as da
import numpy as np
from dask.diagnostics import ProgressBar

from space_exploration.FolderManager import FolderManager
from space_exploration.dataset import s3_access
from space_exploration.dataset.ChannelData import ChannelData

if __name__ == "__main__":
    fs = s3_access.fs
    target_bucket = 'simulations'
    target_dataset = 'paper-dataset.zarr'

    file_path = f"s3://{target_bucket}/{target_dataset}"
    store = s3_access.get_s3_map(file_path)
    ds = da.from_zarr(store)

    channel_data_file: Path = FolderManager.tfrecords / "scaling.npz"
    channel_data = ChannelData(channel_data_file)
    y_dimension = np.load(FolderManager.channel_coordinates / "coordY.npy")

    u_means = channel_data.U_mean.reshape(-1)
    v_means = channel_data.V_mean.reshape(-1)
    w_means = channel_data.W_mean.reshape(-1)

    u_stds = channel_data.U_std.reshape(-1)
    v_stds = channel_data.V_std.reshape(-1)
    w_stds = channel_data.W_std.reshape(-1)

    scale = 100 / 3
    #
    # plt.semilogx(y_dimension[1:64] * 200, u_means[1:] * scale, label='U')
    # plt.semilogx(y_dimension[1:64] * 200, v_means[1:] * scale, label='V')
    # plt.semilogx(y_dimension[1:64] * 200, w_means[1:] * scale, label='W')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("velocity.png")

    if ds.shape[-1] == 3:
        ds = ds.transpose((0, 4, 1, 2, 3))
        print(ds.shape)

    print("actual dataset")

    components = [0, 1, 2]  # channel indices: u, v, w
    stds = [u_stds, v_stds, w_stds]
    means = [u_means, v_means, w_means]

    for c, std, mean in zip(components, stds, means):
        std = std[None, None, :, None]  # Casting into correct shape
        mean = mean[None, None, :, None]
        ds[:, c, :, :, :] = ds[:, c, :, :, :] * std + mean

    with ProgressBar():
        ds.compute()
    print(ds.shape)
    input("store?")
    with ProgressBar():
        ds.to_zarr(store, overwrite=True)

    # with ProgressBar():
    #     plt.semilogx(y_dimension[1:64] * 200, ds[:, 0].mean(axis=(0, 1, 3))[1:] * scale, label='U')
    #     plt.semilogx(y_dimension[1:64] * 200, ds[:, 1].mean(axis=(0, 1, 3))[1:] * scale, label='V')
    #     plt.semilogx(y_dimension[1:64] * 200, ds[:, 2].mean(axis=(0, 1, 3))[1:] * scale, label='W')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("dtvelocity.png")
