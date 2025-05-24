import numpy as np
from sqlalchemy.orm import sessionmaker

from space_exploration.beans.alchemy_base import Base
from space_exploration.beans.channel_bean import Channel
from space_exploration.beans.dataset_stat_bean import DatasetStat
from space_exploration.dataset import db_access
from space_exploration.FolderManager import FolderManager
from space_exploration.beans.dataset_bean import Dataset
from space_exploration.simulation_channel.ChannelData import ChannelData

from pathlib import Path
from sqlalchemy import create_engine


def add_dataset(session, name, s3_storage_name, channel, scaling, u_means,
                v_means, w_means, u_stds, v_stds, w_stds):
    dataset = Dataset(
        name=name,
        s3_storage_name=s3_storage_name,
        scaling=scaling,
        channel=channel,
    )

    for i in range(64):
        stat = DatasetStat(
            y_index=i,
            u_mean=float(u_means[i]),
            v_mean=float(v_means[i]),
            w_mean=float(w_means[i]),
            u_std=float(u_stds[i]),
            v_std=float(v_stds[i]),
            w_std=float(w_stds[i]),
        )
        dataset.stats.append(stat)


    session.add(dataset)


if __name__ == "__main__":


    session = db_access.get_session()

    channel_data_file: Path = FolderManager.tfrecords / "scaling.npz"
    with np.load(channel_data_file) as f:
        U_mean = np.expand_dims(f['U_mean'], axis=-1)[:, :64, :, :]
        V_mean = np.expand_dims(f['V_mean'], axis=-1)[:, :64, :, :]
        W_mean = np.expand_dims(f['W_mean'], axis=-1)[:, :64, :, :]
        U_std = np.expand_dims(f['U_std'], axis=-1)[:, :64, :, :]
        V_std = np.expand_dims(f['V_std'], axis=-1)[:, :64, :, :]
        W_std = np.expand_dims(f['W_std'], axis=-1)[:, :64, :, :]

    u_means = U_mean.reshape(-1)
    v_means = V_mean.reshape(-1)
    w_means = W_mean.reshape(-1)
    u_stds = U_std.reshape(-1)
    v_stds = V_std.reshape(-1)
    w_stds = W_std.reshape(-1)

    channel = Channel.get_channel(session, "paper-channel")
    if channel is None:
        print("Channel not found")
        exit(1)

    add_dataset(
        session=session,
        name="paper-validation",
        s3_storage_name="paper-dataset.zarr",
        scaling=100 / 3,
        channel=channel,
        u_means=u_means,
        v_means=v_means,
        w_means=w_means,
        u_stds=u_stds,
        v_stds=v_stds,
        w_stds=w_stds,
    )


    session.commit()