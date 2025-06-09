from pathlib import Path

import numpy as np

from scripts.database_add import add_dataset
from space_exploration.FolderManager import FolderManager
from space_exploration.beans.channel_bean import Channel
from space_exploration.dataset import db_access
from space_exploration.dataset.dataset_stat import DatasetStats

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

    stats = DatasetStats(
        U_mean.reshape(-1),
        V_mean.reshape(-1),
        W_mean.reshape(-1),
        U_std.reshape(-1),
        V_std.reshape(-1),
        W_std.reshape(-1),
    )

    channel = Channel.get_channel(session, "paper-channel")
    if channel is None:
        print("Channel not found")
        exit(1)

    add_dataset(
        session=session,
        name="paper-train",
        scaling=100 / 3,
        channel=channel,
        stats=stats,
    )

    session.commit()
