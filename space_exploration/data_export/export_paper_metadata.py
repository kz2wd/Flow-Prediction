from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from FolderManager import FolderManager
from space_exploration.dataset.dataset_bean import Dataset, DatasetStat
from space_exploration.simulation_channel.ChannelData import ChannelData

if __name__ == "__main__":
    session = Session()

    dataset = Dataset(
        name="paper-validation",
        s3_storage_name="paper-dataset.zarr",
    )

    channel_data_file: Path = FolderManager.tfrecords / "scaling.npz"
    channel_data = ChannelData(channel_data_file)
    y_dimension = np.load(FolderManager.channel_coordinates / "coordY.npy")

    for i in range(64):
        stat = DatasetStat(
            y_index=i,
            y_coord=y_dimension[i],
            velocity_mean=np.linalg.norm(np.array([channel_data.U_mean, channel_data.V_mean, channel_data.W_mean])),
            velocity_std=np.linalg.norm(np.array([channel_data.U_std, channel_data.V_std, channel_data.W_std])),
        )
        dataset.stats.append(stat)


    session.add(dataset)

    session.commit()