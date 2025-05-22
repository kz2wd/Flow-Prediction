import numpy as np
from sqlalchemy.orm import sessionmaker

from space_exploration.dataset import db_access
from space_exploration.FolderManager import FolderManager
from space_exploration.dataset.dataset_bean import Dataset, DatasetStat, Base
from space_exploration.simulation_channel.ChannelData import ChannelData

from pathlib import Path
from sqlalchemy import create_engine


def add_dataset(session, name, s3_storage_name, raw_data_normalized, y_dimensions, u_means,
                v_means, w_means, u_stds, v_stds, w_stds):
    dataset = Dataset(
        name=name,
        s3_storage_name=s3_storage_name,
        raw_data_normalized=raw_data_normalized,
    )

    for i in range(len(y_dimensions) - 1):
        stat = DatasetStat(
            y_index=i,
            y_coord=float(y_dimension[i]),
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

    engine = create_engine(db_access.get_db_url())
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)

    session = Session()

    channel_data_file: Path = FolderManager.tfrecords / "scaling.npz"
    channel_data = ChannelData(channel_data_file)
    y_dimension = np.load(FolderManager.channel_coordinates / "coordY.npy")

    u_means = channel_data.U_mean.reshape(-1)
    v_means = channel_data.V_mean.reshape(-1)
    w_means = channel_data.W_mean.reshape(-1)

    u_stds = channel_data.U_std.reshape(-1)
    v_stds = channel_data.V_std.reshape(-1)
    w_stds = channel_data.W_std.reshape(-1)

    add_dataset(
        session=session,
        name="paper-validation",
        s3_storage_name="paper-dataset.zarr",
        raw_data_normalized=True,
        y_dimensions=y_dimension,
        u_means=u_means,
        v_means=v_means,
        w_means=w_means,
        u_stds=u_stds,
        v_stds=v_stds,
        w_stds=w_stds,
    )


    session.commit()