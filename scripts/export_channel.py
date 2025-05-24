import numpy as np

from space_exploration.FolderManager import FolderManager
from space_exploration.beans.channel_bean import Channel
from space_exploration.beans.channel_y_bean import ChannelY
from space_exploration.dataset import db_access


def add_channel(name, x_resolution, x_length, y_dimension, z_resolution, z_length, y_scale_to_y_plus):
    channel = Channel(
        name=name,
        x_resolution=x_resolution,
        x_length=float(x_length),
        z_resolution=z_resolution,
        z_length=float(z_length),
        y_scale_to_y_plus=y_scale_to_y_plus
    )

    for i, coord in enumerate(y_dimension):
        channel_y = ChannelY(y_index=i, y_coord=float(coord))
        channel.y_dimension.append(channel_y)

    session.add(channel)

if __name__ == '__main__':
    session = db_access.get_session()

    y_dim = np.load(FolderManager.channel_coordinates / "coordY.npy")
    add_channel("paper-channel",
                64, np.pi,
                y_dim,
                64, np.pi / 2,
                200)

    session.commit()