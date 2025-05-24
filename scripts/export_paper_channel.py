import numpy as np

from scripts.database_add import add_channel
from space_exploration.FolderManager import FolderManager
from space_exploration.beans.channel_bean import Channel
from space_exploration.beans.channel_y_bean import ChannelY
from space_exploration.dataset import db_access


if __name__ == '__main__':
    session = db_access.get_session()

    y_dim = np.load(FolderManager.channel_coordinates / "coordY.npy")
    add_channel(session,
                "paper-channel",
                64, np.pi,
                y_dim,
                64, np.pi / 2,
                200)

    session.commit()