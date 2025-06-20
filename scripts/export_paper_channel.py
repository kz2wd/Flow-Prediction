import numpy as np

from space_exploration.beans.database_add import add_channel
from space_exploration.FolderManager import FolderManager
from space_exploration.dataset.db_access import global_session

if __name__ == '__main__':

    y_dim = np.load(FolderManager.channel_coordinates / "coordY.npy")
    add_channel(
                "paper-channel",
                64, np.pi,
                y_dim,
                64, np.pi / 2,
                200,
                discard_first_y=True,
                )

    global_session.commit()