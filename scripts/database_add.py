from space_exploration.beans.channel_bean import Channel
from space_exploration.beans.channel_y_bean import ChannelY
from space_exploration.beans.dataset_bean import Dataset
from space_exploration.dataset.db_access import global_session


def add_dataset(name, channel, scaling):
    dataset = Dataset(
        name=name,
        scaling=scaling,
        channel=channel,
    )
    global_session.add(dataset)
    return dataset

def add_channel(name, x_resolution, x_length, y_dimension, z_resolution, z_length, y_scale_to_y_plus, discard_first_y=False):
    channel = Channel(
        name=name,
        x_resolution=x_resolution,
        x_length=float(x_length),
        z_resolution=z_resolution,
        z_length=float(z_length),
        y_scale_to_y_plus=y_scale_to_y_plus,
        discard_first_y=discard_first_y
    )

    for i, coord in enumerate(y_dimension):
        channel_y = ChannelY(y_index=i, y_coord=float(coord))
        channel.y_dimension.append(channel_y)

    global_session.add(channel)
    return channel
