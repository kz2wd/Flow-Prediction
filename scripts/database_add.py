
from space_exploration.beans.channel_bean import Channel
from space_exploration.beans.channel_y_bean import ChannelY
from space_exploration.beans.dataset_bean import Dataset
from space_exploration.beans.dataset_stat_bean import DatasetStat
from space_exploration.beans.training_bean import Training
from space_exploration.dataset.transforms.AllTransforms import TransformationReferences
from space_exploration.models.AllModels import ModelReferences


def add_dataset(session, name, channel, scaling, stats):
    dataset = Dataset(
        name=name,
        scaling=scaling,
        channel=channel,
    )

    for i in range(len(stats.u_means)):
        stat = DatasetStat(
            y_index=i,
            u_mean=float(stats.u_means[i]),
            v_mean=float(stats.v_means[i]),
            w_mean=float(stats.w_means[i]),
            u_std=float(stats.u_stds[i]),
            v_std=float(stats.v_stds[i]),
            w_std=float(stats.w_stds[i]),
        )
        dataset.stats.append(stat)

    session.add(dataset)

def add_channel(session, name, x_resolution, x_length, y_dimension, z_resolution, z_length, y_scale_to_y_plus, discard_first_y=False):
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

    session.add(channel)
    return channel


def try_add_training(session, model_name, dataset_name, x_transform_name, y_transform_name, batch_size, data_amount=-1, name=None):

    model_ref = ModelReferences(model_name)

    model = model_ref.model

    dataset = Dataset.get_dataset_or_fail(session, dataset_name)

    x_transform_ref = TransformationReferences(x_transform_name)
    y_transform_ref = TransformationReferences(y_transform_name)


    training = Training(

    )