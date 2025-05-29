import numpy as np
import pandas as pd
import plotly.express as px

from space_exploration.beans.channel_bean import Channel
from space_exploration.dataset import db_access

CHANNEL_VISUALIZATIONS = {}

def visualization(name):
    def decorator(func):
        CHANNEL_VISUALIZATIONS[name] = func
        return func
    return decorator

def get_channels(ids):
    return db_access.get_session().query(Channel).filter(Channel.id.in_(ids)).all()

@visualization("y profile")
def y_profile(ids):
    channels = get_channels(ids)
    names = [ch.name for ch in channels]
    y_profiles = [ch.get_simulation_channel().y_dimension for ch in channels]

    # Flatten into a long-form DataFrame
    df = pd.DataFrame({
        'y': np.concatenate(y_profiles),
        'name': np.repeat(names, [len(y) for y in y_profiles]),
        'index': np.concatenate([np.arange(len(y)) for y in y_profiles])
    })

    fig = px.line(df, x="index", y="y", color="name")
    fig.update_layout(title="Channels y profiles", xaxis_title="Index", yaxis_title="y")
    return fig
