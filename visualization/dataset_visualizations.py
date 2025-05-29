import pandas as pd
import plotly.express as px

from space_exploration.beans.dataset_bean import Dataset
from space_exploration.dataset import db_access

DATASET_VISUALIZATIONS = {}


def visualization(name):
    def decorator(func):
        DATASET_VISUALIZATIONS[name] = func
        return func
    return decorator

combined_df = pd.concat((ds.benchmark_df for ds in db_access.get_session().query(Dataset).all()), ignore_index=True)

def get_datasets(ids):
    return db_access.get_session().query(Dataset).filter(Dataset.id.in_(ids)).all()

@visualization("Compare Dataset Sizes")
def compare_dataset_sizes(ids):
    datasets = get_datasets(ids)
    names = [ds.name for ds in datasets]
    sizes = [ds.size for ds in datasets]
    fig = px.bar(x=names, y=sizes)
    fig.update_layout(title="Dataset Sizes")
    return fig

@visualization("U Velocities Along Y")
def u_velo_along_y(ids):
    filtered_df = combined_df[combined_df['dataset_id'].isin(ids)]
    filtered_df = filtered_df[filtered_df["component"] == "u"]

    fig = px.line(filtered_df, x="y_dimension", y="velocity_mean", color="name", log_x=True)
    fig.update_layout(title="Dataset Velocities")
    return fig


@visualization("Stds")
def stds(ids):
    filtered_df = combined_df[combined_df['dataset_id'].isin(ids)]
    filtered_df = filtered_df[filtered_df["component"] == "u"]

    fig = px.line(filtered_df, x="y_dimension", y="velocity_std", color="name")
    fig.update_layout(title="velocity_std")
    return fig



@visualization("squared_velocity_mean")
def squared_velocity_mean(ids):
    filtered_df = combined_df[combined_df['dataset_id'].isin(ids)]

    fig = px.line(filtered_df, x="y_dimension", y="squared_velocity_mean", color="name", line_dash="component", log_x=True)
    fig.update_layout(title="squared_velocity_mean")
    return fig


@visualization("reynold uv")
def reynold_uv(ids):
    filtered_df = combined_df[combined_df['dataset_id'].isin(ids)]

    fig = px.line(filtered_df, x="y_dimension", y="reynolds_uv", color="name")
    fig.update_layout(title="reynold uv")
    return fig

