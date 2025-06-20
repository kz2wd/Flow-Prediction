import pandas as pd
import plotly.express as px

from space_exploration.beans.dataset_bean import Dataset
from space_exploration.dataset import db_access
from space_exploration.dataset.benchmarks.benchmark_keys import DatasetBenchmarkKeys

DATASET_VISUALIZATIONS = {}


def visualization(name):
    def decorator(func):
        DATASET_VISUALIZATIONS[name] = func
        return func
    return decorator


benchmarks_df = {}

def reload_combined_df():
    global benchmarks_df
    dss = db_access.get_session().query(Dataset).all()
    benchmarks = [ds.benchmark for ds in dss if ds.benchmark.loaded]
    if len(benchmarks) == 0:
        return
    benchmarks_df = {}
    for key in DatasetBenchmarkKeys:
        all_sub_benchmark = [b.benchmarks[key] for b in benchmarks if key in b.benchmarks]
        if len(all_sub_benchmark) == 0:
            continue
        benchmarks_df[key] = pd.concat(all_sub_benchmark, ignore_index=True)


reload_combined_df()

def get_combined_df(kind):
    return benchmarks_df[kind]

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
    combined_df = get_combined_df(DatasetBenchmarkKeys.VELOCITY_MEAN_ALONG_Y)
    filtered_df = combined_df[combined_df['dataset_id'].isin(ids)]
    filtered_df = filtered_df[filtered_df["component"] == "u"]

    fig = px.line(filtered_df, x="y_dimension", y=DatasetBenchmarkKeys.VELOCITY_MEAN_ALONG_Y, color="name", log_x=True)
    fig.update_layout(title="Dataset Velocities")
    return fig


@visualization("Stds")
def stds(ids):
    combined_df = get_combined_df(DatasetBenchmarkKeys.VELOCITY_STD_ALONG_Y)
    filtered_df = combined_df[combined_df['dataset_id'].isin(ids)]

    fig = px.line(filtered_df, x="y_dimension", y=DatasetBenchmarkKeys.VELOCITY_STD_ALONG_Y, color="name", line_dash="component")
    fig.update_layout(title="velocity_std")
    return fig



@visualization("squared_velocity_mean")
def squared_velocity_mean(ids):
    combined_df = get_combined_df(DatasetBenchmarkKeys.SQUARED_VELOCITY_MEAN_ALONG_Y)
    filtered_df = combined_df[combined_df['dataset_id'].isin(ids)]

    fig = px.line(filtered_df, x="y_dimension", y=DatasetBenchmarkKeys.SQUARED_VELOCITY_MEAN_ALONG_Y, color="name", line_dash="component", log_x=True)
    fig.update_layout(title="squared_velocity_mean")
    return fig


@visualization("reynold uv")
def reynold_uv(ids):
    combined_df = get_combined_df(DatasetBenchmarkKeys.REYNOLDS_UV)
    filtered_df = combined_df[combined_df['dataset_id'].isin(ids)]

    fig = px.line(filtered_df, x="y_dimension", y=DatasetBenchmarkKeys.REYNOLDS_UV, color="name")
    fig.update_layout(title="reynold uv")
    return fig

# @visualization("histogram")
# def histogram(ids):
#     pass
