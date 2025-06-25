import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from space_exploration.beans.training_bean import Training
from space_exploration.dataset.db_access import global_session
from space_exploration.training.training import ModelTraining
from space_exploration.training.training_benchmark_keys import TrainingBenchmarkKeys

TRAINING_VISUALIZATIONS = {}


def visualization(name):
    def decorator(func):
        TRAINING_VISUALIZATIONS[name] = func
        return func
    return decorator

trainings = {}
trainings_df = {}

def reload_trainings():
    global trainings
    global trainings_df
    trs = global_session.query(Training).all()
    trainings = {tr.id: ModelTraining.from_training_bean(tr) for tr in trs}

    benchmarks = [tr.get_benchmark() for tr in trainings.values()]
    for b in benchmarks:
        b.load()
    if len(benchmarks) == 0:
        return
    trainings_df = {}
    for key in TrainingBenchmarkKeys:
        all_sub_benchmark = [b.benchmarks[key] for b in benchmarks if key in b.benchmarks]
        if len(all_sub_benchmark) == 0:
            continue
        trainings_df[key.value] = pd.concat(all_sub_benchmark, ignore_index=True)


reload_trainings()

@visualization("MSE Along Y")
def mse_along_y(ids):
    combined_df = trainings_df[TrainingBenchmarkKeys.PAPER_LIKE_MSE_ALONG_Y.value]
    filtered_df = combined_df[combined_df['training_id'].isin(ids)]
    #
    fig = px.line(filtered_df, x="y_dimension", y=TrainingBenchmarkKeys.PAPER_LIKE_MSE_ALONG_Y, color="name", log_x=True, line_dash="component")
    fig.update_layout(title="Dataset Velocities")
    return fig

@visualization("U MSE Along Y")
def mse_along_y(ids):
    combined_df = trainings_df[TrainingBenchmarkKeys.PAPER_LIKE_MSE_ALONG_Y.value]
    filtered_df = combined_df[combined_df['training_id'].isin(ids)]
    filtered_df = filtered_df[filtered_df["component"] == "u"]
    fig = px.line(filtered_df, x="y_dimension", y=TrainingBenchmarkKeys.PAPER_LIKE_MSE_ALONG_Y, color="name", log_x=True)
    fig.update_layout(title="Dataset Velocities")
    fig.add_trace(go.Scatter(
        x=[15, 30, 50, 100],
        y=[0.043, 0.137, 0.306, 0.639],
        visible='legendonly',
        name="Paper U",
    ))
    return fig

