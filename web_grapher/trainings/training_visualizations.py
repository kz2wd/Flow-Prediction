import pandas as pd
import plotly.express as px

from space_exploration.beans.training_bean import Training
from space_exploration.dataset.db_access import global_session
from space_exploration.training.training import ModelTraining

TRAINING_VISUALIZATIONS = {}


def visualization(name):
    def decorator(func):
        TRAINING_VISUALIZATIONS[name] = func
        return func
    return decorator

trainings = {}

def reload_trainings():
    global trainings
    trs = global_session.query(Training).all()
    trainings = {tr.id: ModelTraining.from_training_bean(tr) for tr in trs}


reload_trainings()


def mse_along_y(ids):
    # combined_df = get_combined_df(BenchmarkKeys.VELOCITY_MEAN_ALONG_Y)
    # filtered_df = combined_df[combined_df['training_id'].isin(ids)]
    #
    # fig = px.line(filtered_df, x="y_dimension", y=BenchmarkKeys.VELOCITY_MEAN_ALONG_Y, color="name", log_x=True)
    # fig.update_layout(title="Dataset Velocities")
    # return fig
    pass

