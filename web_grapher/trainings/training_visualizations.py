from space_exploration.beans.training_bean import Training
from space_exploration.dataset.db_access import global_session
from space_exploration.run_training import ModelTraining

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
    pass


