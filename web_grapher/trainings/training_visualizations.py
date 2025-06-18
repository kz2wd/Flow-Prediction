from dash import dcc, html

from space_exploration.beans.training_bean import Training
from space_exploration.dataset.db_access import global_session


def get_training_tab(app):

    tab = dcc.Tab(label="Trainings",
                  children=[
                      html.Label("Select trainings:"),
                      dcc.Dropdown(
                          id="training-dropdown",
                          options=[{"label": tr.name, "value": tr.id} for tr in global_session.query(Training).all()],
                          multi=True
                      ),
                  ]
              )
    return tab