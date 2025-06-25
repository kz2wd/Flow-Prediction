import plotly.graph_objects as go
from dash import dcc, html, Input, State, Output

from space_exploration.beans.training_bean import Training
from space_exploration.dataset.db_access import global_session
from space_exploration.training.training import ModelTraining
from web_grapher.trainings.training_visualizations import TRAINING_VISUALIZATIONS


def get_training_tab(app):

    trainings = [ModelTraining.from_training_bean(tr) for tr in global_session.query(Training).filter(Training.benchmarked == True).all()]

    @app.callback(
        Output("training-plot-output", "figure"),
        Input("training-plot-btn", "n_clicks"),
        State("training-dropdown", "value"),
        State("training-viz-dropdown", "value")
    )
    def update_dataset_graph(n_clicks, selected_ids, viz_name):
        if not selected_ids or not viz_name:
            return go.Figure()
        fig = TRAINING_VISUALIZATIONS[viz_name](selected_ids)
        return fig

    tab = dcc.Tab(label="Trainings",
                  children=[
                      html.Label("Select trainings:"),
                      dcc.Dropdown(
                          id="training-dropdown",
                          options=[{"label": str(tr.bean.run_id)[:8], "value": tr.bean.id} for tr in trainings],
                          multi =True
                      ),
                      html.Label("Select Visualization:"),
                      dcc.Dropdown(
                          id="training-viz-dropdown",
                          options=[{"label": name, "value": name} for name in TRAINING_VISUALIZATIONS.keys()],
                          value=""
                      ),

                      html.Button("Generate Graph", id="training-plot-btn", className="btn btn-secondary"),
                      dcc.Graph(id="training-plot-output")
                  ]
              )
    return tab