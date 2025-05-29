# dash_visualization_tool.py

import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go

from space_exploration.beans.channel_bean import Channel
from space_exploration.beans.dataset_bean import Dataset
from space_exploration.dataset import db_access
from visualization.channel_visualizations import CHANNEL_VISUALIZATIONS
from visualization.dataset_visualizations import DATASET_VISUALIZATIONS

# --- Setup SQLAlchemy ---
session = db_access.get_session()

import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MORPH])

app.layout = html.Div([
    html.H1("Research Visualization Tool"),

    html.Label("Select Datasets:"),
    dcc.Dropdown(
        id="dataset-dropdown",
        options=[{"label": ds.name, "value": ds.id} for ds in session.query(Dataset).all()],
        multi=True
    ),

    html.Label("Select Visualization:"),
    dcc.Dropdown(
        id="viz-dropdown",
        options=[{"label": name, "value": name} for name in DATASET_VISUALIZATIONS.keys()],
        value="Compare Dataset Sizes"
    ),

    html.Button("Generate Graph", id="plot-btn", className="btn btn-secondary"),
    dcc.Graph(id="plot-output"),

    html.Label("Select channels:"),
    dcc.Dropdown(
        id="channel-dropdown",
        options=[{"label": ch.name, "value": ch.id} for ch in session.query(Channel).all()],
        multi=True
    ),

    html.Label("Select channel Visualization:"),
    dcc.Dropdown(
        id="viz-dropdown-channel",
        options=[{"label": name, "value": name} for name in CHANNEL_VISUALIZATIONS.keys()],
        value="y profile"
    ),

    html.Button("Generate Graph", id="plot-channel-btn", className="btn btn-secondary"),
    dcc.Graph(id="plot-channel-output"),
])

@app.callback(
    Output("plot-output", "figure"),
    Input("plot-btn", "n_clicks"),
    State("dataset-dropdown", "value"),
    State("viz-dropdown", "value")
)
def update_dataset_graph(n_clicks, selected_ids, viz_name):
    if not selected_ids or not viz_name:
        return go.Figure()
    fig = DATASET_VISUALIZATIONS[viz_name](selected_ids)
    return fig

@app.callback(
    Output("plot-channel-output", "figure"),
    Input("plot-channel-btn", "n_clicks"),
    State("channel-dropdown", "value"),
    State("viz-dropdown-channel", "value")
)
def update_channel_graph(n_clicks, selected_ids, viz_name):
    if not selected_ids or not viz_name:
        return go.Figure()
    fig = CHANNEL_VISUALIZATIONS[viz_name](selected_ids)
    return fig


if __name__ == '__main__':
    app.run(debug=True)
