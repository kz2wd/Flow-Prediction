import plotly.graph_objects as go
from dash import html, dcc, Input, Output, State

from space_exploration.beans.dataset_bean import Dataset
from space_exploration.dataset.db_access import global_session
from web_grapher.datasets.dataset_visualizations import DATASET_VISUALIZATIONS, reload_combined_df


def get_datasets():
    return [ds for ds in global_session.query(Dataset).all() if ds.benchmark.loaded]


def get_dataset_tab(app):
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
        Output("dataset-dropdown", "options"),
        Input("reload-data-btn", "n_clicks"),

    )
    def reload_data(n_clicks):
        datasets = global_session.query(Dataset).all()

        for ds in datasets:
            ds.reload_benchmark()

        reload_combined_df()

        return [{"label": ds.name, "value": ds.id} for ds in datasets]

    tab = dcc.Tab(label="Dataset",
            children=[
                html.Button("Reload data", id="reload-data-btn", className="btn btn-secondary"),

                html.Label("Select Datasets:"),
                dcc.Dropdown(
                    id="dataset-dropdown",
                    options=[{"label": ds.name, "value": ds.id} for ds in get_datasets()],
                    multi=True
                ),

                html.Label("Select Visualization:"),
                dcc.Dropdown(
                    id="viz-dropdown",
                    options=[{"label": name, "value": name} for name in DATASET_VISUALIZATIONS.keys()],
                    value="Compare Dataset Sizes"
                ),

                html.Button("Generate Graph", id="plot-btn", className="btn btn-secondary"),
                dcc.Graph(id="plot-output")
            ]
        )
    return tab