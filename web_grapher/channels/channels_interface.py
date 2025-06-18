from dash import dcc, html, Output, Input, State
import plotly.graph_objects as go

from space_exploration.beans.channel_bean import Channel
from space_exploration.dataset.db_access import global_session
from web_grapher.channels.channel_visualizations import CHANNEL_VISUALIZATIONS



def get_channel_tab(app):
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

    tab = dcc.Tab(label="Channels",
                  children=[
                      html.Label("Select channels:"),
                      dcc.Dropdown(
                          id="channel-dropdown",
                          options=[{"label": ch.name, "value": ch.id} for ch in global_session.query(Channel).all()],
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
                  ]
              )
    return tab