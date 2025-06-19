import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

from web_grapher.channels.channels_interface import get_channel_tab
from web_grapher.datasets.datasets_interface import get_dataset_tab
from web_grapher.trainings.training_interface import get_training_tab

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MORPH])

app.layout = html.Div([
    html.H1("Research Visualization Tool", style={"textAlign": "center"}),

    dcc.Tabs([
        get_dataset_tab(app),
        get_training_tab(app),
        get_channel_tab(app),

    ]),


], style={
    "maxWidth": "720px",
    "margin": "0 auto",
    "padding": "2rem"
})



if __name__ == '__main__':
    app.run(debug=True)
