# for the processing of the CSV data
import pandas as pd
# for the data exchanging
import json
# import dash for the web based GUI development
import dash
# for the implementation of the callback function
from dash.dependencies import Input, Output
# import core component
import dash_core_components as dcc
# import HTML component
import dash_html_components as html
# for the visuliation generation
import plotly.express as px

# create the app
app = dash.Dash(__name__)

# load the CSV data
df = pd.read_csv('25-09-2020-TO-24-09-2021ACCALLN.csv')

# define the layout of the dashboard
app.layout = html.Div([
    dcc.Graph(
        id='client-graph-px'
    ),
    dcc.Store(
        id='client-figure-store-px'
    ),
    # create a drop down to select the series from the data set
    'Series',
    dcc.Dropdown(
        id='client-graph-indicator-px',
        # define the component of the dropdown
        options=[
            {'label': 'Open Price', 'value': 'Open Price'},
            {'label': 'Close Price', 'value': 'Close Price'},
            {'label': 'High Price', 'value': 'High Price'},
            {'label': 'Low Price', 'value': 'Low Price'},
            {'label': 'Average Price', 'value': 'Average Price'}
        ],
        # bydefault shows the average price series
        value='Average Price'
    ),
    # create option for normal scale and logrethomic scale
    'Scaling option',
    # define the options for the radio
    dcc.RadioItems(
        id='client-graph-scale-px',
        options=[
            {'label': x, 'value': x} for x in ['linear', 'log']
        ],
        # default value is linear
        value='linear'
    ),
    # show a horizontal line
    html.Hr(),
    # show the data used for the visualization in JSON format
    html.Details([
        html.Summary('Visualization in JSON format'),
        dcc.Markdown(
            id='client-figure-json-px'
        )
    ])
])

# define callback for selection of the series
@app.callback(
    # define outout location
    Output('client-figure-store-px', 'data'),
    # define input data location
    Input('client-graph-indicator-px', 'value')
)
# implementation of the time-series
def update_store_data(series):
    # return the line graph based on the selected series
    return px.line(df, x='Date', y=str(series))

app.clientside_callback(
    """
    function(figure, scale) {
        if(figure === undefined) {
            return {'data': [], 'layout': {}};
        }
        const fig_r = Object.assign({}, figure, {
            'layout': {
                ...figure.layout,
                'yaxis': {
                    ...figure.layout.yaxis, type: scale
                }
             }
        });
        return fig_r;
    }
    """,
    # define input and output location
    Output('client-graph-px', 'figure'),
    Input('client-figure-store-px', 'data'),
    Input('client-graph-scale-px', 'value')
)
# callback for the JOSN data showing
@app.callback(
    # define the input data source
    Output('client-figure-json-px', 'children'),
    # define output location
    Input('client-figure-store-px', 'data')
)
def generated_px_figure_json(data):
    # return the JSON data used for the visualization
    return '```\n'+json.dumps(data, indent=3)+'\n```'

# create the application instance
if __name__ == '__main__':
    app.run_server(debug=True, port=8889)
