import dash
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import Input, Output, dcc, html

from dash import html

# import pandas to hold dataframe
import pandas as pd

import yfinance as yf

FONT_AWESOME = "https://use.fontawesome.com/releases/v5.10.2/css/all.css"

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}



# create tabuler representation from the data frame
def generate_table_form(df):
    '''
    # return the tabuler representation
    return html.Table([
        # create table header
        html.Thead(
            html.Tr([html.Th(column) for column in df.columns])
        ),
        # define the table body
        html.Tbody([
            # populate the row element
            html.Tr([
                html.Td(df.iloc[i][column]) for column in df.columns
            ]) for i in range(len(df))
        ])
    ])'''
    return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME])

val =  yf.download("BSE.NS", period='1d').reset_index()

date_val="Date: "+str(val.Date)[2:14]+"\n"

sx = yf.Ticker("^BSESN")
nf = yf.Ticker("^NSEI")
cr = yf.Ticker("ITC.NS")

print(sx.info)

sensex =  dbc.CardGroup([
    dbc.CardHeader("SENSEX Value"),
    dbc.CardBody(
        [
            html.H5(date_val, className="card-title"),
            html.P(
                "Current index value: "+str(sx.info['regularMarketPrice']),
                className="card-text",
            ),
        ]
    ),

])

nifty =  dbc.CardGroup([
    dbc.CardHeader("NIFTY Value"),
    dbc.CardBody(
        [
            html.H5(date_val, className="card-title"),
            html.P(
                "Current index value: "+str(nf.info['regularMarketPrice']),
                className="card-text",
            ),
        ]
    ),

])

current  =  dbc.CardGroup([
    dbc.CardHeader("Current Stock"),
    dbc.CardBody(
        [
            html.H5(date_val, className="card-title"),
            html.P(
                "Current price value: "+str(cr.info['regularMarketPrice']),
                className="card-text",
            ),
        ]
    ),


])
cards = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(dbc.Card(sensex, color="dark", inverse=True)),
                dbc.Col(
                    dbc.Card(nifty, color="dark", inverse=True)
                ),
                dbc.Col(dbc.Card(current, color="dark", inverse=True)),
            ],
            className="mb-4",
        ),

    ]
)


app.layout = dbc.Container(
    [
        html.Br(),
        dbc.Row(dbc.Col([cards], md=12)),
        dcc.Store(id="store"),
        html.H1("Stock Market Data Visualization and Prediction Dashboard"),
        html.Hr(),

        dbc.Button(
            "Regenerate graphs",
            color="primary",
            id="button",
            className="mb-3",
        ),
        dbc.Tabs(
            [
                dbc.Tab(label="Time Series Representation", tab_id="line_graph",
                    children=[

                        # create a drop down to select the series from the data set
                        'Series',
                        dcc.Dropdown(
                            id='client-graph-indicator-px',
                            # define the component of the dropdown
                            options=[
                                {'label': 'Open Price', 'value': 'Open'},
                                {'label': 'Close Price', 'value': 'Close'},
                                {'label': 'High Price', 'value': 'High'},
                                {'label': 'Low Price', 'value': 'Low'},
                                {'label': 'Average Price', 'value': 'Adj Close'}
                            ],
                            # bydefault shows the average price series
                            value='Adj Close'
                        ),
                ]),
                dbc.Tab(label="Candle plot", tab_id="candle"),
                dbc.Tab(label="Sales Volume", tab_id="bar"),
                dbc.Tab(label="Tabular Representation", tab_id="table"),
            ],
            id="tabs",
            active_tab="line_graph",
        ),
        html.Div(id="tab-content", className="p-4"),
    ],style={'backgroundColor':'#f0f0f5'}
)


@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"), Input("store", "data")],
)
def render_tab_content(active_tab, data):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if active_tab and data is not None:
        if active_tab == "line_graph":
            return dcc.Graph(figure=data["line_graph"])
        elif active_tab == "bar":
            return dcc.Graph(figure=data["bar"])
        elif active_tab == "candle":
            return dcc.Graph(figure=data["candle"])
        elif active_tab == "table":
            return data["table"]

    return "No tab selected"


@app.callback(Output("store", "data"),  [Input('client-graph-indicator-px', 'value')])
def generate_graphs(series):


    # read the CSV file and create the data frame
    #df = pd.read_csv('25-09-2020-TO-24-09-2021ACCALLN.csv')

    df= yf.download("ITC.NS").reset_index()

    # generate the candle plot image
    candle = go.Figure(data=[go.Candlestick(
        # set the data axis
        x=df['Date'],
        # set the opening price
        open=df['Open'],
        # set the max stock price in a day
        high=df['High'],
        # set min stock price in a day
        low=df['Low'],
        # set the closing price of the day
        close=df['Close'],
        # set the color for the bullish candle
        increasing_line_color= 'green',
        # set the color for the bearish candle
        decreasing_line_color= 'red'
    )])

    candle.update_layout(
        title_text="Candle Plot",
        height= 700
    )

    # Create figure
    trade_vol = go.Figure()
    trade_vol.add_trace(go.Bar(x=pd.to_datetime(df['Date'][-365:]).dt.strftime('%Y-%m-%d'),y=df["Volume"][-365:]))
    # Add range slider
    trade_vol.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),

                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    trade_vol.update_layout(
        title_text="Trade Volume Visualization",
        height= 600
    )



    line_graph=go.Figure()
    line_graph.add_trace(go.Scatter(x=pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d'), y=df[str(series)]))
    line_graph.update_layout(
        title_text="Time Series Representation",
        height= 500
    )
    line_graph.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    # save figures in a dictionary for sending to the dcc.Store
    return {"line_graph": line_graph, "candle":candle,"bar": trade_vol, "table": generate_table_form(df.tail(100))}



# define callback for selection of the series
#@app.callback(
#    # define outout location
#    Output('client-figure-store-px', 'data'),
#    # define input data location
#    Input('client-graph-indicator-px', 'value')
#)
# implementation of the time-series
#def update_store_data(series):
#    # return the line graph based on the selected series
#    return go.Scatter(x=df['Date'], y=df[str(series)])#df, x='Date', y=str(series))



if __name__ == "__main__":
    app.run_server(debug=True, port=8888)
