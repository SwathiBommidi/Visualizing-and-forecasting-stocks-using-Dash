import dash
from dash import html
import dash_bootstrap_components as dbc
import yfinance as yf


FONT_AWESOME = "https://use.fontawesome.com/releases/v5.10.2/css/all.css"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME])

card_icon = {
    "color": "white",
    "textAlign": "center",
    "fontSize": 30,
    "margin": "auto",
}

import dash_bootstrap_components as dbc
#from dash import html

val =  yf.download("BSE.NS", period='1d').reset_index()

date_val="Date: "+str(val.Date)[2:14]+"\n"

sx = yf.Ticker("BSE.NS")
nf = yf.Ticker("INFY.NS")
cr = yf.Ticker("ITC.NS")

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

app.layout = dbc.Container(dbc.Row(dbc.Col([cards], md=12)))


if __name__ == "__main__":
    app.run_server(debug=True)
