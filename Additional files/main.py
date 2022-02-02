# import dash for the visulization designing
import dash
# from dash import dcc
from dash import dcc
# import html
from dash import html
# import plotly.express for the graph generation
import plotly.express as px
# import pandas to hold dataframe
import pandas as pd
# import Input and Output form dash
from dash.dependencies import Input, Output
# graph object
import plotly.graph_objects as go

# creating app using dash
app = dash.Dash(__name__)

# a dictionary for color values
colors_vals = {
    # bg color value
    'background': '#69f5fa',
    # text color value
    'text': '#000001'
}

# create tabuler representation from the data frame
def generate_table_form(df):
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
    ])

# read the CSV file and create the data frame
df_all = pd.read_csv('25-09-2020-TO-24-09-2021ACCALLN.csv')
df_all = pd.read_csv('https://servicebuspro.000webhostapp.com/dataset/dataset1.csv')
# extract only two column 'Date' and 'Total Traded Quantity'
df = df_all[['Date','Total Traded Quantity']]
# generate the bar graph visualization
fig = px.bar(df, x="Date", y="Total Traded Quantity")



# generate the candle plot image
fig1 = go.Figure(data=[go.Candlestick(
    # set the data axis
    x=df_all['Date'],
    # set the opening price
    open=df_all['Open Price'],
    # set the max stock price in a day
    high=df_all['High Price'],
    # set min stock price in a day
    low=df_all['Low Price'],
    # set the closing price of the day
    close=df_all['Close Price'],
    # set the color for the bullish candle
    increasing_line_color= 'green',
    # set the color for the bearish candle
    decreasing_line_color= 'red'
)])

# update figure property
fig.update_layout(
    plot_bgcolor=colors_vals['background'],
    paper_bgcolor=colors_vals['background'],
    font_color=colors_vals['text']
)

# update figure property
fig1.update_layout(
    plot_bgcolor=colors_vals['background'],
    paper_bgcolor=colors_vals['background'],
    font_color=colors_vals['text'],
    height= 650
)

# define the structure of the HTML page
app.layout = html.Div(style={'backgroundColor': colors_vals['background']}, children=[
    # define the page title
    html.H1(
        children='Volume Graph Dashboard',
        # set H1 property
        style={
            'textAlign': 'center',
            'color': colors_vals['text']
        }
    ),

    # create a div to show sub-title
    html.Div(children='Shows Daily Trade Volume.', style={
        'textAlign': 'center',
        'color': colors_vals['text']
    }),

    # create a second visualization
    dcc.Graph(id='graph_with_slider'),
    html.Div(
        [
            # define a range slider
            dcc.RangeSlider(
                id='time-slider',
                min=0,
                max=len(df),
                value=[0,len(df)],
                marks={str(date): str(date) for date in range(len(df))},
                allowCross=False,
                step=None
            ),
        ],style={"display": "grid", "grid-template-columns": "30% 40% 10%"}),
    # add new line
    html.Br(),
    # add one more new line
    html.Br(),
    # candle plot
    html.Div(
        [dcc.Graph(figure=fig1)]),
    # header for table representation
    html.H4(children='NSE Past record'),
    # generate the table dynamically
    generate_table_form(pd.read_csv('25-09-2020-TO-24-09-2021ACCALLN.csv'))

])

# define callback to update the visualization
@app.callback(
    Output('graph_with_slider', 'figure'),
    Input('time-slider', 'value'))
# define method to update graph
def update_volume_graph(time_range):
    # filter the data frame
    filtered_df = df[time_range[0]:time_range[-1]]
    # generate the figure
    fig = px.bar(filtered_df, x="Date", y="Total Traded Quantity")
    # update the layout
    fig.update_layout(transition_duration=500)

    # update figure property
    fig.update_layout(
        plot_bgcolor=colors_vals['background'],
        paper_bgcolor=colors_vals['background'],
        font_color=colors_vals['text']
    )
    # return updated figure
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
