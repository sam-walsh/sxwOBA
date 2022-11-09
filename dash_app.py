# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

## TODO 
## -[X] get PA slider working
## -[X] fix PA totals
## -[X] add player picture functionality from dropdown
## -[X] add player-specific plots
## -[] update font and theme
## -[X] fix player pictures
## -[X] add more player-specific plots

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from dash import Dash, dash_table, html, dcc
from dash.dependencies import Input, Output
from PIL import Image
import requests


app = Dash(__name__)
server = app.server

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.read_csv("spray_xwoba.csv")
bbe = pd.read_csv("bbe.csv")
df = df.drop(['Unnamed: 0'], axis=1)
df= df[['batter_name', 'PA', 'xwOBA', 'sxwOBA', 'Barrels', 'diff', 'diff %', 'BB%', 'K%']]
df['BB%'] = df['BB%'].mul(100)
df['K%'] = df['K%'].mul(100)
print(df.info)
df = df.round(3)
names = df.batter_name.unique().tolist()


pitch_data = pd.read_csv("bbe.csv")
pitch_data['field_x'] = pitch_data.field_x.mul(2.4)
pitch_data['field_y'] = pitch_data.field_y.mul(2.4)

fig = px.scatter(df, x="xwOBA", y="sxwOBA", color="diff", hover_data=["batter_name", "PA"], color_continuous_scale="balance")
fig.add_shape(type="line",
    x0=0.2, y0=0.2, x1=0.45, y1=0.45,
    line=dict(color="RoyalBlue",width=3))

field_swoba = px.density_heatmap(pitch_data,
    x='field_x',
    y="field_y",
    z="sxwOBA",
    histfunc="avg",
    color_continuous_scale='balance',
    height=600,
    width=600)
fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
  )

field_swoba.add_hline(y=0, line_color="yellow")
field_swoba.add_vline(x=0, line_color="yellow")
field_swoba.layout.coloraxis.colorbar.title = "sxwOBA"
field_swoba.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
  )

field_xwoba = px.density_heatmap(pitch_data,
    x='field_x',
    y="field_y",
    z="estimated_woba_using_speedangle",
    histfunc="avg",
    color_continuous_scale="balance",
    height=600,
    width=600)
field_xwoba.add_hline(y=0, line_color="yellow")
field_xwoba.add_vline(x=0, line_color="yellow")
field_xwoba.layout.coloraxis.colorbar.title = "xwOBA"
field_xwoba.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1)

PAGE_SIZE = 10



app.layout = html.Div([
    html.H1(children='sxwOBA: spray angle enhanced xwOBA'),
    dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True} for i in df.columns
        ],
        data=df.to_dict('records'),
        editable=False,
        filter_action="native",
        sort_action="native",
        sort_mode="single",
        column_selectable="single",
        row_selectable="multi",
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size=PAGE_SIZE,
    ),
    html.Div(id='datatable-interactivity-container'),

    html.Div([
        html.Div([
            dcc.Slider(
                1,
                600,
                step=None,
                value=100,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True},
                id='pa-slider'
            ),
            dcc.Graph(
                id='example-graph',
                figure=fig
                ),
        ])
    ]),

    html.H1(children='comparing xwoba and sxwoba'),
    
    html.Div([
        dcc.Dropdown(
            names,
            value='nolan arenado',
            id='dropdown'
        ),

        html.Div([
            html.Img(
                id='player-img',
                src='https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/571448/headshot/67/current'
            )],
                style = {'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '3vw', 'margin-top': '3vw'}),

        html.Div([
            dcc.Graph(id='barchart')
        ], style={'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '3vw', 'margin-top': '3vw'}),
    ], className='row'),

    html.Div([
        dcc.Graph(id='player-scatter')
    ]),

    
    html.H1(children='field plots'),
    html.Div(children=[
        dcc.Graph(id='swoba-graph',
        style={'display': 'inline-block'},
        figure=field_swoba),
        dcc.Graph(id='xwoba-graph',
        style={'display': 'inline-block'},
        figure=field_xwoba)]),
])


## Callbacks

@app.callback(
    Output('barchart', 'figure'),
    [Input('datatable-interactivity', 'selected_rows'),
     Input('dropdown', 'value')]
)
def update_data(chosen_rows, batter_dropval):
    print(chosen_rows, batter_dropval)
    if len(chosen_rows)==0:
        df_filtered = df.loc[df['batter_name'] == batter_dropval]
    else:
        df_filtered = df.loc[df['batter_name'] == batter_dropval]
        print(df_filtered)

    barchart = px.bar(data_frame=df_filtered, x='batter_name', y=['xwOBA', 'sxwOBA'], barmode='group')


    return barchart 

@app.callback(
    Output('player-scatter', 'figure'),
    Input('dropdown', 'value')
)
def update_player_scatter(batter_name):
    url="https://i.imgur.com/oGNYVOR.png"
    im = Image.open(requests.get(url, stream=True).raw)
    player_bbe = bbe.loc[bbe['batter_name']==batter_name]

    bbe_scatter = px.scatter(data_frame=player_bbe,
        x='field_x',
        y='field_y',
        color='sxwOBA',
        size='sxwOBA',
        color_continuous_scale='balance',
        hover_data=['launch_speed', 'launch_angle', 'pulled_barrel'],
        width=800,
        height=700)

    return bbe_scatter

@app.callback(
    Output('player-img', 'src'),
    Input('dropdown', 'value')
)
def update_player_img(batter_name):
    from pybaseball import playerid_lookup
    if batter_name.isalpha():
        id = playerid_lookup(batter_name.split(' ')[-1], batter_name.split(' ')[0])['key_mlbam']
        print(id[0], type(id[0]))
        return'https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/{:04d}/headshot/67/current'.format(id.iloc[0])


@app.callback(
    Output('datatable-interactivity', 'style_data_conditional'),
    Input('datatable-interactivity', 'selected_columns')
)

def update_styles(selected_columns):
    return [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]

@app.callback(
    Output('example-graph', 'figure'),
    Input('pa-slider', 'value'))

def update_table(selected_pa):
    filtered_df = df[df.PA >= selected_pa]

    fig = px.scatter(filtered_df, x="xwOBA", y="sxwOBA", color="diff", hover_data=["batter_name", "PA", "diff %"])

    return fig

if __name__ == '__main__':
    app.run_server(debug=True, host = '127.0.0.1', dev_tools_hot_reload=False)
