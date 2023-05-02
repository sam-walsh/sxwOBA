# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

## TODO 
## -[] DOCUMENT CODE
## -[X] get PA slider working
## -[X] fix PA totals
## -[X] add player picture functionality from dropdown
## -[X] add player-specific plots
## -[] update font and theme
## -[X] fix player pictures
## -[X] add more player-specific plots
## -[] add pulled barrels and % to leaderboard

import plotly.express as px
import pandas as pd
import numpy as np
from dash import Dash, dash_table, html, dcc, callback_context, no_update
from dash.dependencies import Input, Output, State
from PIL import Image
import requests
from dash.exceptions import PreventUpdate
import sqlite3


app = Dash(__name__,
    meta_tags=[{'name':'viewport',
    'content':'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=1.0'}]
)
server = app.server

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

def get_data(year):
    conn = sqlite3.connect("sxwoba_data.db")
    df = pd.read_sql_query(f"SELECT * FROM spray_xwoba WHERE year = {year}", conn)
    bbe = pd.read_sql_query(f"SELECT * FROM bbe WHERE game_year = {year}", conn)

    df = df[["Name", "sxwOBA", "xwOBA", "diff%", "wOBA", "PA", "HR", "pulled_barrels", "Barrels", "BB%", "K%", "LA", "key_mlbam"]]
    df['diff%'] = df['diff%'].round(1)
    df['pulled_barrels'] = df['pulled_barrels'].astype(int)
    df['Barrels'] = df['Barrels'].astype(int)
    df = df.rename(columns={'pulled_barrels':'pbarrels', 'Barrels': 'barrels', 'key_mlbam':'id'})

    conn.close()
    return df, bbe


def comparison_scatter(data_frame):
    fig = px.scatter(data_frame,
        x="xwOBA",
        y="sxwOBA",
        color="diff%",
        color_continuous_scale="balance",
        trendline="ols",
        hover_name="Name",
        hover_data=["Name", "HR", "pbarrels", "BB%", "K%", "LA"],
        width=400,
        height=350)

    fig.update_layout(
        xaxis_range=[0.200, 0.5],
        yaxis_range=[0.200, 0.5],
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True)
    )
    return fig

def update_player_scatter(batter_id, bbe):
    player_bbe = bbe.loc[bbe['batter']==batter_id]

    bbe_scatter = px.scatter(data_frame=player_bbe,
        x='field_x',
        y='field_y',
        color='sxwOBA',
        opacity=0.9,
        color_continuous_scale='balance',
        hover_data=['launch_speed', 'launch_angle', 'pulled_barrel'],
        width=400,
        height=350)

    bbe_scatter.add_shape(type="rect",
        x0=-20, y0=-20, x1=380, y1=380,
        line=dict(
            color="green",
            width=5,
        ),
        opacity=0.2,
        layer="below",
        fillcolor="green",
    )
    bbe_scatter.add_shape(type="circle",
        xref="x", yref="y",
        x0=-360, y0=-360, x1=360, y1=360,
        line_color="yellow",
        opacity=0.5,
        layer="below",
        line=dict(width=3, dash='dash')
    )
    bbe_scatter.add_shape(type="circle",
        xref="x", yref="y",
        x0=-330, y0=-330, x1=330, y1=330,
        line_color="yellow",
        opacity=0.5,
        layer="below",
        line=dict(width=3, dash='dash')
    )
    bbe_scatter.add_shape(type="circle",
        xref="x", yref="y",
        x0=-300, y0=-300, x1=300, y1=300,
        line_color="yellow",
        opacity=0.5,
        layer="below",
        line=dict(width=3, dash='dash')
    )
    bbe_scatter.add_shape(type="circle",
        xref="x", yref="y",
        x0=-155, y0=-155, x1=155, y1=155,
        fillcolor="white",
        opacity=1,
        layer="below",
        line_color="white"
    )
    bbe_scatter.add_shape(type="circle",
        xref="x", yref="y",
        x0=-100, y0=-100, x1=160, y1=160,
        fillcolor="#efdcc3",
        opacity=1,
        layer="below",
        line_color="#efdcc3"
    )
    bbe_scatter.add_shape(type="rect",
        x0=-20, y0=-20, x1=0, y1=155,
        fillcolor="#b7d7c5",
        layer="below",
        opacity=1,
        line_color="#b7d7c5"
    )
    bbe_scatter.add_shape(type="rect",
        x0=-20, y0=-20, x1=155, y1=0,
        fillcolor="#b7d7c5",
        layer="below",
        opacity=1,
        line_color="#b7d7c5"
    )
    bbe_scatter.add_shape(type="rect",
        x0=0, y0=0, x1=90, y1=90,
        line=dict(
            color="white",
            width=5,
        ),
        layer="below",
        opacity=0.8,
    )
    bbe_scatter.add_shape(type="rect",
        x0=0, y0=0, x1=90, y1=90,
        fillcolor="white",
        layer="below",
        opacity=1,
        line_color="white"
    )
    bbe_scatter.add_shape(type="rect",
        x0=0, y0=0, x1=90, y1=90,
        fillcolor="#b7d7c5",
        layer="below",
        opacity=1,
        line_color="white"
    )
    bbe_scatter.add_shape(type="circle",
        xref="x", yref="y",
        x0=-15, y0=-15, x1=15, y1=15,
        fillcolor="#efdcc3",
        opacity=1,
        layer="below",
        line_color="#efdcc3")
    bbe_scatter.add_shape(type="circle",
        xref="x", yref="y",
        x0=50.5, y0=50.5, x1=70.5, y1=70.5,
        fillcolor="#efdcc3",
        opacity=1,
        layer="below",
        line_color="#efdcc3")
    
    bbe_scatter.update_traces(line=dict(dash='dash', width=3, color="white"),
        mode='markers',
        marker=dict(
            sizemode='area',
            sizeref=3.*bbe['sxwOBA'].max()/(25.**2),
            sizemin=4
        ))
    
    bbe_scatter.update_layout(
        xaxis_range=[-20,380],
        yaxis_range=[-20,380],
        xaxis = dict(
            tickmode='array',
            tickvals = [0, 90, 300, 330, 360],
            fixedrange=True
        ),
        yaxis = dict(
            tickmode='array',
            tickvals = [0, 90, 300, 330, 360],
            fixedrange=True
        ),
        dragmode=False
    )
    
    return bbe_scatter

df, bbe = get_data(2023)
dropdown_options = [{'label': row.Name, 'value': row.id} for row in df.itertuples()]

PAGE_SIZE = 10

# arenado_img = Image.open("player_images/571448.png")

app.layout = html.Div([
    dcc.Store(id='bbe-store', storage_type='memory', data=bbe.to_dict('records')),
    html.H1(children='sxwOBA: spray-angle enhanced xwOBA', style={'text-align': 'center'}),

    html.H2(children='Select a year and player:'),
    
    html.Div([
        dcc.Dropdown(
            value=dropdown_options[0]['value'],
            options=dropdown_options,
            id='dropdown',
            style={'display':'inline-block', 'width': '200px'}
        ),
        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': str(year), 'value': year} for year in range(2021, 2024)],
            value=2023,
            style={'display':'inline-block', 'width': '100px'}
        )
    ]),
    html.Div([
        html.Img(
            id='player-img',
            src=Image.open('player_images/680552.png'),
            className='row',
            style={'display':'inline-block', 'align-items': 'left', 'max-width': '50%'}
        ),
        dcc.Graph(id='barchart', className='row', style={'display':'inline-block', 'align-items': 'center', 'max-width': '50%'}),
        dcc.Graph(id='player-scatter', className='row', style={'display':'inline-block', 'align-items': 'right', 'max-width': '50%'})
    ]),    
    
    html.H2(children="leaderboards"),
    html.H6(children="* pbarrels: pulled barrels"),


    html.Div([
        dcc.Input(id='input_text', type='text', placeholder='Minimum PA'),
        html.Button('Update', id='update_button'),
        dash_table.DataTable(
            id='datatable-interactivity',
            columns=[
                {"name": i, "id": i, "deletable": False, "selectable": True} for i in df.columns
            ],
            data=df.to_dict('records'),
            style_data={
            'whiteSpace': 'normal',
            'height': '3%',
            },
            style_table={
                'overflowY': 'scroll',
            },
            editable=False,
            sort_action="native",
            sort_mode="single",
            column_selectable=False,
            row_selectable=False,
            row_deletable=False,
            selected_rows=[],
            page_action="native",
            page_current= 0,
            page_size=PAGE_SIZE,
            fill_width=False,
        )], style={'max-width':'90%'}),

    html.Div([
        html.Div([
            html.H2(children="Comparing xwOBA and sxwOBA"),
            html.H4(children="Select minimum plate appearances"),
            dcc.Slider(
                1,
                df['PA'].max(),
                value=50,
                tooltip={"placement": "bottom", "always_visible": True},
                id='pa-slider',
            ),
        ]),
        html.Div([
            dcc.Graph(
                id='comparison-scatter',
                figure=comparison_scatter(df),
                ),    
        ])

    ], style={'display':'inline-block'})

], style={'font-family': 'system-ui','display': 'flex', 'flex-direction': 'column', 'align-items': 'center'})

@app.callback(
    [Output('barchart', 'figure'),
     Output('player-scatter', 'figure'),
     Output('player-img', 'src'),
     Output('datatable-interactivity', 'data'),
     Output('comparison-scatter', 'figure'),
     Output('dropdown', 'options')],
    [Input('datatable-interactivity', 'selected_rows'),
     Input('dropdown', 'value'),
     Input('year-dropdown', 'value'),
     Input('pa-slider', 'value'),
     Input('update_button', 'n_clicks')],
    [State('datatable-interactivity', 'data'),
     State('input_text', 'value'),
     State('bbe-store', 'data')]
)
def update_outputs(chosen_rows, batter_dropval, year, selected_pa, n_clicks, data, minimum_pa, bbe_data):
    ctx = callback_context
    triggered_component = ctx.triggered[0]['prop_id'].split('.')[0]

    data_changed = False

    if triggered_component == 'year-dropdown':
        global df, bbe
        df, bbe = get_data(year)
        data = df.to_dict('records')
        bbe_data = bbe.to_dict('records')
        data_changed = True

    if triggered_component == 'update_button' and n_clicks is not None:
        data = df.loc[df['PA'] >= int(minimum_pa)].to_dict('records')
        data_changed = True

    if triggered_component == 'pa-slider':
        data = df.loc[df['PA'] >= selected_pa].to_dict('records')
        data_changed = True

    # Update player_scatter
    player_scatter_figure = update_player_scatter(batter_dropval, pd.DataFrame(bbe_data))

    # Update comparison_scatter
    comparison_scatter_figure = comparison_scatter(pd.DataFrame(data))

    # Update dropdown options
    dropdown_options = [{'label': d['Name'], 'value': d['id']} for d in data]


    if len(chosen_rows) == 0:
        df_filtered = df.loc[df['id'] == batter_dropval]
    else:
        df_filtered = df.loc[df['id'] == batter_dropval]

    # Update barchart
    barchart_figure = px.bar(data_frame=df_filtered, x='Name', y=['wOBA', 'xwOBA', 'sxwOBA'], text_auto='.3f', height=400, width=600, barmode='group')
    barchart_figure.update_layout(yaxis_range=[0,0.5])
    barchart_figure.update_xaxes(title='')
    barchart_figure.update_yaxes(title='')

    # Update player_img 
    
    if batter_dropval is not None:
        id = batter_dropval
        path = 'player_images/{}.png'.format(id)
        im = Image.open(path)
        player_img_src = im
    else:
        player_img_src = no_update

    return barchart_figure, player_scatter_figure, player_img_src, data, comparison_scatter_figure, dropdown_options

if __name__ == '__main__':
    app.run_server(debug=True, host = '127.0.0.1', dev_tools_hot_reload=False)
