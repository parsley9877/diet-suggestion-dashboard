from dash import dcc
from dash import html
import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_daq as daq
from datetime import date
from datetime import datetime
from datetime import timedelta

from ml_utils import utils

diet_data_path = './data/Food_Supply_kcal_Data.csv'
activity_data_path = './data/Food_Supply_kcal_Data.csv'
covid_data_path = './data/covid_cases__vaccination.csv'
diet_df = pd.read_csv(diet_data_path)
diet_df['standard'] = ['ISO3']*diet_df.shape[0]
activity_df = pd.read_csv(activity_data_path)
covid_df = pd.read_csv(covid_data_path)

covid_df = covid_df.loc[~covid_df['country'].isin(['Africa', 'Asia', 'Europe', 'European Union', 'International', 'North America', 'Oceania', 'South America', 'World', 'Timor'])]
diet_df['Country_names'] = diet_df['Country'].apply(utils.standard_country_names)
covid_df['country'] = covid_df['country'].apply(utils.standard_country_names)


state_map = dcc.Graph(id='state_map', style={'width': '100%'})
map_div = html.Div(children=[state_map])

selected_data = html.Div([
    dcc.Graph(id='selected-data'),
    dcc.Store(id='intermediate-value')
], style={'display': 'inline-block', 'width': '100%'}, className='loading_wrapper')

dp = dcc.DatePickerSingle(
    min_date_allowed=date(2020, 1, 1),
    max_date_allowed=date(2022, 2, 23),
    month_format='MMM Do, YY',
    placeholder='MMM Do, YY',
    date=date(2020, 2, 14),
    id='datepicker'
)

dp_div = html.Div(children=[html.A('Please pick a date  '), dp], className='dp-div')

# Dropdown

services = ["Alcoholic Beverages","Animal Products","Animal fats","Aquatic Products, Other","Cereals - Excluding Beer","Eggs","Fish, Seafood","Fruits - Excluding Wine","Meat","Milk - Excluding Butter","Miscellaneous","Offals","Oilcrops","Pulses","Spices","Starchy Roots","Stimulants","Sugar Crops","Sugar & Sweeteners","Treenuts","Vegetal Products","Vegetable Oils","Vegetables","Obesity","Undernourished","Confirmed","Deaths","Recovered","Active","Population"]
vac_services = []
red_scheme = ["Alcoholic Beverages", "Animal fats","Sugar Crops","Sugar & Sweeteners","Obesity","Confirmed","Deaths"]
blue_schemes = ["Aquatic Products, Other", "Animal Products", "Cereals - Excluding Beer","Eggs""Fish, Seafood","Fruits - Excluding Wine","Meat","Milk - Excluding Butter","Miscellaneous","Offals","Oilcrops","Pulses","Spices","Starchy Roots","Stimulants","Treenuts","Vegetal Products","Vegetable Oils","Vegetables", "Population"]
drop_down = dcc.Dropdown(services + vac_services, 'Alcoholic Beverages', id='dropdown')
dd_div = html.Div(children=[html.A('What to Visualize:  '), drop_down], className='dd-div')

# Title
header = html.Div(id='header', style={'backgroundColor': '#051833'}, children=[
    html.H1(children='Diet Suggestion and Data Analysis Dashboard', className='main-title')
])

# Buttons

but1 = html.Div(id='but-1', children=html.A('Insight', id='tab-1-nav', className='nav-buttons', href='tab1'))
# but2 = html.Div(id='but-2', children=html.A('Data Analysis', id='tab-2-nav', href='tab2', className='nav-buttons'))
but3 = html.Div(id='but-3', children=html.A('Prediction', id='tab-3-nav', href='tab3', className='nav-buttons'))

# Navbar
navbar = html.Div(id='navbar', className='top-nav', children=[
    but1,
    but3,
])

# Loader
dcc.Loading(
    parent_className='loading_wrapper',
    children=[selected_data]
)

# Bar chart
barchart = dcc.Graph(id='bar-chart', style={'width': '100%'})
barchart.__annotations__ = [
    {
        "text": "No matching data found",
        "xref": "paper",
        "yref": "paper",
        "showarrow": False,
        "font": {
            "size": 28
        }
    }
]
bar_div = html.Div(children=[barchart])
combo_box_pred_tab = dcc.Dropdown(['a', 'b'], 'California', id='dropdown-2', multi=True)
cmpt_div = html.Div(children=[html.A('Please select a US state:  '), combo_box_pred_tab], className='dd-div-2')
day_input_tab2 = daq.NumericInput(label='Time span (Days)', labelPosition='top', value=30,
                                  size=120, min=1, max=100, id='day-picker')
dit_div = html.Div(children=[day_input_tab2], className='di-div-2')

dp2 = dcc.DatePickerSingle(
    min_date_allowed=date(2020, 12, 14),
    max_date_allowed=date(2022, 2, 23),
    month_format='MMM Do, YY',
    placeholder='MMM Do, YY',
    date=date(2020, 12, 14),
    id='datepicker-2'
)

dp2_div = html.Div(children=[html.A('Please pick a date  ', className='label-dp-2'), dp2], className='dp-div-2')
dd3 = dcc.Dropdown(services + vac_services + ['Select Features ...'], id='dropdown-3', multi=True)
dd3_div = html.Div(children=[dd3], className='dd3-div')
but4 = html.Button('Get Pearson Correlation', id='submit-val', n_clicks=0)
but4_div = html.Div(children=[but4], className='but4-div')

second_row_tab_pred = html.Div(children=[cmpt_div, dit_div, dp2_div, dd3_div, but4_div], className='second-row-tab2')

cor_heatmap = dcc.Graph(id='cor-hm')
cor_heatmap_div = html.Div(children=[cor_heatmap], className='cor-hm-div')
pval_heatmap = dcc.Graph(id='pval-hm')
pval_heatmap_div = html.Div(children=[pval_heatmap], className='pvals-hm-div')
hp_div = html.Div(children=[cor_heatmap_div, pval_heatmap_div])

# Pred tab second section
combo_box_pred_tab_v_2 = dcc.Dropdown(['a', 'b'], 'California', id='dropdown-v-2', multi=False)
cmpt_div_v2 = html.Div(children=[html.A('Please select a US state:  '), combo_box_pred_tab_v_2], className='dd-div-v-2')
day_input_tab_v_2 = daq.NumericInput(label='Days Before', labelPosition='top', value=10,
                                     size=120, min=1, max=100, id='day-picker-v-2')
dit_div_v_2 = html.Div(children=[day_input_tab_v_2], className='di-div-v-2')

day_input_tab_v_2_e = daq.NumericInput(label='Days After', labelPosition='top', value=10,
                                       size=120, min=1, max=100, id='day-picker-v-2-e')
dit_div_v_2_e = html.Div(children=[day_input_tab_v_2_e], className='di-div-v-2-e')

dp2_v2 = dcc.DatePickerSingle(
    min_date_allowed=date(2020, 12, 14),
    max_date_allowed=date(2022, 2, 23),
    month_format='MMM Do, YY',
    placeholder='MMM Do, YY',
    date=date(2020, 12, 14),
    id='datepicker-v-2'
)
pred_line = dcc.Graph(id='pred-line')
pred_line_div = html.Div(children=[pred_line], className='pred-line-div')
dp2_div_v2 = html.Div(children=[html.A('Please pick a date  ', className='label-dp-v-2'), dp2_v2],
                      className='dp-div-v-2')
dd3_v2 = dcc.Dropdown(services + vac_services + ['Select Features ...'], id='dropdown-v-3')
dd3_div_v2 = html.Div(children=[dd3_v2], className='dd3-div-v-2')
but4_v2 = html.Button('Get Prediction', id='submit-val-v-2', n_clicks=0)
but4_div_v2 = html.Div(children=[but4_v2], className='but4-div-v-2')
third_row_tab_pred = html.Div(
    children=[cmpt_div_v2, dit_div_v_2, dit_div_v_2_e, dp2_div_v2, dd3_div_v2, but4_div_v2, pred_line_div],
    className='third-row-tab2')
#
first_row_tab_1 = html.Div(children=[dp_div, dd_div], className='first-row-tab1')
tab_1 = html.Div(className='tab-1', children=[first_row_tab_1, map_div, selected_data, bar_div])
# tab_2 = html.Div(className='tab-2', children=[])
tab_3 = html.Div(className='tab-3', children=[second_row_tab_pred, hp_div, third_row_tab_pred])

current_tab = html.Div(id='current-tab', children=[tab_1])

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = 'COVID-19 Dashboard'
app.layout = html.Div(id='layout', children=[dcc.Location(id='url', refresh=False),
                                             html.Div(children=[header, navbar, html.Div(children=[current_tab])])])

@app.callback(
    dash.dependencies.Output('bar-chart', 'figure'),
    [dash.dependencies.Input('state_map', 'selectedData'), dash.dependencies.Input('datepicker', 'date'),
     dash.dependencies.Input('dropdown', 'value')]
)
def update_bar(selectedData, date, value):
    if selectedData != None:
        fig = px.bar(selectedData['points'], x="location", y="z", title=value)
        fig.update_traces(marker_color='#04AA6D')
        fig.update_layout(margin={"r": 0, "t": 70, "l": 0, "b": 0})
        fig.update_layout(
            xaxis=go.layout.XAxis(
                tickangle=-45
            )
        )
        return fig
    else:
        return utils.empty_bar


# Navbar callbacks
@app.callback([dash.dependencies.Output('current-tab', 'children')],
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/tab1':
        return [tab_1]
    # elif pathname == '/tab2':
    #     return [tab_2]
    elif pathname == '/tab3':
        return [tab_3]
    else:
        return [tab_1]

@app.callback(
    dash.dependencies.Output('state_map', 'figure'),
    [dash.dependencies.Input('url', 'pathname'), dash.dependencies.Input('datepicker', 'date'),
     dash.dependencies.Input('dropdown', 'value')])
def update_state_map(url, date, value):

    global diet_df
    # diet_df['Country_names'] = diet_df['Country'].apply(utils.standard_country_names)

    if value in red_scheme:
        fig = go.Figure(data=go.Choropleth(
            locations=diet_df['Country_names'],  # Spatial coordinates
            z=diet_df[value].astype(float),  # Data to be color-coded
            colorscale='Reds',
        ))
    elif value in blue_schemes:
        fig = go.Figure(data=go.Choropleth(
            locations=diet_df['Country_names'],  # Spatial coordinates
            z=diet_df[value].astype(float),  # Data to be color-coded
            colorscale='Blues',
        ))

    else:
        fig = go.Figure(data=go.Choropleth(
            locations=diet_df['Country_names'],  # Spatial coordinates
            z=diet_df[value].astype(float),  # Data to be color-coded
            colorscale='mint',
        ))
    return fig

@app.callback(
    dash.dependencies.Output('selected-data', 'figure'),
    [dash.dependencies.Input('state_map', 'selectedData'), dash.dependencies.Input('dropdown', 'value'),
     dash.dependencies.Input('intermediate-value', 'data'), dash.dependencies.Input('datepicker', 'date')]
)
def update_select_data(selectedData, value, df_range, date):
    if selectedData == None:
        return utils.empty_bar
    global covid_df
    datetime_object = datetime.strptime(date, '%Y-%m-%d')
    start_date = datetime_object + timedelta(days=-20)  # extra day for death differences, remove later
    end_date = datetime_object + timedelta(days=20)
    date_range = pd.date_range(start=start_date, end=end_date, inclusive='right').strftime('%Y-%m-%d')
    covid_df_new = covid_df.loc[covid_df['date'].isin(date_range)]
    state_list = []
    for item in selectedData["points"]:
        state_list.append(item["location"])
    covid_df_new = covid_df_new.iloc[::-1]
    covid_df_new = covid_df_new.loc[covid_df_new['country'].isin(state_list)]
    df_by_states = covid_df_new[covid_df_new['daily_deaths'].notna()]

    fig = px.line(df_by_states, x='date', y='daily_deaths', color='country', title='daily_deaths' + '_by_day',
                  markers=True)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)