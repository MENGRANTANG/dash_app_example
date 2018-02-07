import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import backend as K

stock_data = {}

stock_data["apple"] = pd.read_csv('https://www.dropbox.com/s/6dqxaprahtvny6f/aapl.us.txt?dl=1', sep=',',
                                  parse_dates=['Date'])
stock_data["amazon"] = pd.read_csv('https://www.dropbox.com/s/rbypc6vwhsudb7w/amzn.us.txt?dl=1', sep=',',
                                   parse_dates=['Date'])
stock_data["twitter"] = pd.read_csv('https://www.dropbox.com/s/se4e9dtj38svuk1/twtr.us.txt?dl=1', sep=',',
                                    parse_dates=['Date'])
stock_data["walmart"] = pd.read_csv('https://www.dropbox.com/s/ye17rgx6eb7e3kf/wmt.us.txt?dl=1', sep=',',
                                    parse_dates=['Date'])
stock_data["microsoft"] = pd.read_csv('https://www.dropbox.com/s/xo86gzacxoavyhd/msft.us.txt?dl=1', sep=',',
                                      parse_dates=['Date'])
for company in stock_data.keys():
    stock_data[company] = stock_data[company][stock_data[company]['Date'] >= '2014-01-01']
    stock_data[company]['Year'] = stock_data[company]['Date'].apply(lambda x: x.year)

app = dash.Dash(__name__)
server = app.server

available_companies = ['apple', 'amazon', 'twitter', 'walmart', 'microsoft']
available_labels = ['Open', 'High', 'Low', 'Close']
split_date = '2017-06-01'
stock_predictions = {}

for company in stock_data.keys():
    data = stock_data[company]
    training_set, test_set = data[data['Date'] < split_date], data[data['Date'] >= split_date]
    training_set = training_set.drop(['Date', 'OpenInt'], 1)
    test_set_copy = test_set.copy()
    test_set = test_set.drop(['Date', 'OpenInt'], 1)
    window_len = 10
    LSTM_training_inputs = []
    for i in range(len(training_set) - window_len):
        temp_set = training_set[i:(i + window_len)].copy()

        for col in list(temp_set):
            temp_set[col] = temp_set[col] / temp_set[col].iloc[0] - 1

        LSTM_training_inputs.append(temp_set)
    LSTM_training_outputs = (training_set['Close'][window_len:].values / training_set['Close'][:-window_len].values) - 1

    LSTM_test_inputs = []
    for i in range(len(test_set) - window_len):
        temp_set = test_set[i:(i + window_len)].copy()

        for col in list(temp_set):
            temp_set[col] = temp_set[col] / temp_set[col].iloc[0] - 1

        LSTM_test_inputs.append(temp_set)
    LSTM_test_outputs = (test_set['Close'][window_len:].values / test_set['Close'][:-window_len].values) - 1

    LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
    LSTM_training_inputs = np.array(LSTM_training_inputs)

    LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
    LSTM_test_inputs = np.array(LSTM_test_inputs)


    def build_model(inputs, output_size, neurons, activ_func="linear",
                    dropout=0.25, loss="mae", optimizer="adam"):
        K.clear_session()
        model = Sequential()

        model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
        model.add(Dropout(dropout))
        model.add(Dense(units=10))
        model.add(Activation(activ_func))
        model.add(Dense(units=10))
        model.add(Activation(activ_func))
        model.add(Dense(units=output_size))
        model.add(Activation(activ_func))
        model.compile(loss=loss, optimizer=optimizer)
        return model


    np.random.seed(202)

    nn_model = build_model(LSTM_training_inputs, output_size=1, neurons=10)

    nn_history = nn_model.fit(LSTM_training_inputs, LSTM_training_outputs,
                              epochs=200, batch_size=16, verbose=False)
    predicted_price = ((np.transpose(nn_model.predict(LSTM_test_inputs)) + 1) * test_set['Close'].values[:-window_len])[
        0].tolist()
    real_price = test_set['Close'][window_len:].tolist()
    time_list = list(test_set_copy['Date'][window_len:].apply(lambda x: str(x)[:10]).values)
    MAE = np.mean(np.abs(np.array(predicted_price) - np.array(real_price)))
    MAE = '%.4f' % MAE
    stock_predictions[company] = {}
    stock_predictions[company]["Time"] = time_list
    stock_predictions[company]["Real Price"] = real_price
    stock_predictions[company]["Predicted Price"] = predicted_price
    stock_predictions[company]["Mean Absolute Error"] = MAE

app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                id='company-name',
                options=[{'label': i, 'value': i} for i in available_companies],
                value='microsoft'
            )
        ],
            style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='y-type',
                options=[{'label': i, 'value': i} for i in available_labels],
                value=['High', 'Low'],
                multi=True
            ),
        ],
            style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),

    html.Div([
        dcc.Graph(
            id='stock-scatter'
        )
    ]),

    html.Div(dcc.Slider(
        id='year--slider',
        min=stock_data["apple"]['Year'].min(),
        max=stock_data["apple"]['Year'].max(),
        value=stock_data["apple"]['Year'].max(),
        step=None,
        marks={str(year): str(year) for year in stock_data["apple"]['Year'].unique()}
    ), style={'padding': '0px 20px 20px 20px'}),

    html.Div(id='MAE value'),

    html.Div([
        dcc.Graph(
            id='prediction-plot'
        )
    ]),
])

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


@app.callback(
    dash.dependencies.Output('stock-scatter', 'figure'),
    [dash.dependencies.Input('company-name', 'value'),
     dash.dependencies.Input('y-type', 'value'),
     dash.dependencies.Input('year--slider', 'value')])
def update_graph(company, y_type, year_value):
    df = stock_data[company]
    dff = df[df['Year'] == year_value]
    return {
        'data': [go.Scatter(
            x=dff['Date'],
            y=dff[y_type[i]],
            mode='lines+markers',
            name=y_type[i],
            marker={
                'size': 5,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            },
            line=dict(
                shape='spline'
            )
        ) for i in range(len(y_type))
        ],
        'layout': go.Layout(
            xaxis={
                'title': "Time"
            },
            yaxis={
                'title': 'Price'
            },
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            height=450,
            hovermode='closest'
        )
    }


@app.callback(
    dash.dependencies.Output(component_id='MAE value', component_property='children'),
    [dash.dependencies.Input(component_id='company-name', component_property='value')]
)
def update_output_div(company_name):
    return 'The mean absolute error between prediction and real price of {} stock is {}'.format(company_name,
                                                                                                stock_predictions[company_name]['Mean Absolute Error'])


@app.callback(
    dash.dependencies.Output('prediction-plot', 'figure'),
    [dash.dependencies.Input('company-name', 'value')])
def update_prediction_graph(company_name):
    return {
        'data': [go.Scatter(
            x=stock_predictions[company_name]["Time"],
            y=stock_predictions[company_name]["Real Price"],
            mode='lines+markers',
            name="Real Price",
            marker={
                'size': 5,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            },
            line=dict(
                shape='spline'
            )
        ),
            go.Scatter(
                x=stock_predictions[company_name]["Time"],
                y=stock_predictions[company_name]["Predicted Price"],
                mode='lines+markers',
                name="Predicted Price",
                marker={
                    'size': 5,
                    'opacity': 0.5,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                line=dict(
                    shape='spline'
                )
            )
        ],
        'layout': go.Layout(
            xaxis={
                'title': "Time"
            },
            yaxis={
                'title': 'Price'
            },
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            height=450

        )
    }


if __name__ == '__main__':
    app.run_server(debug=True)
