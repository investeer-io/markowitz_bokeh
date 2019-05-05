import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objs as go

import numpy as np

# input
N=100
returns_input = np.array([0.04,0.06])
volas_input = np.array([0.06,0.10])

corr_input=0

def covariance(corr_coef, volas):
    corr = np.array([[1, corr_coef],[corr_coef, 1]])
    cov = np.diag(volas) @ corr @ np.diag(volas)
    return cov

# def minvar(returns, cov)
#     minvar_w = np.divide(np.linalg.inv(cov) @ np.ones(len(returns)), np.ones(len(returns)).T @ np.linalg.inv(cov) @ np.ones(len(returns)))
#     w1 = np.linspace(0, 1, N)
#     w2=1-w1
#     w = np.array([w1,w2]).T
#     return w

def efffrontier(returns, cov):
    A = np.array(returns).T @ np.linalg.inv(cov) @ np.ones(len(returns))
    B = np.array(returns).T @ np.linalg.inv(cov) @ np.array(returns)
    C = np.ones(len(returns)).T @ np.linalg.inv(cov) @ np.ones(len(returns))
    D = B * C - A**2
    exp_return = np.linspace(min(returns), max(returns), num=100)
    sigma_return = np.sqrt(1/D * (C * np.multiply(exp_return, exp_return) - 2 * A * exp_return + B))

    return exp_return, sigma_return

cov_input = covariance(corr_input)

x, y = efffrontier(returns_input, cov_input)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Graph(id='graph-with-slider'),
    dcc.Slider(
        id='corr-slider',
        min=-1,
        max=1,
        value=0,
        step=0.1
    )
])


@app.callback(
    Output('graph-with-slider', 'figure'),
    [Input('corr-slider', 'value')])
def update_figure(selected_corr):
    cov_updated = covariance(selected_corr)
    out1, out2 = efffrontier(returns_input, cov_updated)
    traces = []
    traces.append(go.Scatter(
        x = out2,
        y = out1,
        mode='markers',
        opacity=0.7,
        marker={
            'size': 15,
            'line': {'width': 0.5, 'color': 'white'}
        },
    ))

    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'title': 'Sigma'},
            yaxis={'title': 'Mu'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1},
            hovermode='closest'
        )
    }


if __name__ == '__main__':
    app.run_server(debug=True)