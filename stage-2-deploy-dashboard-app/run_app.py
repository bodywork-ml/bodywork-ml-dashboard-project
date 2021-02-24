"""
This module defines what will happen in 'stage-2-deploy-dashboard-app':

- load dataset and ML model from AWS S3;
- batch-score the dataset using the model;,
- compute model performance metrics and 'predicted vs. actual' plot; and,
- serve simple dashboard to present results to a user.
"""
import logging
import sys
from datetime import date
from typing import Dict
from urllib.request import urlopen

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
from joblib import load
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_percentage_error, max_error, r2_score


MODEL_URL = ('http://bodywork-ml-dashboard-project.s3.eu-west-2.amazonaws.com/'
             'models/regressor.joblib')

DATASET_URL = ('http://bodywork-ml-dashboard-project.s3.eu-west-2.amazonaws.com/'
               'datasets/regression-dataset.csv')

log = logging.getLogger(__name__)


def main() -> None:
    """Main script to be executed."""
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.setLevel(logging.INFO)

    date_stamp = date.today()
    dataset = get_dataset(DATASET_URL)
    model = get_model(MODEL_URL)

    dataset['y_pred'] = model.predict(dataset['X'].values.reshape(-1, 1))
    model_metrics = compute_model_metrics(dataset['y'], dataset['y_pred'])

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])
    app.config.requests_pathname_prefix = ''

    navbar = make_navbar()
    metrics_table = make_metrics_table(model_metrics)
    plot = make_scatter_plot(dataset, 'y', 'y_pred')
    info_alert = make_alert(model, date_stamp)

    app.layout = dbc.Container(
        [
            navbar,
            dbc.Row(
                [
                    dbc.Col(plot, width=8),
                    dbc.Col(metrics_table, width=4)
                ],
                align='center'
            ),
            dbc.Row(
                [
                    dbc.Col(info_alert, width=12)
                ],
                align='center'
            )
        ]
    )

    app.run_server(host='0.0.0.0', debug=False)


def make_navbar() -> dbc.Navbar:
    logo = ('https://bodywork-media.s3.eu-west-2.amazonaws.com/'
            'website_logo_transparent_background.png')

    navbar = dbc.Navbar(
        [
            dbc.Row(
                dbc.Col(html.Img(src=logo, height='50px'), width=12)
            )
        ],
        color='dark',
        dark=True
    )
    return navbar


def make_alert(model: BaseEstimator, model_date: date) -> dbc.Alert:
    text = f'Training metrics for model of class {type(model)} trained on {model_date}'
    return dbc.Alert(text, color='info')


def make_metrics_table(metrics: Dict[str, float]) -> dbc.Table:
    table_header = html.Thead(
        html.Tr([html.Th('Metric'), html.Th('Value')])
    )
    table_body = html.Tbody(
        [html.Tr([html.Td(k), html.Td(f'{v:.2f}')]) for k, v in metrics.items()]
    )
    table = dbc.Table([table_header, table_body], bordered=True, dark=True, striped=True)
    return table


def make_scatter_plot(data: pd.DataFrame, x: str, y: str) -> dcc.Graph:
    max_value = np.max([np.max(data[x].values), np.max(data[y].values)])
    plot = px.scatter(
        data,
        x=x,
        y=y,
        range_x=[0, max_value],
        range_y=[0, max_value],
        opacity=0.5,
        marginal_x='histogram',
        marginal_y='histogram',
        trendline='lowess',
        trendline_color_override='red',
        template='plotly_white'
    )
    plot.update_traces(marker={'color': 'rgb(124, 70, 165)'})
    return dcc.Graph(id='dataset', figure=plot)


def compute_model_metrics(
    y_actual: np.ndarray,
    y_predicted: np.ndarray
) -> Dict[str, float]:
    """Return regression metrics record."""
    mape = mean_absolute_percentage_error(y_actual, y_predicted)
    r_squared = r2_score(y_actual, y_predicted)
    max_residual = max_error(y_actual, y_predicted)
    metrics_record = {
        'MAPE': mape,
        'r_squared': r_squared,
        'max_residual': max_residual
    }
    return metrics_record


def get_dataset(url: str) -> pd.DataFrame:
    """Get data from cloud object storage."""
    print(f'downloading training data from {DATASET_URL}')
    data_file = urlopen(url)
    return pd.read_csv(data_file)


def get_model(url: str) -> BaseEstimator:
    """Get model from cloud object storage."""
    model_file = urlopen(url)
    return load(model_file)


if __name__ == '__main__':
    main()
