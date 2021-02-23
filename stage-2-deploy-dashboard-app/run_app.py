"""
This module defines what will happen in 'stage-2-deploy-dashboard-app':

- load dataset and ML model from AWS S3;
- batch-score the dataset using the model;,
- compute model performance metrics and 'predicted vs. actual' plot; and,
- serve simple dashboard to present results to a user.
"""
import logging
import re
import sys
from datetime import datetime, date
from io import BytesIO
from typing import Dict, Tuple

import boto3 as aws
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
from botocore.exceptions import ClientError
from joblib import load
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_percentage_error, max_error, r2_score


AWS_S3_BUCKET = 'bodywork-ml-dashboard-project'

log = logging.getLogger(__name__)


def main() -> None:
    """Main script to be executed."""
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.setLevel(logging.INFO)

    dataset, dataset_date = download_latest_dataset(AWS_S3_BUCKET)
    model, model_date = download_latest_model(AWS_S3_BUCKET)

    dataset['y_pred'] = model.predict(dataset['X'].values.reshape(-1, 1))
    model_metrics = compute_model_metrics(dataset['y'], dataset['y_pred'])

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])

    navbar = make_navbar()
    metrics_table = make_metrics_table(model_metrics)
    plot = make_scatter_plot(dataset, 'y', 'y_pred')
    info_alert = make_alert(model, model_date)

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

    app.run_server(debug=True)


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


def download_latest_dataset(aws_bucket: str) -> Tuple[pd.DataFrame, date]:
    """Get all available data from AWS S3 bucket.

    This function reads all CSV files from an AWS S3 bucket and then
    combines them into a single Pandas DataFrame object.
    """
    def _date_from_object_key(key: str) -> date:
        """Extract date from S3 file object key."""
        date_string = re.findall('20[2-9][0-9]-[0-1][0-9]-[0-3][0-9]', key)[0]
        file_date = datetime.strptime(date_string, '%Y-%m-%d').date()
        return file_date

    def _load_dataset_from_aws_s3(s3_obj_key: str) -> pd.DataFrame:
        """Load CSV datafile from AWS S3 into DataFrame."""
        object_data = s3_client.get_object(
            Bucket=aws_bucket,
            Key=s3_obj_key
        )
        return pd.read_csv(object_data['Body'])

    log.info(f'downloading all available training data from s3://{aws_bucket}/datasets')
    try:
        s3_client = aws.client('s3')
        s3_objects = s3_client.list_objects(Bucket=aws_bucket, Prefix='datasets/')
        object_keys_and_dates = [
            (obj['Key'], _date_from_object_key(obj['Key']))
            for obj in s3_objects['Contents']
        ]
        ordered_dataset_objs = sorted(object_keys_and_dates, key=lambda e: e[1])
        latest_dataset_obj = ordered_dataset_objs[-1]
        dataset = _load_dataset_from_aws_s3(latest_dataset_obj[0])
    except ClientError:
        log.info(f'failed to download training data from s3://{aws_bucket}/datasets')
    most_recent_date = latest_dataset_obj[1]
    return (dataset, most_recent_date)


def download_latest_model(aws_bucket: str) -> Tuple[BaseEstimator, date]:
    """Get latest model from AWS S3 bucket."""
    def _date_from_object_key(key: str) -> date:
        """Extract date from S3 file object key."""
        date_string = re.findall('20[2-9][0-9]-[0-1][0-9]-[0-3][0-9]', key)[0]
        file_date = datetime.strptime(date_string, '%Y-%m-%d').date()
        return file_date

    log.info(f'downloading latest model data from s3://{aws_bucket}/models')
    try:
        s3_client = aws.client('s3')
        s3_objects = s3_client.list_objects(Bucket=aws_bucket, Prefix='models/')
        object_keys_and_dates = [
            (obj['Key'], _date_from_object_key(obj['Key']))
            for obj in s3_objects['Contents']
        ]
        latest_model_obj = sorted(object_keys_and_dates, key=lambda e: e[1])[-1]
        latest_model_obj_key = latest_model_obj[0]
        object_data = s3_client.get_object(Bucket=aws_bucket, Key=latest_model_obj_key)
        model = load(BytesIO(object_data['Body'].read()))
        dataset_date = latest_model_obj[1]
    except ClientError:
        log.info(f'failed to download model from s3://{aws_bucket}/models')
    return (model, dataset_date)


if __name__ == '__main__':
    main()
