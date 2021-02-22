"""
This module defines what will happen in 'stage-2-deploy-scoring-service':

- ONE;
- TWO; and,
- THREE.
"""
import logging
import re
import sys
from datetime import datetime, date
from io import BytesIO
from typing import Tuple

import boto3 as aws
import dash
import numpy as np
import pandas as pd
import plotly.express as px
from botocore.exceptions import ClientError
import dash_core_components as dcc
import dash_html_components as html
from joblib import load
from sklearn.base import BaseEstimator

AWS_S3_BUCKET = 'bodywork-ml-dashboard-project'

app = dash.Dash(
    __name__,
    external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']
)

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

log = logging.getLogger(__name__)


def main() -> None:
    """Main script to be executed."""
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.setLevel(logging.INFO)

    dataset = download_latest_dataset(AWS_S3_BUCKET)
    model = download_latest_model(AWS_S3_BUCKET)

    app.run_server(debug=True)


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
        dataset = pd.concat(
            _load_dataset_from_aws_s3(obj_key[0])
            for obj_key in ordered_dataset_objs
        )
    except ClientError:
        log.info(f'failed to download training data from s3://{aws_bucket}/datasets')
    most_recent_date = object_keys_and_dates[-1][1]
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
