"""
This module defines what will happen in 'stage-1-train-model':

- create dataset using random number generators;
- train machine learning model on synthetic dataset; and,
- save model and dataset to cloud storage (AWS S3).
"""
import logging
import sys
from datetime import date

import boto3 as aws
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

AWS_S3_BUCKET = 'bodywork-ml-dashboard-project'
N_SAMPLES = 24 * 60


def main() -> None:
    """Main script to be executed."""
    date_stamp = date.today()

    log.info(f'creating synthetic dataset for date = {date_stamp}')
    dataset = generate_dataset(N_SAMPLES)

    log.info(f'training model for date = {date_stamp}')
    model = train_model(dataset)

    log.info(f'persisting dataset and model for date = {date_stamp}')
    persist_model(model, AWS_S3_BUCKET)
    persist_dataset(dataset, AWS_S3_BUCKET)


def generate_dataset(n: int) -> pd.DataFrame:
    """Create synthetic regression data using linear model with Gaussian noise."""
    datestr = np.full(n, str(date.today()))
    alpha = 0.5 * np.random.beta(2, 2)
    beta = 2 * np.random.beta(2, 2)
    sigma = 0.75

    X = np.random.uniform(0, 10, n)
    epsilon = np.random.normal(0, 1, n)
    y = alpha + beta * X + sigma * epsilon
    dataset = pd.DataFrame({'date': datestr, 'y': y, 'X': X})
    return dataset.query('y >= 0')


def train_model(data: pd.DataFrame) -> BaseEstimator:
    """Train regression model and compute metrics."""
    X = data['X'].values.reshape(-1, 1)
    y = data['y'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    ols_regressor = LinearRegression(fit_intercept=True)
    ols_regressor.fit(X_train, y_train)
    return ols_regressor


def persist_dataset(dataset: pd.DataFrame, aws_bucket: str) -> None:
    """Upload dataset metrics to AWS S3."""
    dataset_filename = 'regression-dataset.csv'
    dataset.to_csv(dataset_filename, header=True, index=False)
    try:
        s3_client = aws.client('s3')
        s3_client.upload_file(
            dataset_filename,
            aws_bucket,
            f'datasets/{dataset_filename}'
        )
        log.info(f'uploaded {dataset_filename} to s3://{aws_bucket}/datasets/')
    except Exception:
        log.error('could not upload dataset to S3 - check AWS credentials')


def persist_model(model: BaseEstimator, aws_bucket: str) -> None:
    """Upload trained model to AWS S3."""
    model_filename = 'regressor.joblib'
    dump(model, model_filename)
    try:
        s3_client = aws.client('s3')
        s3_client.upload_file(
            model_filename,
            aws_bucket,
            f'models/{model_filename}'
        )
        log.info(f'uploaded {model_filename} to s3://{aws_bucket}/models/')
    except Exception:
        log.error('could not upload model to S3 - check AWS credentials')


def configure_logger() -> logging.Logger:
    """Configure a logger that will write to stdout."""
    log_handler = logging.StreamHandler(sys.stdout)
    log_format = logging.Formatter(
        '%(asctime)s - '
        '%(levelname)s - '
        '%(module)s.%(funcName)s - '
        '%(message)s'
    )
    log_handler.setFormatter(log_format)
    log = logging.getLogger(__name__)
    log.addHandler(log_handler)
    log.setLevel(logging.INFO)
    return log


if __name__ == '__main__':
    log = configure_logger()
    main()
