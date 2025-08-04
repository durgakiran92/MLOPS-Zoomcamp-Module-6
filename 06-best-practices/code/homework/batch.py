# batch.py

import os
import pickle
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime


def read_data(filename, categorical):
    df = pd.read_parquet(filename)
    return prepare_data(df, categorical)


def prepare_data(df, categorical):
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


def train_model(df, categorical):
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)

    y_train = df['duration'].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)

    return model, dv, mse


def run_model(df, categorical, dv, model):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    y_val = df['duration'].values
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)

    return mse, y_pred


def save_model(dv, model, output_file):
    with open(output_file, 'wb') as f_out:
        pickle.dump((dv, model), f_out)


def main(year: int, month: int):
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_model_file = f'model_{year:04d}_{month:02d}.bin'
    output_predictions_file = f'taxi_type=yellow_year={year:04d}_month={month:02d}.parquet'

    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(input_file, categorical)

    # Train
    model, dv, train_mse = train_model(df, categorical)
    print(f"Training MSE: {train_mse:.3f}")

    # Validate on the same data (for simplicity)
    val_mse, y_pred = run_model(df, categorical, dv, model)
    print(f"Validation MSE: {val_mse:.3f}")

    # Save predictions to Parquet
    df_result = df.copy()
    df_result['prediction'] = y_pred
    df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df_result.index.astype('str')
    df_result[['ride_id', 'prediction']].to_parquet(output_predictions_file, index=False)

    # Save model
    save_model(dv, model, output_model_file)


if __name__ == '__main__':
    main(2023, 3)
