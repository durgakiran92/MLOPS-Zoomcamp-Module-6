import os
import pandas as pd
import pickle
import mlflow
from pathlib import Path

# ✅ STEP 2: Set up dynamic input path using env variable
def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)

# ✅ STEP 2: Set up dynamic output path using env variable
def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

# ✅ STEP 3: Updated to read from localstack S3 if env is set
def read_data(filename):
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
    if S3_ENDPOINT_URL:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }
        return pd.read_parquet(filename, storage_options=options)
    else:
        return pd.read_parquet(filename)

# ✅ STEP 4: Updated to write to localstack S3 if env is set
def write_data(df, filename):
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
    if S3_ENDPOINT_URL:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }
        df.to_parquet(filename, index=False, storage_options=options)
    else:
        df.to_parquet(filename, index=False)

def prepare_features(df, categorical):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].astype(str)

    return df

def load_model(model_path):
    with open(model_path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

def run():
    year = 2021
    month = 3

    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(input_file)
    df = prepare_features(df, categorical)

    dv, model = load_model('model.bin')
    X_val = dv.transform(df[categorical])
    y_pred = model.predict(X_val)

    print(f"Mean predicted duration: {y_pred.mean()}")

    df_result = df[['ride_id']].copy()
    df_result['predicted_duration'] = y_pred

    write_data(df_result, output_file)

if __name__ == "__main__":
    run()
