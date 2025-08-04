import os
import sys
import pandas as pd
import mlflow
import s3fs
import pickle


S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")
S3_BUCKET = os.getenv("S3_BUCKET", "mlops-zoomcamp")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = S3_ENDPOINT_URL

fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL})


def get_input_path(year, month):
    return f"s3://{S3_BUCKET}/test_data_{year:04d}-{month:02d}.parquet"


def get_output_path(year, month):
    return f"s3://{S3_BUCKET}/predictions_{year:04d}-{month:02d}.parquet"


def read_data(filename):
    return pd.read_parquet(filename, filesystem=fs)


def save_data(df, filename):
    df.to_parquet(filename, engine="pyarrow", index=False, filesystem=fs)


def apply_model(input_file, run_id):
    logged_model = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(logged_model)

    df = read_data(input_file)
    df["duration"] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    df["PU_DO"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)
    dicts = df[["PU_DO", "trip_distance"]].to_dict(orient="records")

    preds = model.predict(dicts)

    df_result = df[["ride_id", "tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID", "trip_distance"]].copy()
    df_result["predicted_duration"] = preds

    return df_result


def run():
    year = int(sys.argv[1])  # e.g. 2023
    month = int(sys.argv[2])  # e.g. 1

    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    run_id = os.getenv("RUN_ID")
    if run_id is None:
        raise ValueError("RUN_ID environment variable is not set.")

    df_result = apply_model(input_file, run_id)
    save_data(df_result, output_file)

    print(f"✅ Predictions saved to: {output_file}")


if __name__ == "__main__":
    run()
