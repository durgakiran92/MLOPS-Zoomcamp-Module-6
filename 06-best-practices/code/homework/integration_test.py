import os
import pandas as pd
from datetime import datetime
import mlflow
import pickle
import s3fs
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from batch import save_data


# Set up S3 and MLflow environment variables
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")
S3_BUCKET = os.getenv("S3_BUCKET", "mlops-zoomcamp")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = S3_ENDPOINT_URL


def generate_test_data():
    data = {
        "ride_id": ["001", "002", "003"],  # <-- Add this line
        "PULocationID": [1, 2, 3],
        "DOLocationID": [1, 2, 3],
        "trip_distance": [1.0, 2.0, 3.0],
        "tpep_pickup_datetime": [
            datetime(2023, 1, 1, 0, 0, 0),
            datetime(2023, 1, 1, 1, 0, 0),
            datetime(2023, 1, 1, 2, 0, 0),
        ],
        "tpep_dropoff_datetime": [
            datetime(2023, 1, 1, 0, 10, 0),
            datetime(2023, 1, 1, 1, 20, 0),
            datetime(2023, 1, 1, 2, 30, 0),
        ],
    }
    df = pd.DataFrame(data)
    save_data(df, "s3://mlops-zoomcamp/test_data_2023-01.parquet")

    print("✅ Test data written to: s3://mlops-zoomcamp/test_data_2023-01.parquet")



def train_and_log_model():
    # Fake training data
    data = [
        (1.0, "1_1", 10.0),
        (2.0, "2_2", 20.0),
        (3.0, "3_3", 30.0)
    ]
    df_train = pd.DataFrame(data, columns=["trip_distance", "PU_DO", "duration"])
    X_train = df_train[["trip_distance", "PU_DO"]].to_dict(orient="records")
    y_train = df_train["duration"]

    dv = DictVectorizer()
    X_train_vectorized = dv.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_vectorized, y_train)

    pipeline = Pipeline([
        ("dv", dv),
        ("model", model)
    ])

    

    # Log the model with MLflow
    # Log the model with MLflow
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        run_id = run.info.run_id


    return run_id


def test_batch_pipeline():
    generate_test_data()

    print("🧪 Starting integration test...")

    run_id = train_and_log_model()
    assert run_id is not None, "❌ RUN_ID not set"

    os.environ["RUN_ID"] = run_id

    # Run batch.py
    exit_code = os.system("python batch.py 2023 1")
    assert exit_code == 0, "❌ batch.py failed to run"

    # Read prediction output
    output_file = f"s3://{S3_BUCKET}/predictions_2023-01.parquet"
    df_result = pd.read_parquet(output_file, filesystem=s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT_URL}))

    pred_sum = df_result["predicted_duration"].sum()
    print(f"✅ Sum of predicted durations: {pred_sum}")
    return pred_sum


if __name__ == "__main__":
    result = test_batch_pipeline()
    print(f"🎯 Integration test succeeded with predicted duration sum = {result}")
