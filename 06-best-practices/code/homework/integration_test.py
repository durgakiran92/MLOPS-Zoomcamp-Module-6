import pandas as pd
import os

# from batch import read_data

def generate_test_data():
    data = {
        'PULocationID': [1, 2, 3],
        'DOLocationID': [4, 5, 6],
        'lpep_pickup_datetime': [
            '2023-01-01 00:00:00',
            '2023-01-01 01:00:00',
            '2023-01-01 02:00:00',
        ],
        'lpep_dropoff_datetime': [
            '2023-01-01 00:10:00',
            '2023-01-01 01:20:00',
            '2023-01-01 02:30:00',
        ],
    }

    return pd.DataFrame(data)

def main():
    year = 2023
    month = 1

    input_pattern = os.getenv("INPUT_FILE_PATTERN")
    input_file = input_pattern.format(year=year, month=month)

    S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")

    options = {
        "client_kwargs": {
            "endpoint_url": S3_ENDPOINT_URL,
            "region_name": "us-east-1",  # ← CRUCIAL
        }
    }

    df_input = generate_test_data()

    df_input.to_parquet(
        input_file,
        engine="pyarrow",
        compression=None,
        index=False,
        storage_options=options,
    )

    print(f"✅ Test data written to: {input_file}")

if __name__ == "__main__":
    main()
