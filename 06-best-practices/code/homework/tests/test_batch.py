import pandas as pd
from datetime import datetime
from batch import prepare_data

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),        # 9 min ✅
        (1, 1, dt(1, 2), dt(1, 10)),              # 8 min ✅
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),     # < 1 min ❌
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),         # > 60 min ❌
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']
    actual_df = prepare_data(df, categorical)

    # Only first two rows should be retained
    expected_data = [
        (None, None, dt(1, 1), dt(1, 10), 9.0),
        (1, 1, dt(1, 2), dt(1, 10), 8.0),
    ]
    expected_columns = columns + ['duration']
    expected_df = pd.DataFrame(expected_data, columns=expected_columns)

    # Check that DataFrames are equal (ignoring dtype differences)
    pd.testing.assert_frame_equal(actual_df.reset_index(drop=True), expected_df.reset_index(drop=True), check_dtype=False)
