import pandas as pd
from datetime import datetime
from pandas.testing import assert_frame_equal
from batch import prepare_data

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    

    categorical = ['PULocationID', 'DOLocationID']
    df = pd.DataFrame(data, columns=columns)

    expected_data = [
        (-1, -1, dt(1, 1), dt(1, 10), 9.0),
        (1, -1, dt(1, 2, 0), dt(1, 2, 59), 0.9833333333333333)
    ]

    expected_df = pd.DataFrame(expected_data, columns=columns + ['duration'])
    expected_df[categorical] = expected_df[categorical].astype('int').astype('str')

    result_df = prepare_data(df, categorical)

    assert result_df.shape[0] == expected_df.shape[0], f"Expected {expected_df.shape[0]} rows, but got {result_df.shape[0]}"

    #assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))
    #print(result_df)

#test_prepare_data()

