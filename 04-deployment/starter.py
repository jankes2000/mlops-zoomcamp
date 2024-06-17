#!/usr/bin/env python
# coding: utf-8


import pickle
import pandas as pd
import statistics
import sys




with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename, year, month):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    return df

def save_results(df, y_pred, output_file):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']

    df_result['predicted_duration'] = y_pred
    df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)


if __name__ == '__main__':
     # year = 2023
    # month = 3
    year = int(sys.argv[1]) # 2021
    month = int(sys.argv[2]) # 3
    taxi_type = 'yellow'
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet', year, month)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)


    print(statistics.mean(y_pred))


    output_file = f'output/{taxi_type}_{year:04d}-{month:02d}.parquet'

    save_results(df, y_pred, output_file)




