import requests
from io import BytesIO
from typing import List

import pandas as pd
from pathlib import Path

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader


@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    file_path = '/home/src/mlops/homework_03/data_loaders/data/yellow_tripdata_2023-03.parquet'

    df = pd.read_parquet(file_path)

    return df