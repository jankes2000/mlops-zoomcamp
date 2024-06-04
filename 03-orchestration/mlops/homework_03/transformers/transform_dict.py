from sklearn.feature_extraction import DictVectorizer

from typing import Dict, List, Optional, Tuple

import pandas as pd
import scipy

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(training_set: pd.DataFrame, *args, **kwargs) -> Tuple[scipy.sparse.csr_matrix, pd.DataFrame, DictVectorizer]:
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    train_dicts = training_set[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    # Specify your transformation logic here
    
    return X_train, training_set, dv

