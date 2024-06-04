from sklearn.metrics import mean_squared_error
import scipy
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import pandas as pd
from typing import List, Tuple
import mlflow





if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter



@data_exporter
def train(data: Tuple[scipy.sparse.csr_matrix, pd.DataFrame , DictVectorizer], *args, **kwargs)-> LinearRegression:
    EXPERIMENT_NAME = "linear-regression"

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog()

    X_train, training_set, dv = data
    target = 'duration'
    y_train = training_set[target].values



    lr = LinearRegression()
    lr.fit(X_train, y_train)
        
    mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="sklearn-model",
        input_example=X_train,
        registered_model_name="sk-learn-linear-regressiong-model",
    )

    mlflow.sklearn.log_model(
        sk_model=dv,
        artifact_path="vectorizer",
        input_example=X_train,
        registered_model_name="dict-vectorizer",
    )

    print(lr.intercept_)



    return lr, dv

