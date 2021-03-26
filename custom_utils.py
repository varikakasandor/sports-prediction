import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DateTransformer(BaseEstimator, TransformerMixin):
        def __init__(self):
            super().__init__()

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            # Numerical features to pass down the numerical pipeline
            X_=X.copy()
            X_["date"]=X_["date"].apply(lambda d:float(d[0:4]+d[5:7]+d[8:10]))
            return X_


def create_integer_scores(arr):
    return np.rint(arr).astype(int)