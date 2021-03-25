import numpy as np
import pandas as pd
import tensorflow as tf
import pycountry as pc

from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor

#import joblib
import pickle

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error


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



def model_training():
    df=pd.read_csv("results.csv")
    X=df[["date","home_team","away_team","country","neutral"]]
    y=df[["home_score","away_score"]]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.75)
    input_shape=[2+len(X_train["home_team"].unique())+len(X_train["away_team"].unique())]

    def create_model():
        neuralnet = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=input_shape),
            layers.Dense(512, activation='relu'),
            layers.Dense(2),
        ])
        neuralnet.compile(loss='mae',optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=['mse','accuracy'])
        return neuralnet

    model = KerasRegressor(build_fn=create_model, epochs=15, batch_size=64, verbose=1)

    preprocessor = ColumnTransformer(
        transformers=[
            ('date_scale',StandardScaler(),['date']),
            ('neutral', OrdinalEncoder(),['neutral']),
            ('home',OneHotEncoder(handle_unknown='ignore', sparse=False),['home_team']),
            ('away',OneHotEncoder(handle_unknown='ignore', sparse=False),['away_team'])
        ])

    pipeline = Pipeline(steps=[
        ('date',DateTransformer()),
        ('pre',preprocessor),
        ('model',model)
    ], verbose=True)

    pipeline.fit(X_train, y_train)
    #model.model.save("First_Model")
    preds = pipeline.predict(X_valid)
    score = mean_absolute_error(y_valid, preds)
    print(X_valid)
    print(preds)
    print(y_valid)
    print('MAE:', score)

if(__name__=="__main__"):
    model_training()