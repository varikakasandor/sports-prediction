import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import joblib

from tensorflow.keras import models


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from custom_utils import DateTransformer


def load_a_model():

    curr_dir=Path(__file__).resolve().parent
    model=models.load_model(Path(curr_dir,'Saved Neural Network Models','early_stopping_model'))
    pipeline=joblib.load(Path(curr_dir,'Saved Fitted Preprocessing Pipelines','simple_pipeline.pkl')) 


    df=pd.read_csv("results.csv")
    X=df[["date","home_team","away_team","country","neutral"]]
    y=df[["home_score","away_score"]]
    _, X_test, _, y_test = train_test_split(X, y, train_size=0.50)

    X_test=pipeline.transform(X_test)
    y_pred = model.predict(X_test)
    score = mean_absolute_error(y_test, y_pred)
    print(X_test)
    print(y_pred)
    print(y_test)
    print('MAE: ',score)

if(__name__=="__main__"):
    load_a_model()