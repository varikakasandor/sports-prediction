import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import joblib

from tensorflow.keras import models


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from custom_utils import DateTransformer, create_integer_scores


def load_a_model():

    curr_dir=Path(__file__).resolve().parent
    with open("model_count.txt","r") as mc:
        cnt=str(int(mc.read())-1)
        model=models.load_model(Path(curr_dir,'Saved Neural Network Models',f'model_{cnt}'))
        pipeline=joblib.load(Path(curr_dir,'Saved Fitted Preprocessing Pipelines',f'pipeline_{cnt}.pkl'))

    df=pd.read_csv("tests.csv")
    X_test=df[["date","home_team","away_team","country","neutral"]]
    y_test=df[["home_score","away_score"]]

    X_test=pipeline.transform(X_test)
    y_pred = model.predict(X_test)
    y_pred = create_integer_scores(y_pred)
    score = mean_absolute_error(y_test, y_pred)
    df[["pred_home_score","pred_away_score"]]=pd.DataFrame(y_pred)#, index=df.index)

    print(df)
    df.to_csv("VB_selejtezo.csv")
    print('MAE: ',score)

if(__name__=="__main__"):
    load_a_model()