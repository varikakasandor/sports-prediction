import numpy as np
import pandas as pd
import pycountry as pc
from pathlib import Path
import joblib

from tensorflow.keras import layers, models, optimizers, callbacks

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error

from custom_utils import DateTransformer


def model_training():
    df=pd.read_csv("results.csv")
    X=df[["date","home_team","away_team","country","neutral"]]
    y=df[["home_score","away_score"]]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.80)

    preprocessor = ColumnTransformer(
        transformers=[
            ('date_scale',StandardScaler(),['date']),
            ('neutral', OrdinalEncoder(),['neutral']),
            ('home',OneHotEncoder(handle_unknown='ignore', sparse=False),['home_team']),
            ('away',OneHotEncoder(handle_unknown='ignore', sparse=False),['away_team'])
        ])

    pipeline = Pipeline(steps=[
        ('date',DateTransformer()),
        ('pre',preprocessor)
    ], verbose=True)

    X_train=pipeline.fit_transform(X_train)
    X_valid=pipeline.transform(X_valid)
    input_shape=[X_train.shape[1]]

    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=input_shape),
        layers.Dense(512, activation='relu'),
        layers.Dense(2)
    ])
    model.compile(loss='mse',optimizer=optimizers.Adam(learning_rate=0.01))

    early_stopping = callbacks.EarlyStopping(
        patience=5,
        min_delta=0.001,
        restore_best_weights=True,
    )

    history=model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=500, batch_size=64,
        callbacks=[early_stopping],
        verbose=1)


    curr_dir=Path(__file__).resolve().parent
    with open("model_count.txt","r") as mc:
        cnt=mc.read()
        models.save_model(model,Path(curr_dir,'Saved Neural Network Models',f'model_{cnt}'))
        joblib.dump(pipeline, Path(curr_dir,'Saved Fitted Preprocessing Pipelines',f'pipeline_{cnt}.pkl'))
    with open("model_count.txt","w") as mc:
        new_cnt=str(int(cnt)+1)
        mc.write(new_cnt) 

if(__name__=="__main__"):
    model_training()