import pandas as pd
import numpy as np
import pycountry as pc


def preprocess_source():
    df=pd.read_csv("results.csv")
    df.drop(["tournament"],axis=1,inplace=True,errors="ignore")
    df.drop(["city"],axis=1,inplace=True,errors="ignore")
    #existing_countries=set(map(lambda x:x.name,list(pc.countries)))
    #df=df[df["home_team"].isin(existing_countries) & df["away_team"].isin(existing_countries)]
    df.to_csv("results.csv", index=False)


if(__name__=="__main__"):
    preprocess_source()