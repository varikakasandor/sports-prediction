import pandas as pd

def inspect_source():
    df=pd.read_csv("results.csv")
    print(len(df["home_team"].unique()))
    print(df.head())
    print(df["home_team"].unique())
    byteam=df.groupby(by="home_team")["date"].count()
    print(byteam)
    byteam.to_csv("Team Statistics.csv")

if(__name__=="__main__"):
    inspect_source()