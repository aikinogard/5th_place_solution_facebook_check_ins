import pandas as pd

df = pd.read_csv("train.csv")
t_max = df["time"].max()
df[df["time"] <= t_max - 171360].to_csv("train-tr.csv", index=False)
df[df["time"] > t_max - 171360].to_csv("train-va.csv", index=False)

#t_max=786239
#take 21.8% of training set as validation