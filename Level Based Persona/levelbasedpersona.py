#############################################
# PROJECT: LEVEL BASED PERSONA DEFINITION, SEGMENTATION & RULE BASED SEGMENTATION
#############################################

import pandas as pd
import numpy as np

users = pd.read_csv("odevler/dataset/users.csv")
users.head()
users.shape

purchases = pd.read_csv("odevler/dataset/purchases.csv")
purchases.head()
purchases.shape

df = purchases.merge(users, how="inner", on="uid")
df.shape
df.head()

agg_df = df.groupby(["country", "device", "gender", "age"]).agg({"price": "sum"}).sort_values(ascending=False, by="price")
agg_df.head()

# turn indexes into columns
agg_df.reset_index(inplace=True)
agg_df.head()
agg_df["age"].sort_values()

# the grouped ages' labels will be the ones that I gave as argument, the max value for the bin is given dynamically
bins = [0, 19, 24, 31, 41, agg_df["age"].max()]
mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["age"].max())]

agg_df["age_cat"] = pd.cut(agg_df["age"], labels=mylabels, bins=bins)

# group the information together
agg_df["level_based_customers"] = [row[0] + "_" + row[1].upper() + "_" + row[2] + "_" + row[5] for row in agg_df.values]

agg_df['level_based_customers'].nunique()

agg_df = agg_df.groupby("level_based_customers").agg({"price": "mean"})
agg_df = agg_df.reset_index()
agg_df["level_based_customers"].count()

# mean prices for each segment
agg_df["segment"] = pd.qcut(agg_df["price"], 4, labels=["D", "C", "B", "A"])
agg_df.head()

agg_df.groupby("segment").agg({"price":"mean"}).round(3)
pd.set_option('display.max_rows', None)
agg_df.sort_values(by="level_based_customers")

# a Turkish lady age :42 who uses IOS belongs to which segment?
pattern =r"TUR_IOS_F_[34]"
agg_df[agg_df["level_based_customers"].str.contains(pattern)]
new_user = "TUR_IOS_F_41_75"
agg_df.loc[agg_df["level_based_customers"] == new_user]