import seaborn as sns
import pandas as pd
import missingno as msno
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

pd.set_option('display.max_columns', None)

df = sns.load_dataset("planets")
df.head()

df.info()

####################################################
# CHECK THE NUMBER OF MISSING VALUES
####################################################

# are there any null values?
df.isnull().values.any()

# sum of null values per each variable
df.isnull().sum()

# null value over total value ratio per each variable, sorted descendingly
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

# 50% of mass values, 21% of distance values and 4% of orbital_period values are missing

# observations with more than one missing values
df[df.apply(lambda x: sum(x.isnull().values), axis = 1) > 1]
len(df[df.apply(lambda x: sum(x.isnull().values), axis = 1) > 1])     #244

# observations with more than two missing values
df[df.apply(lambda x: sum(x.isnull().values), axis = 1) > 2]
len(df[df.apply(lambda x: sum(x.isnull().values), axis = 1) > 2])     #11


"""
if the missing values in different variables are correlated, those values contain an important information, so shouldn't be dropped or reassigned
if the missing values appeared by chance, then they can be dropped or reassigned

"""

####################################################
# VISUALIZE (PLOT)
####################################################

# plot each variable's notnull value counts, that will be used for comparison
msno.bar(df)
plt.show()
# plot to check if null values in each variable have a correlation or not
msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()
# mass and orbital_period have a positive correlation of 0.2
# there is a 0.5 positive correlation between mass and distance


####################################################
# SOLUTIONS FOR HANDLING MISSING VALUES
####################################################

# let's assume that the missing values appeared by chance. I didn't assign in order not to change the df

# DROP

# drop all
df.dropna()

# drop row with all values missing
df.dropna(how="all")
df.shape[0] - df.dropna(how="all").shape[0]    # there aren't any in this df

# keep the rows where there are at least 2 notnull values
df.dropna(thresh=2)

# FILL VIA LAMBDA OR APPLY
missing_index = [998, 1001, 1015, 1027, 1029, 1030]
df.loc[missing_index]
df.describe().T

# fillna with the mean values of the rows, do nothing for the rows with object dtype (so it doesn't throw error)
df.apply(lambda x : x.fillna(x.mean()) if x.dtype != 'O' else x, axis=0).loc[missing_index]

# mark/flag  missing values
df.fillna("missing")

# ASSIGN VIA SCIKITLEARN

imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
imp_mean.fit([[1, 7,10], [5, np.nan, 4], [12, 4, np.nan]])
imp_mean.transform([[1, 7,10], [5, np.nan, 4], [12, 4, np.nan]])

"""
SimpleImputer arguments:
missing_values:the definition, the elements to be imputed
strategy: fill them with what? "mean/median/most frequent/a constant"
fill_value: the value for constant (if selected)

"""


