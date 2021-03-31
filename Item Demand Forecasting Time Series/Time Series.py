######################################################
# ITEM DEMAND FORECASTING WITH TIME SERIES ANALYSIS
######################################################

import numpy as np
import pandas as pd
import lightgbm as lgb
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore')
pd.set_option('display.width', None)

# Train and test datasets are being concatenated, because I will need the past in order to predict the future based on
# some possible patterns that occured in the past.

train = pd.read_csv("../input/demand-forecasting-kernels-only/train.csv", parse_dates=['date'])
test = pd.read_csv("../input/demand-forecasting-kernels-only/test.csv", parse_dates=['date'])
df = pd.concat([train, test], sort=False)

######################################################
# EXPLORATORY DATA ANALYSIS (EDA)
######################################################

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(train)
check_df(test)

# OUTLIER

def outlier_thresholds(dataframe, col_name, q1_perc=0.05, q3_perc=0.95):
    """
    given dataframe, column name, q1_percentage and q3 percentage, function calculates low_limit and up_limit

    """
    quartile1 = dataframe[col_name].quantile(q1_perc)
    quartile3 = dataframe[col_name].quantile(q3_perc)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1_perc=0.01, q3_perc=0.99):
    outlier_list = []
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1_perc=0.01, q3_perc=0.99)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True

    else:
        return False

check_outlier(df, 'sales')
# no outliers

# check the concatenated df
df.head()

print(f"total number of stores: {df['store'].nunique()}\n")
print(f"total number of items: {df['item'].nunique()}\n")
print(f"total number of items per each store: {df.groupby(['store'])['item'].nunique()}")

# before building an ML model, let's have a basic prediction by taking the mean values for each item on each store
df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})

######################################################
# TIME SERIES DECOMPOSITION
######################################################

train_plot = train.set_index('date')
y = train_plot['sales'].resample('MS').mean()

result = sm.tsa.seasonal_decompose(y, model='additive')
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(8, 6)
plt.show()

######################################################
# DATA PREPROCESSING - FEATURE ENGINEERING
######################################################


def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    # 1.1.2013 is Tuesday, so our starting point is the 2nd day of week
    df['day_of_week'] = df.date.dt.dayofweek + 1
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df

######################################################
# RANDOM NOISE
######################################################

# in small datasets like this one, add random noise so that the model learns the patterns instead of memorizing
# eliminate overfitting, random numbers with normal distribution, mean of 0 and standadr deviation of 1 (default)
# with the size of df

df = create_date_features(df)
df.head()

def random_noise(dataframe):

    return np.random.normal(size=(len(dataframe),))

######################################################
# LAG-SHIFTED FEATURES
######################################################

# sort the values per store, item and date so that values would be shifted equally
df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

# the feature name will be created dynamically with regards to the lag value for a given list of lags
# names of the variables will be set dynamically
# shift one day so that the mean values are not correlated with today's value
def lag_features(dataframe, lags):
    dataframe = dataframe.copy()
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])

######################################################
# ROLLING-MEAN FEATURES
######################################################

# a value that occured on time t is highly dependent with the value that occured on time (t-1)

def roll_mean_features(dataframe, windows):
    dataframe = dataframe.copy()
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(dataframe)
    return dataframe

df = roll_mean_features(df, [365, 546, 730])
df.head()

# values for the newly derived lag and rolling mean features will be NaN for most of the train part of the dataframe.
# normal as we are trying to find patterns in order to be able to predict the values in test dataset.

######################################################
# ROLLING-MEAN FEATURES
######################################################

# values shouldn't be equally weighted while taking the mean
# shift one day, take the weighted mean, weights are given as a list. a for loop will be used

def ewm_features(dataframe, alphas, lags):
    dataframe = dataframe.copy()
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales']. \
                    transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]
# start with 91 because the prediction period is 3 months

df = ewm_features(df, alphas, lags)
df.tail()

######################################################
# ONE HOT ENCODING
######################################################

df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])

# take log of sales, the numbers will be lower and the residuals will be lower too, so the no of iterations will
# be less, model's computation time will be less
df['sales'] = np.log1p(df["sales"].values)
df['sales'].head()

######################################################
# CUSTOM COST FUNCTION
######################################################

# custom cost function that uses SMAPE that fits LGBM
def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds-target)
    denom = np.abs(preds)+np.abs(target)
    smape_val = (200*np.sum(num/denom))/n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

######################################################
# TRAIN TEST SPLIT
######################################################

# train-test split won't be made randomly, validation set will be taken inside train
train = df.loc[(df["date"] < "2017-01-01"), :]
train["date"].min(), train["date"].max()

val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

# columns with no useful information or with information that is already derived will be dropped.
cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]

######################################################
# BASE MODEL
######################################################

# no of iteration: 15000, for each 200 iteration, pause, check the cost values, if they have decreased since
# the previous 200, continue, if errors increased, early stop in order not to overfit
# if overfitting, the train dataset's l1 will decrease whereas test dataset's will increase

lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 15000,
              'early_stopping_rounds': 200,
              'nthread': -1}

# model wants like this:
lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

# lgbtrain and lgbval's datatype is LightGBM Dataset
type(lgbtrain)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=200)

# sales values in train dataset will be predicted, these are the log values
y_pred_val = model.predict(X_val)

# log values are reversed and the predicted sales values are revealed.
smape(np.expm1(y_pred_val), np.expm1(Y_val))

######################################################
# FEATURE IMPORTANCE
######################################################

def plot_lgb_importances(model,plot=True,num=10):
    from matplotlib import pyplot as plt
    import seaborn as sns
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    print(feat_imp.head(num))

plot_lgb_importances(model,30) # without plot
plot_lgb_importances(model,True, 30) # with plot

######################################################
# FINAL MODEL
######################################################

# train and validation values are concatenated
train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]

test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)
test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)