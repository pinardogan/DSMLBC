import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
from sklearn.preprocessing import MinMaxScaler

##########################################################
# CLTV = (Customer_Value / Churn_Rate) x Profit_margin.
##########################################################

# Customer_Value = Average_Order_Value * Purchase_Frequency
# Average_Order_Value = Total_Revenue / Total_Number_of_Orders
# Purchase_Frequency =  Total_Number_of_Orders / Total_Number_of_Customers
# Churn_Rate = 1 - Repeat_Rate
# Profit_margin

df_ = pd.read_excel("/Users/pinardogan/Desktop/dsmlbc4/odevler/dataset/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.shape
df.describe([0.01, 0.25, 0.5, 0.75, 0.99]).T

# DATA PREPARATION

# are there any invoices containing the letter "C"? (cancellation invoices)
df[df['Invoice'].str.contains("C", na= False)]

# exclude them from df
df = df[~df["Invoice"].str.contains("C", na= False)]

# drop missing values
df.dropna(inplace=True)

# derive TotalPrice variable from Quantity and Price
df['TotalPrice'] = df['Quantity'] * df['Price']
df.head()

# group the variables by Customer ID in order to eliminate duplicates
cltv_df = df.groupby('Customer ID').agg({"Invoice": lambda inv: inv.nunique(),
                                         'Quantity': np.sum,
                                         'TotalPrice': np.sum})

cltv_df.head()

# re-assign variable names
cltv_df.columns = ['total_transaction', 'total_unit', 'total_price']
cltv_df.head()

# Average Order Value
cltv_df['avg_order_value'] = cltv_df['total_price'] / cltv_df['total_transaction']
cltv_df.head()

# Purchase Frequency (total transaction / total number of observations)
cltv_df['purchase_frequency'] = cltv_df['total_transaction'] / cltv_df.shape[0]
cltv_df.head()
cltv_df.shape

# Repeat Rate & Churn Rate
# Repeat Rate : customers that purchase more than 1 / total number of customers
repeat_rate = cltv_df[cltv_df.total_transaction > 1].shape[0] / cltv_df.shape[0]
churn_rate = 1 - repeat_rate

# Profit Margin
# with the assumption that profit rate is 5% for all the products
cltv_df['profit_margin'] = cltv_df['total_price'] * 0.05
cltv_df.head()

# CLTV calculation
cltv_df['CV'] = cltv_df['avg_order_value'] * cltv_df['purchase_frequency'] / churn_rate
cltv_df['CLTV'] = cltv_df['CV'] * cltv_df['profit_margin']
cltv_df.head()

# scale CLTV between 1-100, to be able to interpret the rank of a customer among all the customers
scaler = MinMaxScaler(feature_range=(1, 100))
scaler.fit(cltv_df[["CLTV"]])
cltv_df["SCALED_CLTV"] = scaler.transform(cltv_df[["CLTV"]])
# visualise the top 5 observations in descending order
cltv_df.sort_values(by='SCALED_CLTV', ascending=False).head()

cltv_df['SCALED_CLTV'].describe([0.01, 0.25, 0.5, 0.75, 0.99])

# a more precise visualization for a comparison between top SCALED_CLTV and top total_price observations
cltv_df.head()
cltv_df[['total_transaction', 'total_unit', 'total_price', 'CLTV', 'SCALED_CLTV' ]].sort_values("SCALED_CLTV", ascending=False).head(10)
cltv_df[['total_transaction', 'total_unit', 'total_price', 'CLTV', 'SCALED_CLTV' ]].sort_values("total_price", ascending=False).head(10)

# Customer with the highest CLTV score (100) isn't the one that purchased the most, but the one that paid highest value
# this is a proof that CLTV calculation depends more on total price rather than frequency
# top 10 customers having the highest CLTV and total_price are the same customers.
# no matter how high the frequency is, given a high total_price, scaled CLTV is high

# segmentation of CLTV in 4 categories with qcut
cltv_df['Segment'] = pd.qcut(cltv_df['SCALED_CLTV'], 4, labels=['D', 'C', 'B', 'A'])

cltv_df[["Segment", "total_transaction", "total_unit", "total_price", "CLTV", "SCALED_CLTV"]].sort_values(
    by="SCALED_CLTV",
    ascending=False).head()

cltv_df.groupby("Segment")[["total_transaction", "total_unit", "total_price", "CLTV", "SCALED_CLTV"]].agg(
    {"count", "median", "mean"})

"""
Conclusion:
-----------

* it's a better idea to use median instead of mean, for dataset doesn't have a normal distribution
* due to outliers, scaling process couldn't be maintained in an effective way, mean scaled values per each segment is close
* it's a better idea to concatenate the similar values into categories instead of using qcut and making a quantile cut 


"""

