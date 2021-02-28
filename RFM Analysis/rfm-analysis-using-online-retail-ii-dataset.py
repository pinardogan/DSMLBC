import numpy as np
import pandas as pd
import datetime as dt

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

"""

In order to maintain a good CRM (Customer Relationship Management), a company should be aware of each customer’s 
attitude towards the company. The company should know the answers to the following questions:

when did the customer last purchased? (Recency)
how often does the customer purchase? (Frequency)
how much money did the customer spend? (Monetary)
The answers to the above questions would make the customer be categorized and after this process, the company would 
be dealing with tens of segments instead of tens of thousands customers. RFM is an acronym that stands for Recency, 
Frequency and Monetary. In order to assign each customer into the appropriate segment, RFM metrics should be 
calculated and afterwards RFM scores should be computed.

Getting to know the variables:
InvoiceNo : The number of the invoice, unique per each purchase. Refund invoice numbers contain "C"
StockCode : Unique code per each item
Description : Name of the item
Quantity : The number of items within the invoice
InvoiceDate : Date and time of the purchase
UnitPrice : Price of a single item, as of Sterlin
CustomerID : Unique id number per each customer
Country : The country where the customer is living

"""

# READ DATA
df_ = pd.read_excel("/Users/pinardogan/Desktop/dsmlbc4/odevler/dataset/online_retail_II.xlsx",
                    sheet_name="Year 2009-2010")
df = df_.copy()
df.head()

# DATA UNDERSTANDING
df.info()

# how many countries in df:
df['Country'].nunique()

# the names of the countries with the total values:
df['Country'].value_counts()

# the most expensive products:
df.sort_values(by='Price', ascending=False).head()

# number of unique products:
df['Description'].nunique()

# most purchased items:
df.groupby("Description").agg({"Quantity": lambda x: x.sum()}).sort_values("Quantity", ascending=False).head()

# check the number of uniques for StockCode ve Description variables:
print(f"Number of uniques in StockCode: {df['StockCode'].nunique()}")
print(f"Number of uniques in Description:{df['Description'].nunique()}")

# the values were expected to be equal, so there must be more than one unique value in Description variable
# for one unique StockCode.
# let's check each StockCode value with the corresponding Description values, get every StockCode that has
# more than one unique Description in a list form

a =df.groupby('StockCode').agg({'Description': "nunique"})
a.reset_index(inplace=True)
a.head()

b = list(a.loc[a['Description'] > 1, 'StockCode'])

for urun_kodu in b:
    print(f"urun kodu = {urun_kodu} {df.loc[df['StockCode'] == urun_kodu, 'Description'].unique()}")

# as an example derived from the above list, both the Descriptions: 'PINK SPOTTY BOWL' and 'PINK POLKADOT BOWL'
# have the same StockCode: 20677. This means that there are duplicates in Description variable (possibly due to manuel
# entries or merge) so it would be better to use StockCode.
df.loc[df['StockCode'] == 20677, 'Description'].unique()

# DATA PREPARATION

# drop na values
df.dropna(inplace=True)

df.info()

df.describe([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T

# there are negative values on Quantity variable, this is caused by the refund invoices
# (Invoices containing the letter "C"), reassign df without refund invoices
df = df[~df["Invoice"].str.contains("C", na=False)]
df.describe([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T

# negative values are excluded. We are not removing outliers (such as the max value on Quantity and Price variables)
# because we will be scoring the dataset.
df['TotalPrice'] = df['Quantity'] * df['Price']
df.head()

# RFM METRICS
# the last date of purchase:
df['InvoiceDate'].max()

# assign "today's date" as 2 days after the last date of purchase to make sure that none of the Recency values become zero
today_date = df['InvoiceDate'].max() + dt.timedelta(days=2)
today_date

"""

create a new df called rfm in order to calculate Recency, Frequency and Monetary values.
df is grouped by customers and:

the number of days between today_date and the last purchase date of this customer is Recency
the number of unique invoices of this customer is Frequency
the sum of TotalPrice is this customer's Monetary

"""

rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                    'Invoice': lambda inv: inv.nunique(),
                                    'TotalPrice': lambda price: price.sum()})
rfm.head()

# renaming rfm columns:
rfm.columns = ['Recency', 'Frequency', 'Monetary']
rfm.head()

# check if there are any zeros in rfm:
rfm.describe([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T

"""

RFM Scores¶
the min number of Recency metric means that this customer has just purchased, so the highest score (5) 
should be given to the lower number of Recency.
the max number of Frequency and Monetary metrics mean that the customer is purchasing frequently and spending more 
money, so the highest score (5) should be given to the highest Frequency and Monetary values.

"""
rfm["RecencyScore"] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["FrequencyScore"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RFM_SCORE"] = (rfm['RecencyScore'].astype(str) +
                    rfm['FrequencyScore'].astype(str) +
                    rfm['MonetaryScore'].astype(str))

rfm.head()

# display some of the customers with the highest scores:
rfm[rfm['RFM_SCORE'] == "555"].head()

# NAMING RFM SCORES
# the following dict has been made according to the famous RFM graphic
seg_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At_Risk',
    r'[1-2]5': 'Cant_Loose',
    r'3[1-2]': 'About_to_Sleep',
    r'33': 'Need_Attention',
    r'[3-4][4-5]': 'Loyal_Customers',
    r'41': 'Promising',
    r'51': 'New_Customers',
    r'[4-5][2-3]': 'Potential_Loyalists',
    r'5[4-5]': 'Champions'
}

# we will be using Recency and Frequency scores for customer segmentation. We are assuming that a customer who has
# recently purchased and who is often purchasing should have high RFM scores.
rfm['Segment'] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str)
rfm.head()

rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)
rfm.head()

# see the number of customers that fall into each category
rfm['Segment'].value_counts()

# TIME FOR ACTION

"""

Now that we have all the scores for the customers and we have been able to categorize them into 10 groups, 
it's time for action. We will be using metrics for this process, not scores. We will be focusing on the groups that 
need a better customer relationship and try to figure out what we can do in order to make that specific segment 
purchase more frequently and become loyal. Thanks to RFM scores of the segments, we know what exactly 
that segment needs.

"""

rfm[["Segment", "Recency", "Frequency", "Monetary"]].groupby("Segment").agg(["mean", "count"])

# take a closer look at the customers that need attention:
rfm[rfm["Segment"] == "Need_Attention"].head()

"""

Need_Attention segment has 207 customers that last purchased nearly 2 months ago, despite the fact that they don't 
frequently purchase, they spend quite a good amount of money. So we should be focusing on this group. In order to 
make them transform into a customer that purchases frequently, we can offer them some discount with a time limit 
of 30 days. So that they would revisit and purchase.

Can't_Loose segment has purchased for 9 times this year but the last date of this was nearly 4 months, they spend a 
good amount of money and they used to be our loyal customers, we can't loose them. We should put this 77 customers 
into our loyalty program, offer them seasonal discounts, make them feel special while purchasing from our company 
and make them loyal again. We can export the customer id list into an excel file and pass this file to our 
Marketing Department.

"""

marketing_df = pd.DataFrame()
marketing_df["Cant_Loose"] = rfm[rfm["Segment"] == "Cant_Loose"].index

marketing_df.head()

# change the dtype of Customer ID variable in order to get rid of the decimal part:
marketing_df['Cant_Loose'] = marketing_df['Cant_Loose'].astype(int)
marketing_df.head()

marketing_df.info

marketing_df.to_csv("cant_loose.csv")