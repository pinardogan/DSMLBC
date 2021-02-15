###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

# http://jmcauley.ucsd.edu/data/amazon/links.html

# reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
# asin - ID of the product, e.g. 0000013714
# reviewerName - name of the reviewer
# helpful - helpfulness rating of the review, e.g. 2/3 (positive / total)
# reviewText - text of the review
# overall - rating of the product
# summary - summary of the review
# unixReviewTime - time of the review (unix time)
# reviewTime - time of the review (raw)


import pandas as pd
import datetime as dt
import scipy.stats as st
import math


pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

###################################################
# MISSION 1 : RATING BASED ON WEIGHTED TIME
###################################################

# STEP 1 : READ DATASET
df_sub = pd.read_csv("/Users/pinardogan/Desktop/dsmlbc4/odevler/dataset/df_sub.csv")
df_sub.head()
df_sub.info()

df_sub.dropna(inplace=True)

df_sub['asin'].value_counts()
# due to size issues, dataset has been degraded to the most selling product: B007WTAJTO

# STEP 2 : PRODUCT'S MEAN SCORE
df_sub['overall'].mean()
df_sub['overall'].describe()

# STEP 3 : PREPROCESSING

# day_diff calculation : how many days passed after the comment
# we are not sure about the date pattern in order to calculate the weighted score of the product, so we will use time quarters

df_sub['reviewTime'] = pd.to_datetime(df_sub['reviewTime'], dayfirst=True)
current_date = pd.to_datetime('2014-12-08 0:0:0')
df_sub["day_diff"] = (current_date - df_sub['reviewTime']).dt.days
a = df_sub["day_diff"].quantile(0.25)
b = df_sub["day_diff"].quantile(0.50)
c = df_sub["day_diff"].quantile(0.75)
df_sub.head(15)

# STEP 4 : WEIGHTED SCORE BASED ON TIME
# oldest scores' weight is minimum whereas newest score's weights are max.
weighted_overall = (df_sub.loc[df_sub['day_diff'] >= c, 'overall'].mean() * 22 / 100) + \
                   (df_sub.loc[(df_sub['day_diff'] >= b) & (df_sub['day_diff'] < c), 'overall'].mean() * 24 / 100) + \
                   (df_sub.loc[(df_sub['day_diff'] < b) & (df_sub['day_diff'] >= a), 'overall'].mean() * 26 / 100) + \
                   (df_sub.loc[df_sub['day_diff'] < a, 'overall'].mean() * 28 / 100)

print(f"the mean overall value is {df_sub['overall'].mean() :.2f}\nthe weighted mean is {weighted_overall :.2f}")

#################################################################
# MISSION 2 : FIND THE TOP 20 COMMENT TO BE SHOWN ON PRODUCT PAGE
#################################################################

# STEP 1 : 3 NEW VARIABLE DERIVED FROM HELPFUL VARIABLE [POSITIVE, TOTAL]
# pick the number and get rid of brackets
df_sub['helpful_yes'] = df_sub[['helpful']].applymap(lambda x : x.split(', ')[0].strip('[')).astype(int)
df_sub['total_vote'] = df_sub[['helpful']].applymap(lambda x : x.split(', ')[1].strip(']')).astype(int)
df_sub['helpful_no'] = df_sub['total_vote'] - df_sub['helpful_yes']
df_sub.info()

df_sub['helpful_no'].describe([0.25, 0.5, 0.75, 0.85, 0.99])

# STEP 2 : SCORE_POS_NEG_DIFF CALCULATION
df_sub['score_pos_neg_diff'] = df_sub['helpful_yes'] - df_sub['helpful_no']
df_sub.sort_values(by='score_pos_neg_diff', ascending=False).head(15)

# STEP 3 : SCORE_AVG_DIFF CALCULATION
# add an if for avoiding ZeroDivisionError
def score_avg_rating(pos, neg):
    if pos + neg == 0:
        return 0
    else:
        return (pos / (pos + neg) )

df_sub['score_avg_rating'] = df_sub.apply(lambda x : score_avg_rating(x['helpful_yes'], x['helpful_no']), axis = 1).astype(float)
df_sub.head(20)

# sorting based on positive / total ratio
df_sub.sort_values(by='score_avg_rating', ascending=False).head(15)

# STEP 4 : WILSON LOWER BOUND CALCULATION
"""
wilson lower bound: 
treat the existing comments/ratings as a statistical sample (potential customers in the whole world is the population), 
compute the confidence interval based on 95% confidence,
lower score will be wilson lower score

"""


def wilson_lower_bound(pos, neg, confidence=0.95):
    n = pos + neg
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df_sub['wilson_lower_bound'] = df_sub.apply(lambda x : wilson_lower_bound(x['helpful_yes'], x['helpful_no']), axis = 1).astype(float)

# STEP 5 : TOP 20 COMMENT TO BE DISPLAYED ON THE PRODUCT PAGE

# first 20
df_sub[['reviewerID', 'helpful', 'score_pos_neg_diff', 'score_avg_rating', 'wilson_lower_bound']].sort_values(by='wilson_lower_bound', ascending=False).head(20)

#################################################
# COMMENTS AND ADDITIONAL INFORMATION (EN + TR)
#################################################

"""
Both 3 approaches are successful for catching the best comments that are 3 or 4 digits 

for the comments that were rated 100% helpful:
-Wilson Lower Bound rated them due to the number of helpful rating
-Score_avg_rating put them all to 1st place, so a comment that has been upvoted for 1500 times and another one upvoted for 5 times are the same

Wilson Lower Bound has proven its success.

********************************

3 ve 4 haneli pozitif oylanan yorumların oldugu kullanicilar 3 yontemle de basarili bir sekilde yakalanmistir

tamami faydali olarak puanlanan yorumlarda:
-wilson_lower_bound faydali bulan oy adedine gore dengeli bir siralama yapmistir,
-score_avg_rating tamamini 1. siraya koymustur, yani 1500 ya da 5 kez oylanip tamami faydali oy alan yorumlar esit degerde gorulmustur

siralama wilson_lower_bound yontemine gore cok daha saglikli bir sekilde yapilmistir.

"""

