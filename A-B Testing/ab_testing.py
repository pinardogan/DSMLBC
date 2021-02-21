import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
from scipy.stats import shapiro, levene, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns


"""
Dataset story:
a website randomly divides its user traffic into two groups:
Facebook ad's maximum bidding campaign is served to control group
Facebook ad's average bidding campaign (new campaign) is served to test group
two groups are being observed for 40 days in order to find out if Facebook's new campaign "average bidding" is worth launching

variables:
impression  :user sees ad
click       :user clicks the url of the ad
purchase    :user purchases product
earning     :money earned from the purchase

index       :observation day
"""


control = pd.read_excel("/Users/pinardogan/Desktop/dsmlbc4/odevler/dataset/ab_testing_data.xlsx",
                        sheet_name="Control Group")
test = pd.read_excel("/Users/pinardogan/Desktop/dsmlbc4/odevler/dataset/ab_testing_data.xlsx",
                     sheet_name="Test Group")

#check for na values and outliers:
control.head()
control.isna().sum()
control.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T

test.head()
test.isna().sum()
test.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T

print(f"control group's mean purchase value is {control['Purchase'].mean() :.2f} \
\ntest group's mean purchase value is {test['Purchase'].mean() :.2f}")
# values are close, we need to check if there is a statistically significant difference

# FEATURE ENGINEERING

# conversion rate:
control['Conversion_Rate'] = control['Purchase'] / control['Click'] * 100
test['Conversion_Rate'] = test['Purchase'] / test['Click'] * 100

# earning per purchase:
control['Earning_Per_Purchase'] = control['Earning'] / control['Purchase'] * 100
test['Earning_Per_Purchase'] = test['Earning'] / test['Purchase'] * 100

# click through rate (CTR):
control['Click_Through_Rate'] = control['Click'] / control['Impression'] * 100
test['Click_Through_Rate'] = test['Click'] / test['Impression'] * 100

# add new variables to each df with the group id's and concatenate the dfs into an AB df
control['Group'] = "C"
test['Group'] = "T"
AB = pd.concat([control, test], ignore_index=True)

sns.pairplot(AB, hue='Group')
plt.show()
# according to the graph, feature are mostly overlaping, this is a sign of having similar patterns

# NORMALITY TEST : Shapiro-Wilks Test

# H0: there is no statistically significant difference between the two samples
# H1: there is statistically significant difference between the two samples
shapiro(control['Purchase'])
# p value > 0.05 H0 (null) hypothesis cannot be rejected for the control group, normality assumption is valid


shapiro(test['Purchase'])
# p value > 0.05 H0 (null) hypothesis cannot be rejected for the test group either, normality assumption is valid

# HOMOGENEITY OF VARIANCE : Levene Test

#H0: variances of the populations are equal
#H1: variances of the populations are equal
levene(control['Purchase'], test['Purchase'])
#p value > 0.05 H0 hypothesis cannot be rejected, homogeneity of variance is valid

# the two assumptions are valid, so we have to take Independent samples t test
# INDEPENDENT SAMPLES T TEST
#HO: there is no statistically significant difference between the means of the two groups
#H1: there is statistically significant difference between the means of the two groups
ttest_ind(control['Purchase'], test['Purchase'], equal_var=True)
#p value > 0.05 H0 hypothesis cannot be rejected, there is no statistically significant difference

"""
Further Questions:
1.How would you define the hypothesis for AB testing?

H0: There is no statically significant difference between the results gained from A and B groups
H1: There is statically significant difference between the results gained from A and B groups

2.Can we make a statistically significant conclusion?

Independent Samples T Test have been executed because the two tests are valid. Based on the third test (T Test) there is 
no statistically significant difference between the two groups

3.Which tests did you use and why?
As a result of normality and variance homogeneity tests to be valid, Independent Samples T Test have been used.

4.What would you advise to the customer according to the second answer?
The distributions are alike, I would suggest the customer to continue observation in order to gain more insight.

"""

