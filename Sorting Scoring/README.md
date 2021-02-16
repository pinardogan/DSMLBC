The rating and scoring of the comments is crucial for e-commerce websites. 

In this project I used Amazon's sub dataset (containing one product and the comments) in order to test three approaches for sorting and scoring, which are:

1. The difference of positive and negative rates (score_pos_neg_diff)
2. The ratio of positive ratings over the total ratings (score_avg_rating)
3. Wilson Lower Bound (wilson_lower_bound)

I calculated the number of days that passed since the day the comment has been made (today date taken according to the dataset)
Divided the days into quantiles so that I would be able to compute the weighted mean value based on days.
The ratings were in a form : [positive, total] so I took them outside the brackets and derived 3 new variables (yes, no, total ratings)

As a conclusion, Wilson Lower Bound approach was the best that fit all the different scenarios. 

**Wilson Lower Bound**: Suppose that there is a world full of potential customers to buy and to rate this product, that is the population. The customers that have already bought and rated the product is the sample. If we compute the confidence interval within a confidence level of 95%, the lower value of this interval is Wilson Lower Bound.
