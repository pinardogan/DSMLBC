The rating and scoring of the comments is crucial for e-commerce websites. 

In this project I used Amazon's sub dataset (containing one product and the comments) in order to test three approaches for sorting and scoring, which are:

1. The difference of positive and negative rates (score_pos_neg_diff)
2. The ratio of positive ratings over the total ratings (score_avg_rating)
3. Wilson Lower Bound (wilson_lower_bound)

I calculated the number of days that passed since the day the comment has been made (today date taken according to the dataset)
Divided the days into quantiles so that I would be able to weighted mean value based on days.
The ratings were in a form : [positive, total] so I took them inside the brackets and derived 3 new variables (yes, no, total ratings)

As a conclusion, Wilson Lower Bound approach was the best that fit all the different scenarios. 
You can find some additional information about this approach in the .py file.
