I used Online Retail dataset II for my RFM Analysis Project. The dataset can be found [here](https://www.kaggle.com/mathchi/online-retail-ii-data-set-from-ml-repository)

This dataset contains the purchase values of a wholesale company's customers in UK between 2009-2010. I imported three main libraries: Pandas, NumPy and Datetime. 
This dataset is my first big data experience, so I spent some time understanding the dataset.    

I used some EDA techniques using methods such as info, value_counts, nunique. In order to understand the dynamics of the dataset, I used groupby and aggregate the 
grouped dataset per one variable and do some maths around the variables in order to find out the most purchased items.
I found out that number of unique values for StockCode and Description variables are not equal, I created a new df with StockCode and Description values (grouped by StockCode) and wrote a for loop to see the unique StockCode value and the Description values in a list. I found out that Description variable had duplicate values possibly caused by manual entries.     

I dropped all the na values, excluded the invoices that begin with "C" (they were refund values and had negative values for Quantity) and created a new variable for TotalPrice calculation.   

For RFM metrics, I used groupby (Customer ID) and aggregate functions and:     

* I calculated each customer's last purchase date (I added 2 days) comparing the last date of the dataset (that is the Recency value). 
* Found the frequency of the customer's purchase using the number of unique Invoice values.  
* I summed up the TotalPrice values.

I gained the metrics but I need the scores in order to find out the customer's place among the other customers. I used qcut function and created 5 segments with equal size and named the segments.   

For RFM scores, I used only Recency and Frequency values, I added them as strings. I used the segmentation map and replaced the values with the keys of the maps, I used regex because in some segments, the value would be either 1 or 2 etc.  

The customers are now segmented into 10 categories, I used another groupby+aggregate functions in order to find the mean RFM metrics and the number of customers that fall into each group. This new table will help me to gain insight of the segment's attitude towards the company. I picked two categories that needed to take care of, I exported the Customer ID's of a group to be passed to the Marketing Department. Finally I talked about some possible actions for the two groups.   



