CLTV, alias for **Customer Life Time Value** and it's a good metric for CRM. 

It's not possible to deal with every customer, so finding the similar customer patterns bringing the similar customers together and grouping them provides a better way of using the company resources effectively.

* I used Online Retail 2 dataset in this project. This dataset contains the purchase values of a wholesale company's customers in UK between 2010-2011. The purchase process of an item is transformed into an observation, so if a customer purchased various kind of items in one invoice, all different kinds are transformed into different observations. In order to eliminate duplicate value problem, I used groupby function and grouped the df regarding Customer ID.

* **Pandas** and **NumPy** libraries are mostly used in this project, plus **scikit-learn**'s preprocessing module (where I used MinMaxScaler)

* I excluded the cancellation invoices, dropped null values and derived a new variable called TotalPrice, because I will be calculating the sum of the money spent by each customer.

* I calculated the total number of invoices (how many times did the customer purchase), total number of items and total price spent.

* In order to calculate CLTV, I needed the following metrics: Average Order Value, Purchase Frequency, Repeat Rate, Churn Rate and Profit Margin.

* CLTV calculation methods can vary in companies.

* CLTV values couldn't be a metric to find the rank of a customer among all the customers, so I scaled all the CLTV values between 1-100 using MinMaxScaler

* This is an imbalanced dataset, I used describe function to see some specific percentailes of the variable, the first scaled CLTV value is 100 whereas the second is 86.

* 99% of scaled_CLTV values are around 0, this is another example of the dataset being imbalanced. Further actions can be maintained.

* When I sorted the dataset by scaled_CLTV, I realized that the money spent is a powerful metric for CLTV calculation whereas frequency is not so powerful.

