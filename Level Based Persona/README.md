**Level Based Persona** Project segments the users into categories based on their customer infos. With this information, it is easy to assign the future customers into an appropriate category.

The dataset consist of the df's:
- users df contains a unique id number per each user, reg. date, device category (either IOS or android), the country of the user as well as the user's gender and age.  
- purchases df contains the same unique id number and the date of the purchase and the price that has been paid.

- two df's have been merged based on the uid row which both df's contain.
- our goal is to predict the income of a customer given his/her information (such as country, age, device etc.)
- aggregated all the columns except for uid and price and summed the price
- after using groupby function, the columns that have been grouped appear as indexes so I used reset_index to transform them into column names
- the dataset contains too much age value so I made some categories, used cut function and gave the labels and bins as arguments. The max value for the label was given dynamically.
- created the fnal column which is level_based_customers with aggregating country, device, gender and age names
- now what I have is a column containing the aforementioned columns and the price value
- segmented the rows into 4 different categories based on the amount of price that has been paid, segment A is the premium/best category wheereas segment D is the lowest one
- grouped each segment and computed the average price for confirmation
- finally entered a new customer and predicted the new customer's segment.


