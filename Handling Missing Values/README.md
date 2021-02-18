A summary of what I have learned about handling the missing values.

**Check the Number of Missing Values** :   
- find out if there are any null values in either df or a specific row  
- find out specific number of columns that are null on a row (e.g. min 2 null values in a row)
- pandas library has been used

**Visualize** :  
- used missingno library and matplotlib library (because I worked on PyCharm)
- plots to check the number of null values and compare them between each column
- find out if the null values appeared by chance or are the columns correlated

**Solutions** :  (with an assumption that they appeared by chance)  
-  dropna with variations by using parameters
-  fillna using a method (e.g. mean for numerical, mode for categorical) using apply func. with lambda
-  SimpleImputer from Scikitlearn library, created a SimpleImputer object with a way of filling, fit and transform the array
