##################################################
# Churn Prediction using PySpark
##################################################


import warnings
import findspark
import pandas as pd
import seaborn as sns
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

findspark.init("/Users/pinardogan/spark/spark-3.1.1-bin-hadoop2.7")

spark = SparkSession.builder \
    .master("local") \
    .appName("pyspark_giris") \
    .getOrCreate()

sc = spark.sparkContext

spark_df = spark.read.csv("/Users/pinardogan/Desktop/dsmlbc4/odevler/dataset/churn2.csv", header=True, inferSchema=True)
# inferschema:False: cast every feature as string, if set to True: checks the content and casts the appropriate type

spark_df.show(5)

##################################################
# EDA
##################################################

# Number of observations and variables
print("No of observations: ", spark_df.count(), ", No of variables: ", len(spark_df.columns))

# variables and their types
spark_df.printSchema()

# dtypes method outputs all the types with a list that is made of several tuples,
# each tuple's 1st element is the variable name whereas the 2nd element is the type
spark_df.dtypes

# numeric cols (col[1] for the variable type, col[0] for variable name)
num_cols = [col[0] for col in spark_df.dtypes if col[1] != 'string']
spark_df.select(num_cols).describe().show()


# categorical cols
cat_cols = [col[0] for col in spark_df.dtypes if col[1] == 'string']

for col in cat_cols:
    spark_df.select(col).distinct().show()

# descriptive statistics of the numeric columns with regards of churn
for col in num_cols:
    spark_df.groupby("Exited").agg({col: "mean"}).show()

# descriptive statistics of the categorical columns with regards of churn
for col in cat_cols:
    spark_df.groupby([col, "Exited"]).count().show()

##################################################
 # DATA PREPROCESSING & FEATURE ENGINEERING
##################################################

##################################################
 # Missing Values
##################################################

from pyspark.sql.functions import when, count, col

spark_df.select([count(when(col(c).isNull(), c)).alias(c) for c in spark_df.columns]).toPandas().T
# no null values

spark_df = spark_df.withColumn('BalanceOverSalary', spark_df.Balance / spark_df.EstimatedSalary)
spark_df = spark_df.withColumn('BalanceOverCreditScore', spark_df.Balance / spark_df.CreditScore)
spark_df = spark_df.withColumn('CreditScoreOverSalary', spark_df.CreditScore / spark_df.EstimatedSalary)
spark_df = spark_df.withColumn('AgeOverSalary', spark_df.Age / spark_df.EstimatedSalary)
spark_df = spark_df.withColumn('AgeOverBalance', spark_df.Age / spark_df.Balance)

# if gender's female, then the value is 0, if men then the value is 1
spark_df = spark_df.withColumn("isMale", when(spark_df["gender"] == "Female", 0).otherwise(1))
spark_df = spark_df.drop("gender")
spark_df.show(5)

spark_df.filter(spark_df['EstimatedSalary'] == 0).show()
# each customer has a salary above 0

# rename exited as churn
spark_df = spark_df.withColumnRenamed("Exited", "Churn")

# lower all the variable names
for col in spark_df.columns:
    spark_df = spark_df.withColumnRenamed(col, col.lower())

spark_df.printSchema()

##################################################
# Bucketization / Bining / Num to Cat
##################################################

############################
# Feature Engineering with Bucketizer
############################

from pyspark.ml.feature import Bucketizer

spark_df.select('age').describe().toPandas().transpose()

# with Bucketizer object we name the bins and assign the binned values to the new variable
bucketizer = Bucketizer(splits=[0, 35, 45, 92], inputCol="age", outputCol="agecat")
spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df)
# if some values fall outside the buckets, categorize them into a separate bin

spark_df.show(20)

# add 1 to agecat so it won't start with 0
spark_df = spark_df.withColumn('agecat', spark_df.agecat + 1)
spark_df.groupby("agecat").count().show()

spark_df.groupby("agecat").agg({'churn': "mean"}).show()
# get rid of the decimal part
spark_df = spark_df.withColumn("agecat", spark_df["agecat"].cast("integer"))

############################
# Feature Engineering via when (segment)
############################

# divide tenure variable into two categories
spark_df.select('tenure').describe().toPandas().transpose()

# derive a new feature called segment, if tenure<5 then segment_b, else segment_a
spark_df = spark_df.withColumn('segment', when(spark_df['tenure'] < 5, "segment_b").otherwise("segment_a"))

spark_df.withColumn('agecat2',
                    when(spark_df['age'] < 36, "young").
                    when((35 < spark_df['age']) & (spark_df['age'] < 46), "mature").
                    otherwise("senior")).show()

# drop age
spark_df = spark_df.drop('age')

##################################################
# Label Encoding
##################################################

# put segment into indexer, the most frequent class is labeled as 0
indexer = StringIndexer(inputCol="segment", outputCol="segment_label")
indexer.fit(spark_df).transform(spark_df).show(5)
temp_sdf = indexer.fit(spark_df).transform(spark_df)
# no float values wanted at this stage, return with the same variable name but cast integer
spark_df = temp_sdf.withColumn("segment_label", temp_sdf["segment_label"].cast("integer"))

spark_df = spark_df.drop('segment')
# pyspark doesn't allow string, first label encoding and label "1,2,3,4,etc" then one hot encoding

indexer = StringIndexer(inputCol="geography", outputCol="geography_label")
indexer.fit(spark_df).transform(spark_df).show(5)
temp_sdf = indexer.fit (spark_df).transform (spark_df)
spark_df = temp_sdf.withColumn ("geography_label", temp_sdf["geography_label"].cast ("integer"))

##################################################
# One Hot Encoding
##################################################

encoder = OneHotEncoder(inputCols=["agecat", "geography_label"], outputCols=["age_cat_ohe", "geography_label_ohe"])
spark_df = encoder.fit(spark_df).transform(spark_df)

spark_df.show(20)

##################################################
# Define TARGET
##################################################

# have to name target variable as "target", no string allowed, should label first

# define TARGET
stringIndexer = StringIndexer(inputCol='churn', outputCol='label')
temp_sdf = stringIndexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("label", temp_sdf["label"].cast("integer"))

##################################################
# Define Feature
##################################################
spark_df.show(5)

cols = ['creditscore', 'tenure', 'balance', 'numofproducts', 'hascrcard', 'isactivemember', 'estimatedsalary',
        'balanceoversalary', 'balanceovercreditscore', 'creditscoreoversalary', 'ageoversalary',
        'ismale', 'segment_label', 'age_cat_ohe', 'geography_label_ohe']

# X values should be named as feature, requested by the pyspark

spark_df[cols].printSchema()
# any string values? no

va = VectorAssembler(inputCols=cols, outputCol="features")
va_df = va.transform(spark_df)
va_df.show()

# Final sdf
# all features in a feature vector, target feature in a target vector
final_df = va_df.select("features", "label")

train_df, test_df = final_df.randomSplit([0.7, 0.3])
train_df.show(10)
test_df.show(10)

print("Training Dataset Count: " + str(train_df.count()))
print("Test Dataset Count: " + str(test_df.count()))


##################################################
# MODELING
##################################################

##################################################
# Logistic Regression
##################################################

log_model = LogisticRegression(featuresCol='features', labelCol='label').fit(train_df)
y_pred = log_model.transform(test_df)   # tranform instead of predict
y_pred.show()
# rawpredictions in output are the values not put into sigmoid function, after sigmoid they are probability

y_pred.select("label", "prediction").show()
y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()

evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName='areaUnderROC')

evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

acc = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "accuracy"})
precision = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "precisionByLabel"})
recall = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "recallByLabel"})
f1 = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "f1"})
roc_auc = evaluator.evaluate(y_pred)

print("accuracy: %f, precision: %f, recall: %f, f1: %f, roc_auc: %f" % (acc, precision, recall, f1, roc_auc))
# accuracy: 0.835169, precision: 0.848641, recall: 0.965719, f1: 0.809581, roc_auc: 0.642364


##################################################
# GBM
##################################################

gbm = GBTClassifier(maxIter=100, featuresCol="features", labelCol="label")
gbm_model = gbm.fit(train_df)
y_pred = gbm_model.transform(test_df)

y_pred.show(5)

y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()

##################################################
# Model Tuning
##################################################

evaluator = BinaryClassificationEvaluator()

gbm_params = (ParamGridBuilder()
              .addGrid(gbm.maxDepth, [2, 4, 6, 8])
              .addGrid(gbm.maxBins, [20, 30, 40])
              .addGrid(gbm.maxIter, [10, 20, 30])
              .build())

cv = CrossValidator(estimator=gbm,
                    estimatorParamMaps=gbm_params,
                    evaluator=evaluator,
                    numFolds=5)

cv_model = cv.fit(train_df)

y_pred = cv_model.transform(test_df)
ac = y_pred.select("label", "prediction")
ac.filter(ac.label == ac.prediction).count() / ac.count()
# 0.8565231898565232