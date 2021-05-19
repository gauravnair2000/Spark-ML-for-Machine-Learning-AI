# Regression
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .master("local[*]") \
    .appName("Classifier_iris") \
    .getOrCreate()

from pyspark.ml.regression import LinearRegression
pp_df = spark.read.csv("power_plant.csv", header=True, inferSchema=True)
print(pp_df)

from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols=["AT", "V", "AP", "RH"], outputCol="features")
vpp_df = vectorAssembler.transform(pp_df)
print(vpp_df.take(1))

lr = LinearRegression(featuresCol="features", labelCol="PE")
lr_model = lr.fit(vpp_df)

print(lr_model.coefficients)
print(lr_model.intercept)
print(lr_model.summary.rootMeanSquaredError)

lr_model.save("lr1.model")

# Decision tree regression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

pp_df = spark.read.csv("power_plant.csv", header=True, inferSchema=True)
print(pp_df.take(1))
vectorAssembler = VectorAssembler(inputCols=["AT", "V", "AP", "RH"], outputCol="features")
vpp_df = vectorAssembler.transform(pp_df)
print(vpp_df.take(1))

splits = vpp_df.randomSplit([0.7,0.3])
train_df = splits[0]
test_df = splits[1]
print(train_df.count())
print(test_df.count())
print(vpp_df.count())

dt = DecisionTreeRegressor(featuresCol="features", labelCol="PE")
dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(test_df)

dt_evaluator = RegressionEvaluator(labelCol="PE", predictionCol="prediction", metricName="rmse")
rmse = dt_evaluator.evaluate(dt_predictions)
print(rmse)

# Gradient-boosted tree regression
from pyspark.ml.regression import GBTRegressor
gbt = GBTRegressor(featuresCol="features", labelCol="PE")
gbt_model = gbt.fit(train_df)
gbt_predictions = gbt_model.transform(test_df)

gbt_evaluator = RegressionEvaluator(labelCol="PE", predictionCol="prediction", metricName="rmse")
gbt_rmse = gbt_evaluator.evaluate(gbt_predictions)
print(gbt_rmse)
