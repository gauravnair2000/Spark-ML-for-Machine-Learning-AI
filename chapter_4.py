# Classification Algorithms
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .master("local[*]") \
    .appName("Classifier_iris") \
    .getOrCreate()

from pyspark.sql.functions import *
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
iris_df = spark.read.csv("iris.data", inferSchema=True)
print(iris_df.take(1))
iris_df = iris_df.select(col("_c0").alias("sepal_length"),
                         col("_c1").alias("sepal_width"),
                         col("_c2").alias("petal_length"),
                         col("_c3").alias("petal_width"),
                         col("_c4").alias("species"))
print(iris_df.take(1))

vectorAssembler = VectorAssembler(inputCols=["sepal_length","sepal_width",
                                             "petal_length","petal_width"],
                                  outputCol="features")
viris_df = vectorAssembler.transform(iris_df)
print(viris_df.take(1))

indexer = StringIndexer(inputCol="species", outputCol="label")
iviris_df = indexer.fit(viris_df).transform(viris_df)
iviris_df.show(1)

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
splits = iviris_df.randomSplit([0.6,0.4], 1)
train_df = splits[0]
test_df = splits[1]

print(train_df.count())
print(test_df.count())
print(iviris_df.count())

nb = NaiveBayes(modelType="multinomial", labelCol="label", featuresCol="features")
nbmodel = nb.fit(train_df)
predictions_df = nbmodel.transform(test_df)

print(predictions_df.take(1))

evaluator = MulticlassClassificationEvaluator(labelCol="label",
                                              predictionCol="prediction", metricName="accuracy")
nbaccuracy = evaluator.evaluate(predictions_df)
print(nbaccuracy)

# Multilayer perceptron classification
print(iviris_df)
print(iviris_df.take(1))
print(train_df.count())
print(test_df.count())
print(iviris_df.count())

from pyspark.ml.classification import MultilayerPerceptronClassifier
layers = [4,5,5,3]
mlp = MultilayerPerceptronClassifier(layers=layers, seed=1)
mlp_model = mlp.fit(train_df)
mlp_predicitions = mlp_model.transform(test_df)

mlp_evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
mlp_accuracy = mlp_evaluator.evaluate(mlp_predicitions)
print(mlp_accuracy)

# Decision tree classification
print(iviris_df)
print(iviris_df.take(1))

from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(test_df)

dt_evaluator = MulticlassClassificationEvaluator(labelCol="label",
                                                 predictionCol="prediction", metricName="accuracy")
dt_accuracy = dt_evaluator.evaluate(dt_predictions)
print(dt_accuracy)

