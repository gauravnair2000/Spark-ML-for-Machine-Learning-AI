from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .master("local[*]") \
    .appName("Classifier_iris") \
    .getOrCreate()

# Introduction to preprocessing
# Normalize numeric data
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors


features_df = spark.createDataFrame([
    (1, Vectors.dense([10.0, 10000.0, 1.0]),),
    (2, Vectors.dense([20.0, 30000.0, 2.0]),),
    (3, Vectors.dense([30.0, 40000.0, 3.0]),)
], ["id", "features"])

print(features_df.take(1))

feature_scaler = MinMaxScaler(inputCol="features", outputCol="sfeatures")
smodel = feature_scaler.fit(features_df)
sfeatures_df = smodel.transform(features_df)
print(sfeatures_df.take(1))
sfeatures_df.select("features", "sfeatures").show()

# Standardize numeric data
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors

features_df = spark.createDataFrame([
    (1, Vectors.dense([10.0, 10000.0, 1.0]),),
    (2, Vectors.dense([20.0, 30000.0, 2.0]),),
    (3, Vectors.dense([30.0, 40000.0, 3.0]),)
], ["id", "features"])

feature_stand_scaler = StandardScaler(inputCol="features", outputCol="sfeatures",
                                      withStd=True, withMean=True)
stand_smodel = feature_stand_scaler.fit(features_df)
stand_sfeatures_df = stand_smodel.transform(features_df)

print(stand_sfeatures_df.take(1))
stand_sfeatures_df.show()

# Bucketize numeric data
from pyspark.ml.feature import Bucketizer

splits = [-float("inf"), -10.0, 0.0, 10, float("inf")]
b_data = [(-800.0,), (-10.5,), (-1.7,), (0.0,), (8.2,), (90.1,)]
b_df = spark.createDataFrame(b_data, ["features"])
b_df.show()

bucketizer = Bucketizer(splits=splits, inputCol="features", outputCol="bfeatures")
bucketed_df = bucketizer.transform(b_df)
bucketed_df.show()

# Tokenize text data
from pyspark.ml.feature import Tokenizer

sentences_df = spark.createDataFrame([
    (1, "This is an introduction to Spark MLlib"),
    (2, "MLlib includes libraries for classification and regression"),
    (3, "It also contains supporting tools for pipelines")
], ["id", "sentence"])
sentences_df.show()

sent_token = Tokenizer(inputCol="sentence", outputCol="words")
sent_tokenized_df = sent_token.transform(sentences_df)
sent_tokenized_df.show()

# TF-IDF (Term frequency - Inverse document frequency)
from pyspark.ml.feature import HashingTF, IDF
print(sentences_df)
print(sentences_df.take(1))
print(sent_tokenized_df.take(1))

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
sent_hfTF_df = hashingTF.transform(sent_tokenized_df)
print(sent_hfTF_df.take(1))

idf = IDF(inputCol="rawFeatures", outputCol="idf_features")
idfModel = idf.fit(sent_hfTF_df)
tfidf_df = idfModel.transform(sent_hfTF_df)

print(tfidf_df.take(1))

