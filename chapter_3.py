from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .master("local[*]") \
    .appName("Classifier_iris") \
    .getOrCreate()

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

cluster_df = spark.read.csv("clustering_dataset.csv", header=True, inferSchema=True)
print(cluster_df)
cluster_df.show()
cluster_df.show(75)

vectorAssembler = VectorAssembler(inputCols=["col1", "col2", "col3"], outputCol="features")
vcluster_df = vectorAssembler.transform(cluster_df)
vcluster_df.show()

kmeans = KMeans().setK(3)
kmeans = kmeans.setSeed(1)
kmodel = kmeans.fit(vcluster_df)

centers = kmodel.clusterCenters()
print(centers)

# Hierarchical clustering
vcluster_df.show()

from pyspark.ml.clustering import BisectingKMeans
bkmeans = BisectingKMeans().setK(3)
bkmeans = bkmeans.setSeed(1)
bkmodel = bkmeans.fit(vcluster_df)
bkcenters = bkmodel.clusterCenters()
print(bkcenters)
print(centers)
