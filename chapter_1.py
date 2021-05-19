from pyspark.sql import SparkSession

spark = SparkSession\
        .builder \
        .master("local[*]") \
        .appName("Classifier_iris") \
        .getOrCreate()

# Organizing data in DataFrames
emp_df = spark.read.csv('./employee.txt', header=True)

print(emp_df.schema)
emp_df.printSchema()
print(emp_df.columns)
print(emp_df.take(5))
print(emp_df.count())

sample_df = emp_df.sample(False, 0.3)
print(sample_df.count())

emp_mgrs_df = emp_df.filter("salary > 100000")
print(emp_mgrs_df.count())
emp_mgrs_df.select("salary").show() # will show only top 20


