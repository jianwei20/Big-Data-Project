import csv
import StringIO
//loading
input = sc.textFile("file://home/bigdatas16/workspace/Recommend/data/train100k")def loadRecord(line):
    """Parse a CSV line"""
    input = StringIO.StringIO(line)
    reader = csv.DictReader(input,fieldname=["name","favouriteAnimal"])
    return reader.next()
    input = sc.textFile(inputFile).map(loadRecord)
//saving

def writeRecords(record):
    """Write out CSV line"""
    output = StringIO.StringIO()
    write = csv.DictWriter(output,fieldname=["name","favouriteAnimal"])
    for record in records:
               writer.writerow(record)
     return [output.getvalue()]
     pandaLovers.mapPartitions(writeRecords).saveAsTextFile(outputFile)
~

# My name is Pong Pong ^_^
from pyspark.sql import SQLContext
from pyspark.sql.types import *
    sqlContext = SQLContext(sc)

 customSchema = StructType([
      StructField("id", DoubleType(), False),
      StructField("click", DoubleType(), False),
      StructField("hour", StringType(), True),
      StructField("C1", DoubleType(), False),
      StructField("banner_pos", DoubleType(), False),
      StructField("site_id", StringType(), True),
      StructField("site_domain", StringType(), True),
      StructField("site_category", StringType(), True),
      StructField("app_id", StringType(), True),
      StructField("app_domain", StringType(), True),
      StructField("app_category", StringType(), True),
      StructField("device_id", StringType(), True),
      StructField("device_ip", StringType(), True),
      StructField("device_model", StringType(), True),
      StructField("device_type", DoubleType(), False),
      StructField("device_conn_type", DoubleType(), False),
      StructField("C14", DoubleType(), False),
      StructField("C15", DoubleType(), False),
      StructField("C16", DoubleType(), False),
      StructField("C17", DoubleType(), False),
      StructField("C18", DoubleType(), False),
      StructField("C19", DoubleType(), False),
      StructField("C20", DoubleType(), False),
      StructField("C21", DoubleType(), False)])
# Get file
    df = sqlContext.read.format("com.databricks.spark.csv").options(header= 'true', inferSchema= 'true').schema(customSchema).load("file:///home/bigdatas16/Downloads/train100K.csv")
# Displays the content of the DataFrame to stdout
    df.show()

from pyspark.ml.feature import StringIndexer
    data = StringIndexer(inputCol="click", outputCol="label").fit(df).transform(df)
    data.show()

# RFormula
from pyspark.ml.feature import RFormula
    formula = RFormula(formula="label ~ C1 + banner_pos + site_category + app_category +device_type + device_conn_type + C15 + C16 + C18 + C19", featuresCol="features", labelCol="label")
    output = formula.fit(data).transform(data)
    data1 = output.select("label", "features")
    data1.show()

# Split training and test data.
    training, test = data1.randomSplit([0.7, 0.3], seed = 12345)
    training.show()
