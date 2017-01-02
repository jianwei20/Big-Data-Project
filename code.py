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
import time
start = time.time()
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
df = sqlContext.read.format("com.databricks.spark.csv").options(header= 'true').schema(customSchema).load("file:///home/bigdatas16/Downloads/train100K.csv")
# Displays the content of the DataFrame to stdout
df.show()

from pyspark.ml.feature import StringIndexer
data = StringIndexer(inputCol="click", outputCol="label").fit(df).transform(df)
data.show()

# RFormula
from pyspark.ml.feature import RFormula
formula = RFormula(formula="label ~ banner_pos + app_id + site_category + site_id + site_domain + device_model + C14 + C17 + C18 + C19 + C21 ", featuresCol="features", labelCol="label")
output = formula.fit(data).transform(data)
data1 = output.select("label", "features")
data1.show()

# Split training and test data.
    #(training, test) = data1.randomSplit([0.7, 0.3], seed = 12345)
training, test = data1.randomSplit([0.7, 0.3], seed = 12345)
training.show()
    
# Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and rf (random forest).
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.param import Param, Params
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.util import MLUtils
rf = RandomForestClassifier(numTrees = 100, maxDepth = 10, maxBins = 128)
pipeline = Pipeline(stages=[rf])
pipelineModel = pipeline.fit(training)
#trainingPredictions = pipelineModel.transform(training)
#trainingPredictions.show()
#trainingPredictions.select("prediction", "label", "features").show()
testPredictions = pipelineModel.transform(test)

    #evaluator = MulticlassClassificationEvaluator(
    #labelCol="label", predictionCol="prediction", metricName="precision")
evaluator = BinaryClassificationEvaluator()
from pyspark.mllib.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.param import Param, Params
evaluatorParaMap = {evaluator.metricName: "areaUnderROC"}
#aucTraining = evaluator.evaluate(trainingPredictions, evaluatorParaMap)
aucTest = evaluator.evaluate(testPredictions, evaluatorParaMap)
    
# The multiplies out to (2 x 3 x 3) x 10 = 180 different models being trained.
 # k = 3 and k = 10 are common
from pyspark.ml.tuning import *
paramGrid = ParamGridBuilder().addGrid(rf.impurity, ['entropy', 'gini']).addGrid(rf.numTrees, [10, 30, 50]).build()
 # println(paramGrid(1))
cv = CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(3)
 # Run cross-validation, and choose the best set of parameters.
cvModel = cv.fit(training)
cvPredictions = cvModel.transform(test)
cvAUCTest = evaluator.evaluate(cvPredictions, evaluatorParaMap)

cvPredictions.show()

#	println("pipeline Training AUC: " + aucTraining)
print("pipeline Test AUC: %g" % aucTest)
print("Cross-Validation test AUC: %g" % cvAUCTest)
end = time.time()
print(end - start)
