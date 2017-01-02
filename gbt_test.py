from pyspark.ml.classification import LogisticRegression
from pyspark.ml.param import Param, Params
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import RFormula
from pyspark.mllib.linalg import Vectors

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
df = sqlContext.read.format("com.databricks.spark.csv").options(header= 'true').schema(customSchema).load("file:///home/bigdatas16/Downloads/train100K.csv")

data = StringIndexer(inputCol="click", outputCol="label").fit(df).transform(df)

formula = RFormula(formula="label ~ C1 + banner_pos + site_category + app_category +device_type + device_conn_type + C15 + C16 + C18 + C19", featuresCol="features", labelCol="label")
output = formula.fit(data).transform(data)
data1 = output.select("label", "features")
(training, test) = data1.randomSplit([0.8, 0.2], seed = 12345)


#gbt = GBTClassifier(numTrees = 10, maxDepth = 3, maxBins = 64)
gbt = GBTClassifier(maxIter = 30, maxDepth = 2, impurityType = gini)

#gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxIter=10)
##rf = RandomForestClassifier(numTrees = 25, maxDepth = 4, maxBins = 64)
pipeline = Pipeline(stages=[gbt])
pipelineModel = pipeline.fit(training)

testPredictions = pipelineModel.transform(test)
testPredictions.select("prediction", "label", "features").show(5)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")#.setMetricName("accuracy")
evaluatorParaMap = {evaluator.metricName: "f1"}
aucTest = evaluator.evaluate(testPredictions, evaluatorParaMap)


from pyspark.ml.tuning import *

paramGrid = ParamGridBuilder().addGrid(gbt.maxIter, [1,5]).build()

cv = CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(3)

cvModel = cv.fit(training)
cvPredictions = cvModel.transform(test)
cvAUCTest = evaluator.evaluate(cvPredictions, evaluatorParaMap)

print("pipeline Test AUC: %g" % aucTest)
print("Cross-Validation test AUC: %g" % cvAUCTest)
end = time.time()
print(end - start)
