#source ~/.bashrc
#http://localhost:4040/executors/   後台
#pyspark --master local[*] --executor-memory 4G --driver-memory 2G --packages com.databricks:spark-csv_2.10:1.4.0
#(加起來不要超過8  --executor-memory 2g   --driver-memory 2g 處理排程)
#啟動pyspark指令：
#pyspark --packages com.databricks:spark-csv_2.10:1.4.0
#pyspark --packages com.databricks:spark-csv_2.10:1.4.0 --master local[*] --executor-memory 4G
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import *
from pyspark.ml.feature import RFormula
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.param import Param, Params
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import Row
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import *


from pyspark.sql import SQLContext
from pyspark.sql.types import *
# Load and parse the data file, converting it to a DataFrame.
#data = sqlContext.read.format("com.databricks.spark.csv").options(header='true', inferschema='true').load("/Users/sheng/Downloads/Click_Through_Rate_Prediction/train100000.csv")
data = sqlContext.read.format("com.databricks.spark.csv").options(header='true', inferschema='true').load("/Users/sheng/Downloads/Click_Through_Rate_Prediction/train10000.csv")
#data = sqlContext.read.format("com.databricks.spark.csv").options(header='true', inferschema='true').load("/Users/sheng/Downloads/Click_Through_Rate_Prediction/train.csv")
# Displays the content of the DataFrame to stdout
#df.show()
#data.show()


# Print the schema in a tree format
data.printSchema()

# Select only the "name" column
Schema = data.select("id","click","hour","C1","banner_pos","site_id","site_category").show()

Schema = StructType([
      StructField("id", DoubleType(), True),
      StructField("click", DoubleType(), True),
      StructField("hour", StringType(), True),
      StructField("C1", DoubleType(), True),
      StructField("banner_pos", DoubleType(), True),
      StructField("site_id", StringType(), True),
      StructField("site_domain", StringType(), True),
      StructField("site_category", StringType(), True),
      StructField("app_id", StringType(), True),
      StructField("app_domain", StringType(), True),
      StructField("app_category", StringType(), True),
      StructField("device_id", StringType(), True),
      StructField("device_ip", StringType(), True),
      StructField("device_model", StringType(), True),
      #StructField("device_type", DoubleType(), True),
      #StructField("device_conn_type", DoubleType(), True),
      #StructField("C14", DoubleType(), True),
      #StructField("C15", DoubleType(), True),
      #StructField("C16", DoubleType(), True),
      #StructField("C17", DoubleType(), True),
      #StructField("C18", DoubleType(), True),
      #StructField("C19", DoubleType(), True),
      #StructField("C20", DoubleType(), True),
      #StructField("C21", DoubleType(), True)
        
      StructField("device_type", StringType(), True),
      StructField("device_conn_type", StringType(), True),
      StructField("C14", DoubleType(), True),
      StructField("C15", DoubleType(), True),
      StructField("C16", DoubleType(), True),
      StructField("C17", DoubleType(), True),
      StructField("C18", DoubleType(), True),
      StructField("C19", DoubleType(), True),
      StructField("C20", DoubleType(), True),
      StructField("C21", DoubleType(), True)
    ])
    

from pyspark.ml.feature import StringIndexer
## Index labels, adding metadata to the label column.
## Fit on whole dataset to include all labels in index.
data = StringIndexer(inputCol="click", outputCol="label").fit(data).transform(data)
data.show()
## 可產生另一個檔案.transform(data)不一定要在（data）檔案裡
#labelIndexer  ===> data


# RFormula
from pyspark.ml.feature import RFormula
## RFormula: string input colums will be one-hot encoded, and numeric columns will be cast to doubles.
##特徵值要被修正formula" "
formula = RFormula(
    formula="label ~ banner_pos + app_id + site_category + site_id + site_domain + device_type + device_conn_type",
    #formula="label ~ banner_pos + app_id + site_category + site_id + site_domain + C14 + C17 + C18 + C19 + C21", #0.707636
    #formula="label ~ banner_pos + site_id + site_domain + C14 + C17 + C21", #0.7
    featuresCol="features",
    labelCol="label")
formula_data = formula.fit(data).transform(data)
formula_data.select("features","label").show()


# Split the data into training and test sets (30% held out for testing)
#已經有了！
# Split training and test data.
(training, test) = formula_data.randomSplit([0.7, 0.3], seed = 12345) #what's seed
training.show()


from pyspark.ml.classification import LogisticRegression
from pyspark.ml.param import Param, Params
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.util import MLUtils
#===========#what does it mean from here
# Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and rf (random forest).
#rf = RandomForestClassifier().setMaxBins(70)
rf = RandomForestClassifier(numTrees=100, maxDepth=20, labelCol="label") #maxDepth=20, maxBins=64, 

pipeline = Pipeline(stages=[rf])
pipelineModel = pipeline.fit(training)
trainingPredictions = pipelineModel.transform(training)
#trainingPredictions.show()
trainingPredictions.select("prediction", "label", "features").show()
testPredictions = pipelineModel.transform(test)


#evaluator = MulticlassClassificationEvaluator(
#labelCol="label", predictionCol="prediction", metricName="precision")
evaluator = BinaryClassificationEvaluator()


from pyspark.mllib.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.param import Param, Params
evaluatorParaMap = {evaluator.metricName: "areaUnderROC"}
aucTraining = evaluator.evaluate(trainingPredictions, evaluatorParaMap)
aucTest = evaluator.evaluate(testPredictions, evaluatorParaMap)
print("pipeline Test AUC: %g" % aucTest)

from pyspark.ml.tuning import *
# The multiplies out to (2 x 3 x 3) x 10 = 180 different models being trained.
# k = 3 and k = 10 are common
#from pyspark.ml.tuning import *
#paramGrid = ParamGridBuilder().addGrid(rf.impurity, ['entropy', 'gini']).addGrid(rf.numTrees, [30, 50, 100]).build() #[10, 50, 100]高 50
paramGrid = ParamGridBuilder().addGrid(rf.maxDepth, [10,20,30]).addGrid(rf.impurity, ['entropy', 'gini']).addGrid(rf.numTrees, [30, 50, 100]).build()
#(rf.maxDepth, [10,20,30])
#println(paramGrid(1))

#=============#以上未做cv 以下做cv
cv = CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(3) #setNumFolds(3)
# Run cross-validation, and choose the best set of parameters.
cvModel = cv.fit(training)
cvPredictions = cvModel.transform(test)
cvAUCTest = evaluator.evaluate(cvPredictions, evaluatorParaMap)
cvPredictions.show()
#println("pipeline Training AUC: " + aucTraining)
print("pipeline Test AUC: %g" % aucTest)
print("Cross-Validation test AUC: %g" % cvAUCTest)


