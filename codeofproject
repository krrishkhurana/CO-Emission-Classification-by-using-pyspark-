from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("CO2_DT_Eval").getOrCreate()

# 1. Load dataset (change path to your CSV)
data = spark.read.csv(
    "C:/Users/krris/OneDrive/Desktop/CO2_emission_data.csv",
    header=True,
    inferSchema=True
)
data.show()
data.printSchema()
# 2. Handle missing values
cleaned= data.na.fill({
    "engine_size": 0.0,
    "co2_emission": 0.0,
    "fuel_type": "Unknown",
    "vehicle_type": "Unknown"
})
cleaned.show()

# 3. Create label: emission_level (Low / Medium / High)
df = cleaned.withColumn(
    "emission_level",
    when(col("co2_emission") < 120, "Low")
    .when(col("co2_emission") < 180, "Medium")
    .otherwise("High")
)

df.show()

# 4. Encode categoricals + label
fuel_indexer = StringIndexer(inputCol="fuel_type", outputCol="fuel_type_index", handleInvalid="keep")
veh_indexer  = StringIndexer(inputCol="vehicle_type", outputCol="vehicle_type_index", handleInvalid="keep")
label_indexer = StringIndexer(inputCol="emission_level", outputCol="label", handleInvalid="keep")

data_inde = fuel_indexer.fit(df).transform(df)
data_inde = veh_indexer.fit(data_inde).transform(data_inde)
data_inde = label_indexer.fit(data_inde).transform(data_inde)

# 5. Assemble features

feature_cols = ["engine_size", "fuel_type_index", "vehicle_type_index"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
final_df = assembler.transform(data_inde).select("features", "label")
final_df.show(truncate=False)

# 6. Train / test split

train, test = final_df.randomSplit([0.7, 0.3], seed=42)

# 7. Decision Tree model
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=6)
model = dt.fit(train)

# 8. Predictions
pred = model.transform(test)

# 9. Accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
accuracy = evaluator.evaluate(pred)
print("Accuracy:", accuracy)

# 10. Feature importance values
importances = model.featureImportances
print("Feature Importances:")
for name, score in zip(feature_cols, importances):
    print(f"{name}: {score}")
