# 02_aft_survival_modeling.py
# Objective: Train a PySpark Accelerated Failure Time (AFT) Survival Regression model
# to predict the exact number of days until a machine breaks down.

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import AFTSurvivalRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, rand, randn, when, round, exp, abs 

spark = SparkSession.builder.appName("XPPower_AFT_Modeling").getOrCreate()
print("Starting AFT Survival Analysis Modeling...")

# 1. Load Data from Unity Catalog
input_table = "workspace.default.xp_power_simulated_failures"
print(f"Loading data from {input_table}...\n")
df = spark.table(input_table)

# 2. Prepare Features for PySpark ML
# PySpark requires all input features to be combined into a single "vector" column
feature_cols = ["operating_temp_c", "vibration_hz", "voltage_spike_count", "maintenance_freq_days"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(df)

# 3. Train/Test Split (80/20)
print("Splitting data into 80% Training and 20% Testing...\n")
train_data, test_data = df_assembled.randomSplit([0.8, 0.2], seed=42)

# 4. Initialize the AFT Survival Regression Model
# AFT models need to know the 'time' (observed_days) and whether the event actually happened (censor)
aft = AFTSurvivalRegression(
    featuresCol="features",
    labelCol="observed_days",
    censorCol="censor"
)

# 5. Train the Model
print("Training the AFT Survival Model. This might take a few seconds...\n")
model = aft.fit(train_data)

# 6. Extract Coefficients (Business Intelligence)
# In AFT, a negative coefficient means the feature SHRINKS the lifespan (accelerates failure).
# A positive coefficient means the feature EXTENDS the lifespan.
print("--- MODEL COEFFICIENTS (IMPACT ON LIFESPAN) ---")
coefficients = model.coefficients.toArray()
for feature, coef in zip(feature_cols, coefficients):
    impact = "Accelerates Failure" if coef < 0 else "Extends Lifespan"
    print(f"{feature.ljust(25)}: {coef:>8.4f} ({impact})")

# 7. Make Predictions on Unseen Test Data
print("\nPredicting Remaining Useful Life (RUL) on Test Data...")
predictions = model.transform(test_data)

# Clean up the output 
# 'prediction' is the exact number of days the model expects the machine to survive
final_output = predictions.select(
    "machine_id",
    "operating_temp_c",
    "vibration_hz",
    "censor",
    "observed_days",
    round("prediction", 1).alias("predicted_rul")
)

# 8. Save Predictions back to Unity Catalog
output_table = "workspace.default.xp_power_aft_predictions"
print(f"\nSaving RUL predictions to Unity Catalog: {output_table}...")
final_output.write.format("delta").mode("overwrite").saveAsTable(output_table)

print("\n✅ Modeling and Export Complete! Here are the predictions for the test set:")
display(final_output.limit(10))