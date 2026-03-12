# 01_machine_failure_simulation.py
# Objective: Generate synthetic IoT sensor data for industrial machines (e.g., power supplies).
# We will engineer features to simulate "Time-to-Failure" for PySpark Survival Analysis.

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, randn, when, round, exp

spark = SparkSession.builder.appName("Machine_Failure_Simulation").getOrCreate()
print("Starting Industrial Machine Failure Data Simulation...")

# 1. Generate Base DataFrame (100,000 Machines)
num_machines = 100000
df_base = spark.range(0, num_machines).withColumnRenamed("id", "machine_id")

# 2. Engineer IoT Sensor Features
print("Engineering IoT sensor features (Temperature, Vibration, Voltage Spikes)...")
# Operating Temperature: Normal distribution around 60°F
# Vibration Frequency: Normal distribution around 15 Hz 
# Voltage Spike Count: Poisson distribution with mean 5 spikes/day
# Maintenance Frequency: Uniform distribution between 30 and 90 days
df_features = df_base \
    .withColumn("operating_temp_c", round(60 + (randn() * 10), 1))  \
    .withColumn("vibration_hz", round(15 + (randn() * 5), 2))  \
    .withColumn("voltage_spike_count", round(abs(randn() * 5)))  \
    .withColumn("maintenance_freq_days", round(30 + (rand() * 60)))  

# Clean up any physically impossible negative values
df_features = df_features \
    .withColumn("operating_temp_c", when(col("operating_temp_c") < 20, 20).otherwise(col("operating_temp_c"))) \
    .withColumn("vibration_hz", when(col("vibration_hz") < 1, 1).otherwise(col("vibration_hz"))) 

# 3. Calculate Hidden "Stress Score" and Time-to-Failure
# High temp + high vibration + frequent voltage spikes = High Stress
# High Stress = Machine fails much faster (Accelerated Failure Time)
print("Calculating hidden failure logic and generating target variables...")

df_logic = df_features .withColumn("stress_score", 
                (col("operating_temp_c") * 0.02) +
                (col("vibration_hz") * 0.05) +
                (col("voltage_spike_count") * 0.1) -
                (col("maintenance_freq_days") * 0.005) + 
                (randn() * 0.1)  # Add some random noise 
)

# Generate exact 'days_to_failure' based on the stress score
# We use exponential decay so higher stress drastically shrinks the lifespan
df_logic = df_logic.withColumn("days_to_failure", round(1500 * exp(-col("stress_score"))))

# Define the "Event/Censor" (1.0 = Machine Failed, 0.0 = Still Running/Censored)
# Let's pretend our current observation window is 365 days (1 year).
# If it failed within 365 days, it's a 1.0. If it survived past 365 days, we cap it at 365 and mark it 0.0.
df_final = df_logic \
    .withColumn("censor", when(col("days_to_failure") <= 365, 1.0).otherwise(0.0)) \
    .withColumn("observed_days", when(col("days_to_failure") <= 365, col("days_to_failure")).otherwise(365.0)) \
    .drop("stress_score", "days_to_failure")  # Drop intermediate columns

# 4. Save to Unity Catalog Managed Delta Table
output_table = "workspace.default.xp_power_simulated_failures"
print(f"\nSaving generated IoT data to Unity Catalog: {output_table}...")

df_final.write.format("delta").mode("overwrite").saveAsTable(output_table)

print("✅ Data simulation complete! Ready for PySpark AFT Survival Modeling.")
display(df_final.limit(10))