# Databricks notebook source
# COMMAND ----------
from hotel_reservations.utils import open_config, generate_synthetic_data
from pyspark.sql import SparkSession

# COMMAND ----------
config = open_config("../project_config.yaml", scope="marty-MLOPs-cohort")


# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

try:
    input_table_path = f"{config.catalog}.{config.db_schema}.{config.use_case_name}"
    full_data = spark.read.table(input_table_path)
except Exception as e:
    raise RuntimeError(f"Failed to read table {input_table_path}: {str(e)}") from e

# COMMAND ----------
synthetic_df = generate_synthetic_data(full_data, num_rows=10000)

# COMMAND ----------
try:
    table_path = f"{config.catalog}.{config.db_schema}.{config.use_case_name}"
    synthetic_df.write.format("delta").mode("append").saveAsTable(table_path)
    print(f"Successfully appended synthetic data to {table_path}")
except Exception as e:
    raise RuntimeError(f"Failed to write synthetic data to {table_path}: {str(e)}") from e

# COMMAND ----------
try:
    input_drift_table_path = f"{config.catalog}.{config.db_schema}.{config.use_case_name}_skewed"
    full_data_skewed = spark.read.table(input_drift_table_path)
except Exception as e:
    raise RuntimeError(f"Failed to read table {input_drift_table_path}: {str(e)}") from e

# COMMAND ----------
# Generate synthetic data with drift simulation
# This data will be used for monitoring and detecting data drift patterns
# The drift parameter modifies the data distribution to simulate a real-world drift scenario
synthetic_drift_df = generate_synthetic_data(full_data_skewed, drift = True)

# COMMAND ----------
try:
    drift_table_path = f"{config.catalog}.{config.db_schema}.{config.use_case_name}_skewed"
    synthetic_drift_df.write.format("delta").mode("append").saveAsTable(drift_table_path)
    print(f"Successfully appended synthetic data to {drift_table_path}")
except Exception as e:
    raise RuntimeError(f"Failed to write synthetic data to {drift_table_path}: {str(e)}") from e
