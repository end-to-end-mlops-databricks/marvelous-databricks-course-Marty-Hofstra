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
    raise RuntimeError(f"Failed to read table {input_table_path}: {str(e)}")

# COMMAND ----------
existing_ids = full_data.select(config.primary_key).rdd.flatMap(lambda x: x).collect()

# COMMAND ----------
synthetic_df = generate_synthetic_data(config, full_data)

# COMMAND ----------
try:
    table_path = f"{config.catalog}.{config.db_schema}.{config.use_case_name}"
    synthetic_df.write.format("delta").mode("append").saveAsTable(table_path)
    print(f"Successfully appended synthetic data to {table_path}")
except Exception as e:
    raise RuntimeError(f"Failed to write synthetic data: {str(e)}")

# COMMAND ----------
synthetic_drift_df = generate_synthetic_data(config, full_data, drift = True)

# COMMAND ----------
try:
    drift_table_path = f"{config.catalog}.{config.db_schema}.{config.use_case_name}_drift"
    synthetic_drift_df.write.format("delta").mode("append").saveAsTable(drift_table_path)
    print(f"Successfully appended synthetic data to {drift_table_path}")
except Exception as e:
    raise RuntimeError(f"Failed to write synthetic data: {str(e)}")
