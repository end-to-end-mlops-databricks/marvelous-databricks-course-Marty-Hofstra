# Databricks notebook source
# COMMAND ----------
from hotel_reservations.utils import open_config, generate_synthetic_data
from pyspark.sql import SparkSession

# COMMAND ----------
config = open_config("../project_config.yaml", scope="marty-MLOPs-cohort")


# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

try:
    table_path = f"{config.catalog}.{config.db_schema}.{config.use_case_name}"
    full_data = spark.read.table(table_path)
except Exception as e:
    raise RuntimeError(f"Failed to read table {table_path}: {str(e)}")

# COMMAND ----------
existing_ids = full_data.select(config.primary_key).rdd.flatMap(lambda x: x).collect()

# COMMAND ----------
synthetic_df = generate_synthetic_data(config, full_data)

# COMMAND ----------
-synthetic_df.write.format("delta").mode("append").saveAsTable(f"{config.catalog}.{config.db_schema}.{config.use_case_name}")

try:
    table_path = f"{config.catalog}.{config.db_schema}.{config.use_case_name}"
    synthetic_df.write.format("delta").mode("append").saveAsTable(table_path)
    print(f"Successfully appended synthetic data to {table_path}")
except Exception as e:
    raise RuntimeError(f"Failed to write synthetic data: {str(e)}")
