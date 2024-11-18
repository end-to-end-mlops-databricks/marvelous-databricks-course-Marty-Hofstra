# Databricks notebook source
# COMMAND ----------
from hotel_reservations.utils import open_config, generate_synthetic_data
from pyspark.sql import SparkSession

# COMMAND ----------
config = open_config("../project_config.yaml", scope="marty-MLOPs-cohort")


# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

full_data = spark.read.table(f"{config.catalog}.{config.db_schema}.{config.use_case_name}")

# COMMAND ----------
existing_ids = full_data.select(config.primary_key).rdd.flatMap(lambda x: x).collect()

# COMMAND ----------
synthetic_df = generate_synthetic_data(config, full_data, config.primary_key)

# COMMAND ----------
synthetic_df.write.format("delta").mode("append").saveAsTable(f"{config.catalog}.{config.db_schema}.{config.use_case_name}")