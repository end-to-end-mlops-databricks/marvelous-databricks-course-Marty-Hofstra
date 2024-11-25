import json
import logging
import os
import random
import re
from typing import Any, Optional

import requests
import yaml
from databricks.sdk import WorkspaceClient
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from hotel_reservations.featurisation.featurisation import Featurisation
from hotel_reservations.monitoring.monitoring import Monitoring
from hotel_reservations.types.project_config_types import ProjectConfig


def open_config(path: str, scope: str) -> ProjectConfig:
    """Opens the project config file based on the path given, the user_dir_path and volume_whl_path are retrieved from Databricks dbutils secrets. Make sure you've read the documentation and followed the steps in the create_databricks_secrets notebook before calling this function.

    Args:
        path (str): Path to the project config yaml file
        scope (str): Name of the Databricks secret scope

    Raises:
        FileNotFoundError: When the given path is not found
        ValueError: When the file is not valid YAML

    Returns:
        ProjectConfig: Project configuration file containing the catalog and schema where the data resides. Moreover, it contains the model parameters, numerical features, categorical features and the target variables.
    """
    try:
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        logging.info("Configuration file loaded successfully from project_config.yaml")
    except FileNotFoundError as e:
        msg = f"Configuration file not found at '{path}'. Ensure it exists: {e}"
        logging.error(msg)
        raise FileNotFoundError(msg) from e
    except yaml.YAMLError as e:
        msg = f"Failed to parse configuration file at '{path}'. Ensure it's valid YAML: {e}"
        logging.error(msg)
        raise ValueError(msg) from e

    try:
        w = WorkspaceClient(profile=os.environ.get("DATABRICKS_PROFILE", None))
        config["user_dir_path"] = w.dbutils.secrets.get(scope=scope, key="user_dir_path")
        config["volume_whl_path"] = w.dbutils.secrets.get(scope=scope, key="volume_whl_path")
        return ProjectConfig(**config)
    except Exception as e:
        msg = f"Failed to retrieve Databricks secrets from scope '{scope}': {e}"
        logging.error(msg)
        raise RuntimeError(msg) from e


def get_error_metrics(
    predictions: DataFrame, label_col_name: str = "label", pred_col_name: str = "prediction"
) -> dict[str, float]:
    """Gets the mse, mae, and r2 error metrics.

    Args:
        predictions (DataFrame): DF containing the label and prediction column.
        label_col_name (str, optional): Name of the column containing the label. Defaults to "label".
        pred_col_name (str, optional): Name of the column containing the predictions. Defaults to "prediction".

    Raises:
        ValueError: If the specified label or prediction column is missing from the DataFrame.

    Returns:
        dict[str, float]: Dictionary containing the mse, mae, and r2.
    """
    # Check for presence of label and prediction columns
    if label_col_name not in predictions.columns:
        raise ValueError(f"Label column '{label_col_name}' is missing from the predictions DataFrame.")
    if pred_col_name not in predictions.columns:
        raise ValueError(f"Prediction column '{pred_col_name}' is missing from the predictions DataFrame.")

    evaluator = RegressionEvaluator(labelCol=label_col_name, predictionCol=pred_col_name)
    mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
    mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
    r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

    error_metrics = {"mse": mse, "mae": mae, "r2": r2}

    return error_metrics


def check_repo_info(repo_path: str, dbutils: Optional[Any] = None) -> tuple[str, str]:
    """Retrieves the current branch and sha in the Databricks Git repos, based on the repo path.

    Args:
        repo_path (str): Full path to the Databricks Git repo
        dbutils (Optional[Any], optional): Databricks utilities, only available in Databricks. Defaults to None.

    Returns:
        git_branch (str)
            Current git branch
        git_sha (str)
            Current git sha
    """
    if dbutils is None:
        raise ValueError("dbutils cannot be None. Please pass the dbutils object when calling check_repo_info.")

    nb_context = json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson())

    api_url = nb_context["extraContext"]["api_url"]

    api_token = nb_context["extraContext"]["api_token"]

    db_repo_data = requests.get(
        f"{api_url}/api/2.0/repos",
        headers={"Authorization": f"Bearer {api_token}"},
        params={"path_prefix": repo_path},
    ).json()

    repo_info = db_repo_data["repos"][0]
    db_repo_branch = repo_info.get("branch", "N/A")
    db_repo_head_commit = repo_info.get("head_commit_id", "N/A")

    return db_repo_branch, db_repo_head_commit


def generate_booking_ids_regex(existing_ids: list[str], num_new_ids=1000, prefix="INN") -> list[str]:
    """
    Generate a specified number of new booking IDs based on the given pattern using regex.

    Args:
        existing_ids (list): List of existing booking IDs (e.g., ['INN01205', 'INN01207']).
        num_new_ids (int): Number of new booking IDs to generate.
        prefix (str): The prefix for the booking IDs (default is 'INN').

    Returns:
        new_ids (list): A list of new booking IDs.
    """
    numeric_ids: list[int] = [
        int(match.group(1)) for id in existing_ids if (match := re.search(rf"{prefix}(\d+)", id)) is not None
    ]

    max_id = max(numeric_ids)
    padding_length = len(str(max_id))

    new_ids = [f"{prefix}{str(i).zfill(padding_length)}" for i in range(max_id + 1, max_id + 1 + num_new_ids)]

    return new_ids


spark = SparkSession.builder.getOrCreate()


def generate_synthetic_data(input_data: DataFrame, num_rows: int = 100, drift: bool = False) -> DataFrame:
    """
    Generates synthetic data using PySpark to simulate data ingestion.

    Args:
        input_data (DataFrame): Current input DataFrame with real data.
        num_rows (int, optional): Number of rows to generate. Defaults to 100.
        drift (bool, optional): Simulates data drift. Defaults to False.

    Returns:
        DataFrame: Synthetic PySpark DataFrame with `num_rows` rows.
    """
    num_rows = min(num_rows, 100000)  # Cap the number of rows

    # Hardcoded schema and features
    schema = StructType(
        [
            StructField("Booking_ID", StringType(), True),
            StructField("no_of_adults", IntegerType(), True),
            StructField("no_of_children", IntegerType(), True),
            StructField("no_of_weekend_nights", IntegerType(), True),
            StructField("no_of_week_nights", IntegerType(), True),
            StructField("type_of_meal_plan", StringType(), True),
            StructField("required_car_parking_space", IntegerType(), True),
            StructField("room_type_reserved", StringType(), True),
            StructField("lead_time", IntegerType(), True),
            StructField("arrival_year", IntegerType(), True),
            StructField("arrival_month", IntegerType(), True),
            StructField("arrival_date", IntegerType(), True),
            StructField("market_segment_type", StringType(), True),
            StructField("repeated_guest", IntegerType(), True),
            StructField("no_of_previous_cancellations", IntegerType(), True),
            StructField("no_of_previous_bookings_not_canceled", IntegerType(), True),
            StructField("avg_price_per_room", DoubleType(), True),
            StructField("no_of_special_requests", IntegerType(), True),
            StructField("booking_status", StringType(), True),
        ]
    )

    # Extract existing booking IDs
    existing_ids = input_data.select("Booking_ID").rdd.flatMap(lambda x: x).collect()
    synthetic_ids = generate_booking_ids_regex(existing_ids, num_new_ids=num_rows)

    # Generate synthetic data
    def generate_row(booking_id):
        return {
            "Booking_ID": booking_id,
            "no_of_adults": random.randint(1, 4),
            "no_of_children": random.randint(0, 3),
            "no_of_weekend_nights": random.randint(0, 3),
            "no_of_week_nights": random.randint(0, 7),
            "type_of_meal_plan": random.choice(["Meal_Plan_1", "Meal_Plan_2", "Meal_Plan_3", "Not Selected"]),
            "required_car_parking_space": random.randint(0, 1),
            "room_type_reserved": random.choice(["Room_Type_1", "Room_Type_2", "Room_Type_3", "Room_Type_4"]),
            "lead_time": random.randint(0, 365),
            "arrival_year": 2024,
            "arrival_month": random.randint(1, 12),
            "arrival_date": random.randint(1, 28),
            "market_segment_type": random.choice(["Online", "Corporate", "Offline"]),
            "repeated_guest": random.randint(0, 1),
            "no_of_previous_cancellations": random.randint(0, 2),
            "no_of_previous_bookings_not_canceled": random.randint(0, 5),
            "avg_price_per_room": round(random.uniform(50.0, 300.0), 2),
            "no_of_special_requests": random.randint(0, 3),
            "booking_status": random.choice(["Canceled", "Not_Canceled"]),
        }

    # Create synthetic rows
    synthetic_rows = [generate_row(booking_id) for booking_id in synthetic_ids]

    # Create synthetic DataFrame
    synthetic_df = input_data.sparkSession.createDataFrame(synthetic_rows, schema)

    if drift:
        synthetic_df = synthetic_df.withColumn("avg_price_per_room", col("avg_price_per_room") * lit(1.5)).withColumn(
            "no_of_week_nights", (col("no_of_week_nights") * lit(2)).cast("int")
        )

    return synthetic_df


def predict_refresh_monitor(
    config: ProjectConfig, data_type: str, predict_function, monitoring_instance: Monitoring
) -> None:
    """Predicts based on either the 'normal' data or drifted data, writes the predictions & features_to_serve to a feature table
       and refreshed the Lakehouse monitor if new predictions are written to the predictions table.
       There were some performance issues with predicting on the skewed df, and thus a cache is done in case of the skewed df.
    Args:
        config (ProjectConfig): Project configuration file containing the catalog and schema where the data resides. Moreover, it contains the model parameters, numerical features, categorical features and the target variables.
        data_type (str): Type of input data, can be either "normal" or "drift"
        predict_function (function): Predict function, as loaded with MLFlow.
        monitoring_instance (Monitoring): Instance of the monitoring class, as initialised in the workflow.
    """
    suffix = "_skewed" if data_type == "drift" else ""
    try:
        train_data = spark.read.table(f"{config.catalog}.{config.db_schema}.{config.use_case_name}_train_data{suffix}")
        test_data = spark.read.table(f"{config.catalog}.{config.db_schema}.{config.use_case_name}_test_data{suffix}")
        full_df = train_data.unionByName(test_data)
    except Exception as e:
        logging.error(f"Failed to read training/test data: {e}")
        raise

    if data_type == "drift":
        full_df.cache()  # This was required due to performance issues with predicting on this df
        full_df.count()  # Materialize the cache

    predictions_df = full_df.withColumn("prediction", predict_function(*full_df.columns)).select(
        "prediction", config.features_to_serve
    )
    featurisation_instance = Featurisation(config, predictions_df, "preds", config.primary_key)
    featurisation_instance.write_feature_table(spark)

    monitoring_instance.refresh_monitor()

    print(
        f"New predictions, based on the {data_type} data have been written to the prediction table, and the Lakehouse monitor has been updated"
    )
