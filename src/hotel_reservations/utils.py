import json
import logging
import os
import re
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
import yaml
from databricks.sdk import WorkspaceClient
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import avg as mean
from pyspark.sql.functions import stddev

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


def generate_synthetic_data(config: ProjectConfig, input_data: DataFrame, num_rows: int = 1000) -> DataFrame:
    """Generates synthetic data in order to simulate data ingestion into the input data.

    Args:
        config (ProjectConfig): _description_
        input_data (DataFrame): _description_
        num_rows (int, optional): _description_. Defaults to 1000.

    Returns:
        DataFrame: _description_
    """
    synthetic_data = {}

    # Loop through numerical features with constraints
    num_features = {key: {"min": feature.constraints.min} for key, feature in config.num_features.items()}

    # Loop through the columns and generate data based on constraints
    for col_name, constraints in num_features.items():
        mean_val, std_val = input_data.select(mean(col_name), stddev(col_name)).first()

        # Generate data and apply constraints
        synthetic_data[col_name] = np.round(np.random.normal(mean_val, std_val, num_rows))

        # Apply min constraints
        synthetic_data[col_name] = np.maximum(synthetic_data[col_name], constraints["min"]).astype(int)

    # Loop through categorical features with allowed values
    cat_features = {
        key: [
            int(value) if isinstance(value, str) and value.isdigit() else value
            for value in feature.encoding or feature.allowed_values or []
        ]
        for key, feature in config.cat_features.items()
    }
    for col_name, allowed_values in cat_features.items():
        synthetic_data[col_name] = np.random.choice(allowed_values, num_rows)

    # Create target variable (booking_status) as a random value
    synthetic_data[config.target] = np.random.choice(["Not_Canceled", "Canceled"], num_rows, p=[0.9, 0.1])

    existing_ids = input_data.select(config.primary_key).rdd.flatMap(lambda x: x).collect()
    synthetic_ids = generate_booking_ids_regex(existing_ids, num_new_ids=num_rows)
    synthetic_data[config.primary_key] = synthetic_ids

    # Convert the synthetic data dictionary to a DataFrame
    synthetic_df = spark.createDataFrame(pd.DataFrame(synthetic_data))

    return synthetic_df
