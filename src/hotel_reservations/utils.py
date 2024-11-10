import json
import logging
import os
from typing import Any, Optional

import requests
import yaml
from databricks.sdk import WorkspaceClient
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame

from hotel_reservations.types.project_config_types import ProjectConfig


def get_api_token_from_secrets(scope: str, key: str) -> str:
    """Retrieves Databricks secrets with dbutils, this is used to retrieve the Databricks volume path and the Databricks user directory path. When no profile is assigned to the WorkspaceClient, the DEFAULT profile is used.

    Args:
        scope (str): Name of the Databricks secret scope
        key (str): Name of the secret

    Returns:
        str: The value of the secret
    """
    w = WorkspaceClient(profile=os.environ.get("DATABRICKS_PROFILE", None))
    secret = w.dbutils.secrets.get(scope=scope, key=key)
    return secret


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

    config["user_dir_path"] = get_api_token_from_secrets(scope, "user_dir_path")
    config["volume_whl_path"] = get_api_token_from_secrets(scope, "volume_whl_path")
    return ProjectConfig(**config)


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

    try:
        db_repo_data = requests.get(
            f"{api_url}/api/2.0/repos",
            headers={"Authorization": f"Bearer {api_token}"},
            params={"path_prefix": repo_path},
        ).json()
    except requests.exceptions.RequestException as e:
        msg = f"Failed to fetch repository data: {e}"
        logging.error(msg)
        raise ConnectionError(msg) from e

    if not db_repo_data.get("repos"):
        msg = f"No repository found at path: {repo_path}"
        logging.error(msg)
        raise ValueError(msg)

    repo_info = db_repo_data["repos"][0]
    db_repo_branch = repo_info.get("branch", "N/A")
    db_repo_head_commit = repo_info.get("head_commit_id", "N/A")

    return db_repo_branch, db_repo_head_commit
