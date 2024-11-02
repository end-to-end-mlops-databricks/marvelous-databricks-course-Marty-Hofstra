import json
import logging
from typing import Any, Optional

import requests
import yaml
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame

from hotel_reservations.types.project_config_types import ProjectConfigType


def open_config(path: str) -> ProjectConfigType:
    """Opens the project config file based on the path given

    Args:
        path (str): Path to the project config yaml file

    Raises:
        FileNotFoundError: When the given path is not found
        ValueError: When the file is not valid YAML

    Returns:
        ProjectConfigType: Project configuration file containing the catalog and schema where the data resides. Moreover, it contains the model parameters, numerical features, categorical features and the target variables.
    """
    try:
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        logging.info("Configuration file loaded successfully from project_config.yaml")
    except FileNotFoundError as e:
        msg = f"Configuration file not found. Ensure project_config.yaml exists in the project root: {e}"
        logging.error(msg)
        raise FileNotFoundError(msg) from e
    except Exception as e:
        msg = f"Failed to parse configuration file. Ensure it's valid YAML: {e}"
        logging.error(msg)
        raise ValueError(msg) from e

    return config


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
    nb_context = json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson())  # type: ignore

    api_url = nb_context["extraContext"]["api_url"]

    api_token = nb_context["extraContext"]["api_token"]

    db_repo_data = requests.get(
        f"{api_url}/api/2.0/repos", headers={"Authorization": f"Bearer {api_token}"}, params={"path_prefix": repo_path}
    ).json()

    db_repo_branch = db_repo_data["repos"][0]["branch"]

    db_repo_head_commit = db_repo_data["repos"][0]["head_commit_id"]

    return db_repo_branch, db_repo_head_commit
