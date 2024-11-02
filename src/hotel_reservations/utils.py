import logging

import git
import yaml
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame

from hotel_reservations.types.project_config_types import ProjectConfigType


def get_git_sha() -> str:
    """Retrieves the git sha, this is required for tagging the mlflow run

    Returns:
        str: Current git sha
    """
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    return sha


def get_git_branch() -> str:
    repo = git.Repo(search_parent_directories=True)
    branch = repo.active_branch.name

    return branch


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
