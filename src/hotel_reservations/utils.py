import json
import logging
from typing import Any, Optional

import requests
import yaml
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit

from hotel_reservations.types.project_config_types import ProjectConfig


def open_config(path: str) -> ProjectConfig:
    """Opens the project config file based on the path given

    Args:
        path (str): Path to the project config yaml file

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

    db_repo_data = requests.get(
        f"{api_url}/api/2.0/repos", headers={"Authorization": f"Bearer {api_token}"}, params={"path_prefix": repo_path}
    ).json()

    db_repo_branch = db_repo_data["repos"][0]["branch"]

    db_repo_head_commit = db_repo_data["repos"][0]["head_commit_id"]

    return db_repo_branch, db_repo_head_commit


def adjust_predictions(predictions: DataFrame, pred_col_name: str = "prediction", scale_factor: float = 1.3):
    """Adjusts the predictions by a scale factor

    Args:
        predictions (DataFrame): PySpark DataFrame containing the prediction
        pred_col_name (str, optional): name of the column containing the predictions. Defaults to "prediction".
        scale_factor (float, optional): Factor to scale by. Defaults to 1.3.

    Returns:
        _type_: DataFrame with adjusted predictions
    """
    return predictions.withColumn(pred_col_name, col(pred_col_name) * lit(scale_factor))


def write_feature_table(
    feature_data: DataFrame, catalog: str, schema: str, use_case_name: str, spark: SparkSession
) -> str:
    """Write feature data to the databricks Feature Store. If the table already exists, the data will be upserted. If not, then a table will be created in the Feature Store.

    Args:
        feature_data (DataFrame): Dataframe containing feature data to write to the Feature Store
        catalog (str): Catalog in which to write the feature data
        schema (str): Schema/database in which to write the feature data
        use_case_name (str): Name of the use case
        spark(SparkSession): The SparkSession used for writing to the FS

    Returns:
        str: Message on succesful writing of data to UC
    """
    feature_table_name = f"{catalog}.{schema}.{use_case_name}_features"
    fe = FeatureEngineeringClient()

    if spark.catalog.tableExists(feature_table_name):
        fe.write_table(
            name=feature_table_name,
            df=feature_data,
            mode="merge",
        )
        return f"The feature data has been succesfully upserted into {feature_table_name}"
    else:
        fe.create_table(
            name=feature_table_name,
            df=feature_data,
            primary_keys="Booking_ID",
            description="Hotel reservation feature data",
        )
        return f"Table {feature_table_name} has been created in the Feature Store successfully."
