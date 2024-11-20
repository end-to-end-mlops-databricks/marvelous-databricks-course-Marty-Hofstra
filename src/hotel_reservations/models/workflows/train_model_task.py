"""This script trains a GBTClassifier model, the workflow is configured such that the training task only runs if the input data has been refreshed.
In this task only the experiment run is registered in MLFlow, and the model_uri (containing the experiment run_id) is passed on to the next task for evaluation.
However, if the training task has not been run yet on this workspace (and thus the model does not exists yet in UC), the model will be registered here.
"""

import argparse

import mlflow
from mlflow.models import infer_signature
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.sql import SparkSession

from hotel_reservations.models.model import Model
from hotel_reservations.utils import check_repo_info, get_error_metrics, open_config


def train_model():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--job_run_id",
        action="store",
        default=None,
        type=str,
        required=True,
    )

    spark = SparkSession.builder.getOrCreate()

    config = open_config("../../../../project_config.yaml", scope="marty-MLOPs-cohort")

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    mlflow.set_experiment(experiment_name=f"/{config.user_dir_path}/{config.use_case_name}")

    args = parser.parse_args()
    job_run_id = args.job_run_id

    model_instance = Model(config, spark)

    preprocessing_stages = model_instance.create_preprocessing_stages()

    git_branch, git_sha = check_repo_info(
        f"/Workspace/{config.user_dir_path}/{config.git_repo}",
        dbutils,  # type: ignore # noqa: F821
    )

    train_data = spark.read.table(f"{config.catalog}.{config.db_schema}.{config.use_case_name}_train_data")
    test_data = spark.read.table(f"{config.catalog}.{config.db_schema}.{config.use_case_name}_test_data")

    pipeline = Pipeline(
        stages=preprocessing_stages
        + [
            GBTClassifier(
                featuresCol="features",
                labelCol="label",
                predictionCol="prediction",
                maxDepth=config.parameters.max_depth,
            )
        ]
    )

    with mlflow.start_run(
        tags={"git_sha": git_sha, "branch": git_branch},
    ) as run:
        run_id = run.info.run_id

        model = pipeline.fit(train_data)

        predictions = model.transform(test_data)

        error_metrics = get_error_metrics(predictions)

        print(f"Mean Squared Error: {error_metrics['mse']}")
        print(f"Mean Absolute Error: {error_metrics['mae']}")
        print(f"R2 Score: {error_metrics['r2']}")

        mlflow.log_param("model_type", "GBTClassifier with preprocessing")
        mlflow.log_metric("mse", error_metrics["mse"])
        mlflow.log_metric("mae", error_metrics["mae"])
        mlflow.log_metric("r2_score", error_metrics["r2"])
        signature = infer_signature(model_input=train_data, model_output=predictions.select("prediction"))

        mlflow.spark.log_model(spark_model=model, artifact_path="gbt-pipeline-model", signature=signature)

    model_uri = f"runs:/{run_id}/gbt-pipeline-model"

    try:
        mlflow_client = mlflow.tracking.MlflowClient()
        mlflow_client.get_registered_model("users.martijn_hofstra.hotel_reservations_model_basic")
        print(
            "Model already exists, this run will be evaluated in the next task and it is registered as a new model version in case it performs better than the current version"
        )
    except mlflow.exceptions.RestException as e:
        print(
            f"This is the first time the training task is run on this workspace and the model will be registered: {str(e)}"
        )
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=f"{config.catalog}.{config.db_schema}.{config.use_case_name}_model_basic",
            tags={"git_sha": git_sha, "branch": git_branch, "job_run_id": job_run_id},
        )
        print("New model registered with version:", model_version.version)

    dbutils.jobs.taskValues.set(key="git_sha", value=git_sha)  # type: ignore # noqa: F821
    dbutils.jobs.taskValues.set(key="job_run_id", value=job_run_id)  # type: ignore # noqa: F821
    dbutils.jobs.taskValues.set(key="model_uri", value=model_uri)  # type: ignore # noqa: F821


if __name__ == "__main__":
    train_model()
