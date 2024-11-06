import json

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession

from hotel_reservations.models.custom_model import HotelReservationsModelWrapper
from hotel_reservations.utils import check_repo_info, open_config


def custom_model():
    spark = SparkSession.builder.getOrCreate()
    config = open_config("../../../../project_config.yaml")

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    client = MlflowClient()

    train_data = spark.read.table(f"{config.catalog}.{config.db_schema}.{config.use_case_name}_train_data")

    git_branch, git_sha = check_repo_info(
        f"/Workspace/{config.user_dir_path}/{config.git_repo}",
        dbutils,  # type: ignore # noqa: F821
    )

    run_id_basic_model = mlflow.search_runs(
        experiment_names=[f"/{config.user_dir_path}/{config.use_case_name}"],
        filter_string=f"tags.branch='{git_branch}'",
    )["run_id"].iloc[0]

    loaded_model = mlflow.pyfunc.spark_udf(spark, f"runs:/{run_id_basic_model}/gbt-pipeline-model")

    wrapped_model = HotelReservationsModelWrapper(loaded_model)

    example_input = train_data.limit(1)

    example_prediction = wrapped_model.predict(context=None, model_input=example_input).select("prediction")

    mlflow.set_experiment(experiment_name=f"/{config.user_dir_path}/{config.use_case_name}_pyfunc")

    with mlflow.start_run(tags={"branch": git_branch, "git_sha": git_sha}) as run:
        run_id = run.info.run_id
        signature = infer_signature(model_input=train_data, model_output=example_prediction)
        dataset = mlflow.data.from_spark(
            train_data,
            table_name=f"{config.catalog}.{config.db_schema}.{config.use_case_name}_train_data",
            version="0",
        )
        mlflow.log_input(dataset, context="training")
        conda_env = _mlflow_conda_env(  # type: ignore # noqa: F841
            additional_conda_deps=None,
            additional_pip_deps=[
                f"{config.volume_whl_path}/hotel_reservations-0.0.2-py3-none-any.whl",
            ],
            additional_conda_channels=None,
        )
        mlflow.pyfunc.log_model(
            python_model=wrapped_model,
            artifact_path="pyfunc-hotel-reservations-model",
            code_paths=[f"{config.volume_whl_path}/hotel_reservations-0.0.2-py3-none-any.whl"],
            signature=signature,
        )

    loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc-hotel-reservations-model")
    loaded_model.unwrap_python_model()

    model_name = f"{config.catalog}.{config.db_schema}.{config.use_case_name}_model_pyfunc"

    model_version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/pyfunc-hotel-reservations-model", name=model_name, tags={"git_sha": f"{git_sha}"}
    )

    with open("model_version.json", "w") as json_file:
        json.dump(model_version.__dict__, json_file, indent=4)

    model_version_alias = "latest"
    client.set_registered_model_alias(model_name, model_version_alias, "4")

    model_uri = f"models:/{model_name}@{model_version_alias}"
    model = mlflow.pyfunc.load_model(model_uri)  # type: ignore # noqa: F841

    client.get_model_version_by_alias(model_name, model_version_alias)


if __name__ == "__main__":
    custom_model()
