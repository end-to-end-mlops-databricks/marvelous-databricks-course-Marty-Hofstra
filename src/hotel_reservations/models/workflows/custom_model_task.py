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
    config = open_config("../../../../project_config.yaml").dict()

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    client = MlflowClient()

    train_data = spark.read.table(f"{config['catalog']}.{config['db_schema']}.{config['table_name']}_train_data")

    git_branch, git_sha = check_repo_info(
        "/Workspace/Users/martijn.hofstra@eneco.com/marvelous-databricks-course-Marty-Hofstra",
        dbutils,  # type: ignore # noqa: F821
    )

    run_id_basic_model = mlflow.search_runs(
        experiment_names=["/Users/martijn.hofstra@eneco.com/hotel_reservations"],
        filter_string=f"tags.branch='{git_branch}'",
    )["run_id"].iloc[0]

    loaded_model = mlflow.pyfunc.spark_udf(spark, f"runs:/{run_id_basic_model}/gbt-pipeline-model")

    wrapped_model = HotelReservationsModelWrapper()

    wrapped_model.model = loaded_model

    example_input = train_data.limit(1)

    example_prediction = wrapped_model.predict(model_input=example_input)["Prediction"].select("prediction")

    mlflow.set_experiment(experiment_name="/Users/martijn.hofstra@eneco.com/hotel_reservations_pyfunc")

    with mlflow.start_run(tags={"branch": git_branch, "git_sha": git_sha}) as run:
        run_id = run.info.run_id
        signature = infer_signature(model_input=train_data, model_output=example_prediction)
        dataset = mlflow.data.from_spark(
            train_data,
            table_name=f"{config['catalog']}.{config['db_schema']}.{config['table_name']}_train_data",
            version="0",
        )
        mlflow.log_input(dataset, context="training")
        conda_env = _mlflow_conda_env(  # type: ignore # noqa: F841
            additional_conda_deps=None,
            additional_pip_deps=[
                "Volumes/users/martijn_hofstra/packages/housing_price-0.0.2-py3-none-any.whl",
            ],
            additional_conda_channels=None,
        )
        mlflow.pyfunc.log_model(
            python_model=wrapped_model,
            artifact_path="pyfunc-hotel-reservations-model",
            artifacts={"model_path": wrapped_model.model.metadata.artifact_path},
            code_paths=["Volumes/users/martijn_hofstra/packages/housing_price-0.0.2-py3-none-any.whl"],
            signature=signature,
        )

    loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc-hotel-reservations-model")
    loaded_model.unwrap_python_model()

    model_name = f"{config['catalog']}.{config['db_schema']}.hotel_reservations_model_pyfunc"

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
