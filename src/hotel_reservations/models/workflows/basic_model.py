import mlflow
from mlflow.models import infer_signature
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor

from hotel_reservations.data_processing.workflows.data_processing_task import preprocessing
from hotel_reservations.utils import check_repo_info, get_error_metrics, open_config


def basic_model():
    config = open_config("../../../../project_config.yaml").dict()

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    mlflow.set_experiment(experiment_name="/Users/martijn.hofstra@eneco.com/hotel_reservations")

    preprocessing_stages, train, test = preprocessing()

    git_branch, git_sha = check_repo_info(
        "/Workspace/Users/martijn.hofstra@eneco.com/marvelous-databricks-course-Marty-Hofstra",
        dbutils,  # type: ignore # noqa: F821
    )

    pipeline = Pipeline(
        stages=preprocessing_stages
        + [GBTRegressor(featuresCol="features", labelCol="label", predictionCol="prediction")]
    )

    with mlflow.start_run(
        tags={"git_sha": git_sha, "branch": git_branch},
    ) as run:
        run_id = run.info.run_id

        model = pipeline.fit(train)

        predictions = model.transform(test)

        error_metrics = get_error_metrics(predictions)

        print(f"Mean Squared Error: {error_metrics['mse']}")
        print(f"Mean Absolute Error: {error_metrics['mae']}")
        print(f"R2 Score: {error_metrics['r2']}")

        mlflow.log_param("model_type", "GBTRegressor with preprocessing")
        mlflow.log_metric("mse", error_metrics["mse"])
        mlflow.log_metric("mae", error_metrics["mae"])
        mlflow.log_metric("r2_score", error_metrics["r2"])
        signature = infer_signature(model_input=train, model_output=predictions.select("prediction"))

        mlflow.spark.log_model(spark_model=model, artifact_path="gbt-pipeline-model", signature=signature)

    model_version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/gbt-pipeline-model",
        name=f"{config['catalog']}.{config['schema']}.hotel_reservations_model_basic",
        tags={"git_sha": git_sha},
    )

    print(model_version)


if __name__ == "__main__":
    basic_model()