import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from mlflow.models import infer_signature
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from hotel_reservations.data_processing.workflows.data_processing_task import preprocessing
from hotel_reservations.utils import check_repo_info, get_error_metrics, open_config, write_feature_table


def fe_model():
    spark = SparkSession.builder.getOrCreate()
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    config = open_config("../../../../project_config.yaml").dict()

    hotel_reservation_data = spark.read.table(f"{config['catalog']}.{config['db_schema']}.{config['table_name']}")

    write_feature_table(
        hotel_reservation_data, config["catalog"], config["db_schema"], config["table_name"], "Booking_ID", spark
    )

    fe = feature_engineering.FeatureEngineeringClient()

    preprocessing_stages = preprocessing()

    train_data = (
        spark.read.table(f"{config['catalog']}.{config['db_schema']}.{config['table_name']}_train_data")
        .drop("avg_price_per_room")
        .withColumn("no_of_week_nights", col("no_of_week_nights").cast("int"))
        .withColumn("no_of_weekend_nights", col("no_of_weekend_nights").cast("int"))
    )

    test_data = spark.read.table(f"{config['catalog']}.{config['db_schema']}.{config['table_name']}_test_data")

    function_name = f"{config['catalog']}.{config['db_schema']}.calculate_no_of_nights"

    spark.sql(f"""
    CREATE OR REPLACE FUNCTION {function_name}(no_of_week_nights INT, no_of_weekend_nights INT)
    RETURNS INT
    LANGUAGE PYTHON AS
    $$
    # Calculate the total number of nights
    return no_of_week_nights + no_of_weekend_nights
    $$
    """)

    training_set = fe.create_training_set(
        df=train_data,
        label=config["target"],
        feature_lookups=[
            FeatureLookup(
                table_name=f"{config['catalog']}.{config['db_schema']}.{config['table_name']}_features",
                feature_names=["avg_price_per_room"],
                lookup_key="Booking_ID",
            ),
            FeatureFunction(
                udf_name=function_name,
                output_name="no_of_nights",
                input_bindings={
                    "no_of_weekend_nights": "no_of_weekend_nights",
                    "no_of_week_nights": "no_of_week_nights",
                },
            ),
        ],
    )

    pipeline = Pipeline(
        stages=preprocessing_stages
        + [GBTRegressor(featuresCol="features", labelCol="label", predictionCol="prediction")]
    )

    git_branch, git_sha = check_repo_info(
        "/Workspace/Users/martijn.hofstra@eneco.com/marvelous-databricks-course-Marty-Hofstra",
        dbutils,  # type: ignore # noqa: F821
    )

    mlflow.set_experiment(experiment_name="/Users/martijn.hofstra@eneco.com/hotel_reservations_fe")

    with mlflow.start_run(
        tags={"git_sha": git_sha, "branch": git_branch},
    ) as run:
        run_id = run.info.run_id

        model = pipeline.fit(training_set.load_df())

        predictions = model.transform(test_data)

        error_metrics = get_error_metrics(predictions)

        print(f"Mean Squared Error: {error_metrics['mse']}")
        print(f"Mean Absolute Error: {error_metrics['mae']}")
        print(f"R2 Score: {error_metrics['r2']}")

        mlflow.log_param("model_type", "GBTRegressor with preprocessing")
        mlflow.log_metric("mse", error_metrics["mse"])
        mlflow.log_metric("mae", error_metrics["mae"])
        mlflow.log_metric("r2_score", error_metrics["r2"])
        signature = infer_signature(model_input=training_set.load_df(), model_output=predictions.select("prediction"))

        fe.log_model(
            model=pipeline,
            flavor=mlflow.spark,
            artifact_path="gbt-pipeline-model-fe",
            training_set=training_set,
            signature=signature,
        )

    mlflow.register_model(
        model_uri=f"runs:/{run_id}/gbt-pipeline-model-fe",
        name=f"{config['catalog']}.{config['db_schema']}.house_prices_model_fe",
    )
