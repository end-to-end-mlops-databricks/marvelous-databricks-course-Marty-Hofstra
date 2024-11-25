import random

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
from pyspark.sql import SparkSession

from hotel_reservations.featurisation.featurisation import Featurisation
from hotel_reservations.serving.serving import Serving
from hotel_reservations.utils import check_repo_info, open_config


def feature_serving():
    spark = SparkSession.builder.getOrCreate()
    mlflow.set_registry_uri("databricks-uc")

    config = open_config("../../../../project_config.yaml", scope="marty-MLOPs-cohort")

    train_data = spark.read.table(f"{config.catalog}.{config.db_schema}.{config.use_case_name}_train_data")
    test_data = spark.read.table(f"{config.catalog}.{config.db_schema}.{config.use_case_name}_test_data")

    git_branch, git_sha = check_repo_info(
        f"/Workspace/{config.user_dir_path}/{config.git_repo}",
        dbutils,  # type: ignore # noqa: F821
    )

    run_id_basic_model = mlflow.search_runs(
        experiment_names=[f"/{config.user_dir_path}/{config.use_case_name}"],
        filter_string=f"tags.branch='{git_branch}'",
    )["run_id"].iloc[0]

    predict = mlflow.pyfunc.spark_udf(spark, f"runs:/{run_id_basic_model}/gbt-pipeline-model")

    columns_to_serve = config.features_to_serve

    full_df = train_data.unionByName(test_data)

    predictions_df = full_df.withColumn("prediction", predict(*full_df.columns)).select("prediction", *columns_to_serve)

    featurisation_instance = Featurisation(config, predictions_df, "preds", config.primary_key)

    featurisation_instance.write_feature_table(spark)

    feature_spec = OnlineTableSpec(
        primary_key_columns=[config.primary_key],
        source_table_full_name=featurisation_instance.feature_table_name,
        run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
        perform_full_copy=False,
    )

    token = (
        dbutils.notebook.entry_point.getDbutils()  # type: ignore # noqa: F821
        .notebook()
        .getContext()
        .apiToken()
        .get()
    )
    host = spark.conf.get("spark.databricks.workspaceUrl")

    serving_instance = Serving("hotel-reservations-feature-serving", 10, host, token, config.primary_key)

    try:
        online_table_pipeline = serving_instance.workspace.online_tables.create(  # type: ignore # noqa: F841
            name=f"{featurisation_instance.feature_table_name}_online", spec=feature_spec
        )
    except Exception as e:
        print(f"Online feature table '{featurisation_instance.feature_table_name}_online' not created: {e}")

    features = [
        FeatureLookup(
            table_name=featurisation_instance.feature_table_name,
            lookup_key=featurisation_instance.primary_key,
            feature_names=["avg_price_per_room", "no_of_week_nights", "prediction"],
        )
    ]

    feature_spec_name = f"{config.catalog}.{config.db_schema}.return_predictions"
    fe = feature_engineering.FeatureEngineeringClient()

    try:
        fe.create_feature_spec(name=feature_spec_name, features=features, exclude_columns=None)
    except Exception as e:
        print(f"Feature spec'{feature_spec_name}' not created: {e}")

    serving_instance.create_serving_endpoint(feature_spec_name)

    booking_id_list = predictions_df.select(config.primary_key).rdd.flatMap(lambda x: x).collect()

    response_status, response_text, latency = serving_instance.send_request(random.choice(booking_id_list))

    print("Response status:", response_status)
    print("Reponse text:", response_text)
    print("Execution time:", latency, "seconds")

    total_execution_time, average_latency = serving_instance.execute_and_profile_requests(booking_id_list)

    print("\nTotal execution time:", total_execution_time, "seconds")
    print("Average latency per request:", average_latency, "seconds")


if __name__ == "__main__":
    feature_serving()
