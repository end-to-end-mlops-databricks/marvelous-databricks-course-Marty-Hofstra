import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from hotel_reservations.featurisation.featurisation import Featurisation
from hotel_reservations.utils import open_config


def featurisation():
    spark = SparkSession.builder.getOrCreate()
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    config = open_config("../../../../project_config.yaml", scope="marty-MLOPs-cohort")

    hotel_reservation_data = spark.read.table(f"{config.catalog}.{config.db_schema}.{config.use_case_name}")

    featurisation_instance = Featurisation(config, hotel_reservation_data, "input_features", config.primary_key)

    featurisation_instance.write_feature_table(spark)

    fe = feature_engineering.FeatureEngineeringClient()

    train_data = (
        spark.read.table(f"{config.catalog}.{config.db_schema}.{config.use_case_name}_train_data")
        .drop("avg_price_per_room")
        .withColumn("arrival_year", col("arrival_year").cast("int"))
    )

    function_name = f"{config.catalog}.{config.db_schema}.calculate_years_since_booking"

    spark.sql(f"""
    CREATE OR REPLACE FUNCTION {function_name}(arrival_year INT)
    RETURNS INT
    LANGUAGE PYTHON AS
    $$
    from datetime import datetime
    return datetime.now().year - arrival_year
    $$
    """)

    training_set = fe.create_training_set(
        df=train_data,
        label=config.target,
        feature_lookups=[
            FeatureLookup(
                table_name=f"{config.catalog}.{config.db_schema}.{config.use_case_name}_features",
                feature_names=["avg_price_per_room"],
                lookup_key=config.primary_key,
            ),
            FeatureFunction(
                udf_name=function_name,
                output_name="years_since_booking",
                input_bindings={"arrival_year": "arrival_year"},
            ),
        ],
    )

    display(training_set)  # type: ignore # noqa: F821


if __name__ == "__main__":
    featurisation()
