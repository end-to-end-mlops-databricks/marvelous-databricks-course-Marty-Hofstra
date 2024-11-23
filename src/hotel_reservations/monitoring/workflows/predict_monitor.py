import logging

import mlflow
from mlflow.exceptions import MlflowException
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException

from hotel_reservations.featurisation.featurisation import Featurisation
from hotel_reservations.monitoring.monitoring import Monitoring
from hotel_reservations.utils import open_config

logger = logging.getLogger(__name__)


def predict_monitor():
    spark = SparkSession.builder.getOrCreate()
    mlflow.set_registry_uri("databricks-uc")
    mlflow_client = mlflow.tracking.MlflowClient()

    is_refreshed = dbutils.jobs.taskValues.get(taskKey="preprocessing", key="is_refreshed", debugValue=False)  # type: ignore # noqa: F821
    is_refreshed_drift = dbutils.jobs.taskValues.get(  # type: ignore # noqa: F821
        taskKey="preprocessing_drift", key="is_refreshed_drift", debugValue=False
    )

    config = open_config("../../../../project_config.yaml", scope="marty-MLOPs-cohort")

    try:
        current_model_version = mlflow_client.search_model_versions(
            f"name='{config.catalog}.{config.db_schema}.{config.use_case_name}_model_basic'"
        )[0].version
        print(
            f"The model {config.catalog}.{config.db_schema}.{config.use_case_name}_model_basic exists and the current model version is {current_model_version}"
        )

        predict = mlflow.pyfunc.spark_udf(
            spark,
            f"models:/{config.catalog}.{config.db_schema}.{config.use_case_name}_model_basic/{current_model_version}",
        )
        columns_to_serve = [config.primary_key, "avg_price_per_room", "no_of_week_nights"]

        predictions_table_name = f"{config.catalog}.{config.db_schema}.{config.use_case_name}_preds"

        monitoring_instance = Monitoring(config, table_name=predictions_table_name)

        monitoring_instance.create_lakehouse_monitor()

        if is_refreshed:
            train_data = spark.read.table(f"{config.catalog}.{config.db_schema}.{config.use_case_name}_train_data")
            test_data = spark.read.table(f"{config.catalog}.{config.db_schema}.{config.use_case_name}_test_data")
            full_df = train_data.unionByName(test_data)

            predictions_df = full_df.withColumn("prediction", predict(*full_df.columns)).select(
                "prediction", *columns_to_serve
            )
            featurisation_instance = Featurisation(config, predictions_df, "preds", config.primary_key)
            featurisation_instance.write_feature_table(spark)

            monitoring_instance.refresh_monitor()
        else:
            print(
                "No new input data has been ingested and thus no prediction and refreshing of the data monitor are required"
            )

        if is_refreshed_drift:
            n_partitions = int(spark.conf.get("spark.sql.shuffle.partitions"))

            train_data_skewed = spark.read.table(
                f"{config.catalog}.{config.db_schema}.{config.use_case_name}_train_data_skewed"
            )
            test_data_skewed = spark.read.table(
                f"{config.catalog}.{config.db_schema}.{config.use_case_name}_test_data_skewed"
            )
            full_df_skewed = train_data_skewed.unionByName(test_data_skewed).repartition(n_partitions)

            full_df_skewed.cache()  # This was required due to performance issues with predicting on this df
            full_df_skewed.count()  # Materialize the cache

            predictions_df_skewed = full_df_skewed.withColumn("prediction", predict(*full_df_skewed.columns)).select(
                "prediction", *columns_to_serve
            )
            featurisation_instance_skewed = Featurisation(config, predictions_df_skewed, "preds", config.primary_key)
            featurisation_instance_skewed.write_feature_table(spark)

            monitoring_instance.refresh_monitor()
        else:
            print(
                "No new drift data has been ingested and thus no prediction and refreshing of the data monitor are required"
            )

    except MlflowException as e:
        logger.error(f"MLflow error: {str(e)}")
        raise
    except AnalysisException as e:
        logger.error(f"Spark analysis error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in predict_monitor: {str(e)}")
        raise


if __name__ == "__main__":
    predict_monitor()
