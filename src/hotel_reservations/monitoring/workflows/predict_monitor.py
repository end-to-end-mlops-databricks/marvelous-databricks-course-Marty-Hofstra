"""This script creates a Lakehouse monitor (if is does not exist yet). A Snapshot monitor is created because batch predictions are done with Feature Serving.
Consequently, predictions are done if the input data is refreshed, a distinction is made between 'normal' data and drifted data.
After new predictions have been written to the predictions table, the monitor is refreshed.
Due to performance issues with predictions on the drifted dataset, a cache was required.
"""

import logging

import mlflow
from mlflow.exceptions import MlflowException
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException

from hotel_reservations.monitoring.monitoring import Monitoring
from hotel_reservations.utils import open_config, predict_refresh_monitor

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

        predictions_table_name = f"{config.catalog}.{config.db_schema}.{config.use_case_name}_preds"

        monitoring_instance = Monitoring(config, table_name=predictions_table_name)

        monitoring_instance.create_lakehouse_monitor()

        if is_refreshed:
            predict_refresh_monitor(
                config, data_type="normal", predict_function=predict, monitoring_instance=monitoring_instance
            )
        else:
            print(
                "No new input data has been ingested and thus no prediction and refreshing of the data monitor are required"
            )

        if is_refreshed_drift:
            predict_refresh_monitor(
                config, data_type="drift", predict_function=predict, monitoring_instance=monitoring_instance
            )
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
