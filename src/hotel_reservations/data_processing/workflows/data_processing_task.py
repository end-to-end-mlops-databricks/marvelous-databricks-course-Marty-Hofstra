"""This script preprocesses input data, the following scenarios are accounted for in this script:
- If the preprocessing has already been done on this workspace and no new input data has been ingested, no preprocessing is done.
- If the preprocessing has already been done on this workspace and new input data has been ingested, the preprocessing is done for the new rows and the train/test data is appended to the existing tables.
- If the preprocessing has not been done yet on this workspace, preprocessing is done on all input data and the train/test data is written to UC.

This ensures that the code also works after deploying to a new workspace, while only running the necessary compute.

If the drift parameter is True, this script will preprocess drifted data (from hotel_reservations_skewed), split the data and write to train/test tables with a suffix '_skewed'.
"""

import argparse

from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException, StreamingQueryException

from hotel_reservations.data_processing.data_processor import DataProcessor
from hotel_reservations.utils import open_config


def preprocessing():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--simulate_drift",
        action="store",
        default="False",
        type=str,
        required=True,
    )

    spark = SparkSession.builder.getOrCreate()

    config = open_config("../../../../project_config.yaml", scope="marty-MLOPs-cohort")

    args = parser.parse_args()
    drift = eval(args.simulate_drift)

    data_preprocessor = DataProcessor(config, spark, drift=drift)

    refreshed = False
    suffix = data_preprocessor.suffix

    if spark.catalog.tableExists(
        f"{config.catalog}.{config.db_schema}.{config.use_case_name}_train_data{suffix}"
    ) and spark.catalog.tableExists(f"{config.catalog}.{config.db_schema}.{config.use_case_name}_test_data{suffix}"):
        train_data_booking_ids = spark.read.table(
            f"{config.catalog}.{config.db_schema}.{config.use_case_name}_train_data{suffix}"
        ).select(config.primary_key)
        test_data_booking_ids = spark.read.table(
            f"{config.catalog}.{config.db_schema}.{config.use_case_name}_test_data{suffix}"
        ).select(config.primary_key)

        input_data_booking_ids = data_preprocessor.df.select(config.primary_key)

        new_booking_ids = input_data_booking_ids.join(
            train_data_booking_ids.unionByName(test_data_booking_ids), config.primary_key, "left_anti"
        )

        if new_booking_ids.isEmpty():
            print(
                f"The input data {config.catalog}.{config.db_schema}.{config.use_case_name} has no new booking IDs and thus no further preprocessing is required"
            )
        else:
            try:
                refreshed = True
                data_preprocessor.df = data_preprocessor.df.join(new_booking_ids, config.primary_key)
                train_new, test_new = data_preprocessor.split_data()
                train_new.write.format("delta").mode("append").saveAsTable(
                    f"{config.catalog}.{config.db_schema}.{config.use_case_name}_train_data{suffix}"
                )
                test_new.write.format("delta").mode("append").saveAsTable(
                    f"{config.catalog}.{config.db_schema}.{config.use_case_name}_test_data{suffix}"
                )
                print("The train and test data has been updated for the new booking IDs")
            except (AnalysisException, StreamingQueryException) as e:
                print(f"Error appending to Delta tables: {str(e)}")
                raise
    else:
        refreshed = True

        train, test = data_preprocessor.split_data()
        try:
            train.write.format("delta").mode("overwrite").saveAsTable(
                f"{config.catalog}.{config.db_schema}.{config.use_case_name}_train_data{suffix}"
            )
            test.write.format("delta").mode("overwrite").saveAsTable(
                f"{config.catalog}.{config.db_schema}.{config.use_case_name}_test_data{suffix}"
            )
            print("The train and test data is created for the first time")
        except (AnalysisException, StreamingQueryException) as e:
            print(f"Error creating Delta tables: {str(e)}")
            raise

    dbutils.jobs.taskValues.set(key="refreshed{suffix}", value=refreshed)  # type: ignore # noqa: F821


if __name__ == "__main__":
    preprocessing()
