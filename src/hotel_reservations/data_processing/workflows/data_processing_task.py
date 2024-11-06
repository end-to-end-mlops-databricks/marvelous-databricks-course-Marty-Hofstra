from pyspark.sql import SparkSession

from hotel_reservations.data_processing.data_processor import DataProcessor
from hotel_reservations.utils import open_config


def preprocessing() -> list:
    spark = SparkSession.builder.getOrCreate()

    config = open_config("../../../../project_config.yaml")

    data_preprocessor = DataProcessor(config, spark)

    preprocessing_stages = data_preprocessor.preprocess_data()

    train, test = data_preprocessor.split_data()

    train.write.format("delta").mode("overwrite").saveAsTable(
        f"{config.catalog}.{config.db_schema}.{config.use_case_name}_train_data"
    )
    test.write.format("delta").mode("overwrite").saveAsTable(
        f"{config.catalog}.{config.db_schema}.{config.use_case_name}_test_data"
    )

    return preprocessing_stages


if __name__ == "__main__":
    preprocessing_stages = preprocessing()
