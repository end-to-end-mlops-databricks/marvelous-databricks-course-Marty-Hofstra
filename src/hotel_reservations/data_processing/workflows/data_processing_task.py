from pyspark.sql import DataFrame, SparkSession

from hotel_reservations.data_processing.data_processor import DataProcessor
from hotel_reservations.utils import open_config


def preprocessing() -> list | DataFrame | DataFrame:
    spark = SparkSession.builder.getOrCreate()

    config = open_config("../../../../project_config.yaml").dict()

    data_preprocessor = DataProcessor(config, spark)

    preprocessing_stages = data_preprocessor.preprocess_data()

    train, test = data_preprocessor.split_data()

    return preprocessing_stages, train, test


if __name__ == "__main__":
    preprocessing_stages, train, test = preprocessing()
