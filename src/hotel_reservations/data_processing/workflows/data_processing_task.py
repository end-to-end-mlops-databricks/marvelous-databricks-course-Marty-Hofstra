import logging

import yaml
from pyspark.sql import SparkSession

from hotel_reservations.data_processing.data_processor import DataProcessor


def preprocessing():
    spark = SparkSession.builder.getOrCreate()

    try:
        with open("../../../project_config.yaml", "r") as file:
            config = yaml.safe_load(file)
        logging.info("Configuration file loaded successfully.")
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        return
    except Exception as e:
        logging.error(f"Error loading configuration file: {e}")
        return

    data_preprocessor = DataProcessor(config, spark)

    data_preprocessor.preprocess_data()

    train, test = data_preprocessor.split_data()

    X_features = list(set(config["cat_features"]) | set(config["num_features"]))

    X_train = train.select(X_features)
    X_test = test.select(X_features)
    Y_train = train.select(config["target"])
    Y_test = test.select(config["target"])

    logging.info(f"Training set shape: {X_train.count()}, {len(X_train.columns)}")
    logging.info(f"Test set shape: {X_test.count()}, {len(X_test.columns)}")
    logging.info(f"Training set shape: {Y_train.count()}, {len(Y_train.columns)}")
    logging.info(f"Test set shape: {Y_test.count()}, {len(Y_test.columns)}")


if __name__ == "__main__":
    preprocessing()
