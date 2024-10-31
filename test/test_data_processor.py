from unittest.mock import patch

import pytest
from pyspark.sql import SparkSession

from hotel_reservations.data_processing.data_processor import DataProcessor
from hotel_reservations.types.project_config_types import ProjectConfigType
from test.utils import spark_session

spark = spark_session

mock_config: ProjectConfigType = {
    "catalog": "my_catalog",
    "schema": "my_schema",
    "table_name": "my_table",
    "parameters": {"learning_rate": 0.01, "n_estimators": 1000, "max_depth": 6},
    "num_features": {
        "age": {"type": "integer", "constraints": {"min": 0, "max": 100}},
        "income": {"type": "float", "constraints": {"min": 0.0}},
    },
    "cat_features": {
        "gender": {"type": "string", "allowed_values": ["male", "female", "other"]},
        "city": {"type": "string", "allowed_values": ["New York", "Los Angeles", "Chicago"]},
    },
    "target": "purchased",
}


@pytest.fixture
def mock_dataframe(spark: SparkSession):
    data = [
        {"age": 25, "income": 50000, "gender": "M", "city": "NY", "purchased": 1},
        {"age": 30, "income": 60000, "gender": "F", "city": "LA", "purchased": 0},
        {"age": None, "income": 70000, "gender": "F", "city": None, "purchased": 1},
    ]
    return spark.createDataFrame(data)


# Test case for the __init__ and load function
@patch("src.hotel_reservations.data_processing.data_processor.DataProcessor.read_UC_spark")
def test_data_processor_init(mock_read_UC_spark, mock_dataframe, spark: SparkSession):
    mock_read_UC_spark.return_value = mock_dataframe

    processor = DataProcessor(mock_config, spark)

    mock_read_UC_spark.assert_called_once_with(
        mock_config["catalog"], mock_config["schema"], mock_config["table_name"], spark
    )

    assert processor.df == mock_dataframe


# Test the split_data function
@patch("src.hotel_reservations.data_processing.data_processor.DataProcessor.read_UC_spark")
def test_split_data(mock_read_UC_spark, mock_dataframe, spark: SparkSession):
    mock_read_UC_spark.return_value = mock_dataframe

    processor = DataProcessor(mock_config, spark)
    train, test = processor.split_data(test_size=0.5, random_state=42)

    # Assert that split_data returns two DataFrames
    assert (
        type(train).__name__ == "DataFrame"
    )  # Workaround since isInstance(train, DataFrame) didn't work with databricks-connect
    assert (
        type(test).__name__ == "DataFrame"
    )  # Workaround since isInstance(test, DataFrame) didn't work with databricks-connect

    # Test split sizes
    total_rows = mock_dataframe.count()
    assert train.count() + test.count() == total_rows
