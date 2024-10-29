from unittest.mock import patch

import pytest
from pyspark.sql import SparkSession

from src.data_processor import DataProcessor

mock_config = {
    "catalog": "my_catalog",
    "schema": "my_schema",
    "table_name": "my_table",
    "num_features": ["age", "income"],
    "cat_features": ["gender", "city"],
    "target": "purchased",
}


@pytest.fixture
def mock_dataframe(spark_session: SparkSession):
    data = [
        {"age": 25, "income": 50000, "gender": "M", "city": "NY", "purchased": 1},
        {"age": 30, "income": 60000, "gender": "F", "city": "LA", "purchased": 0},
        {"age": None, "income": 70000, "gender": "F", "city": None, "purchased": 1},
    ]
    return spark_session.createDataFrame(data)


# Test case for the __init__ and load function
@patch("src.data_processor.DataProcessor.read_UC_spark")
def test_data_processor_init(mock_read_UC_spark, mock_dataframe, spark_session: SparkSession):
    mock_read_UC_spark.return_value = mock_dataframe

    processor = DataProcessor(mock_config)

    mock_read_UC_spark.assert_called_once_with(mock_config["catalog"], mock_config["schema"], mock_config["table_name"])

    assert processor.df == mock_dataframe


# Test the split_data function
@patch("src.data_processor.DataProcessor.read_UC_spark")
def test_split_data(mock_read_UC_spark, mock_dataframe, spark_session: SparkSession):
    mock_read_UC_spark.return_value = mock_dataframe

    processor = DataProcessor(mock_config)
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
