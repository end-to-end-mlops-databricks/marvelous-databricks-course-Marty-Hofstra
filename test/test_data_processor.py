from unittest.mock import patch

import pytest
from pyspark.sql import SparkSession
from pyspark.testing import assertDataFrameEqual

from hotel_reservations.data_processing.data_processor import DataProcessor
from hotel_reservations.types.project_config_types import CatFeature, Constraints, NumFeature
from test.utils import spark_session

spark = spark_session

mock_config: dict = {
    "catalog": "my_catalog",
    "schema": "my_schema",
    "table_name": "hotel_reservations",
    "parameters": {"learning_rate": 0.01, "n_estimators": 1000, "max_depth": 6},
    "num_features": {
        "no_of_adults": NumFeature(type="integer", constraints=Constraints(min=0)),
        "avg_price_per_room": NumFeature(type="float", constraints=Constraints(min=0.0)),
        # Add other numerical features similarly
    },
    "cat_features": {
        "type_of_meal_plan": CatFeature(
            type="string", allowed_values=["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"]
        ),
        "required_car_parking_space": CatFeature(type="bool", allowed_values=[True, False], encoding=[1, 0]),
        # Add other categorical features similarly
    },
    "target": "booking_status",
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
@patch.object(DataProcessor, "read_UC_spark")
def test_data_processor_init(mock_read_UC_spark, mock_dataframe, spark: SparkSession):
    mock_read_UC_spark.return_value = mock_dataframe

    processor = DataProcessor(mock_config, spark)

    mock_read_UC_spark.assert_called_once_with(
        mock_config["catalog"], mock_config["schema"], mock_config["table_name"], spark
    )

    assert processor.df == mock_dataframe


# Test the split_data function
@patch.object(DataProcessor, "read_UC_spark")
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


@patch.object(DataProcessor, "read_UC_spark")
def test_split_data_value_error_test_size_low(mock_read_UC_spark, mock_dataframe, spark: SparkSession):
    mock_read_UC_spark.return_value = mock_dataframe

    processor = DataProcessor(mock_config, spark)

    with pytest.raises(ValueError, match="test_size must be between 0 and 1, got"):
        processor.split_data(test_size=0)


@patch.object(DataProcessor, "read_UC_spark")
def test_split_data_value_error_test_size_high(mock_read_UC_spark, mock_dataframe, spark: SparkSession):
    mock_read_UC_spark.return_value = mock_dataframe

    processor = DataProcessor(mock_config, spark)

    with pytest.raises(ValueError, match="test_size must be between 0 and 1, got"):
        processor.split_data(test_size=1.5)


@patch.object(DataProcessor, "read_UC_spark")
def test_data_after_dropping(mock_read_UC_spark, spark_session: SparkSession):
    data_missing_target = [
        {"age": 25, "income": 50000, "gender": "M", "city": "NY", "booking_status": None},
        {"age": 30, "income": 60000, "gender": "F", "city": "LA", "booking_status": None},
    ]
    data_non_missing_target = [
        {"age": None, "income": 70000, "gender": "F", "city": None, "booking_status": 1},
        {"age": 25, "income": 50000, "gender": "M", "city": "NY", "booking_status": 2},
    ]

    mock_data = data_missing_target + data_non_missing_target
    sparse_df = spark_session.createDataFrame(mock_data)
    mock_read_UC_spark.return_value = sparse_df

    processor = DataProcessor(mock_config, spark)
    processor.preprocess_data()

    expected = spark_session.createDataFrame(data_non_missing_target)

    assert processor.df.count() == 2
    assertDataFrameEqual(processor.df, expected)
