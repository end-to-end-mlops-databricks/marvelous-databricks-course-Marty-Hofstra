from unittest.mock import patch

import pytest
from pyspark.sql import SparkSession

from hotel_reservations.data_processing.data_processor import DataProcessor
from hotel_reservations.types.project_config_types import CatFeature, Constraints, NumFeature, ProjectConfig
from test.utils import spark_session

spark = spark_session

# Define the mock config object
mock_config = ProjectConfig(
    catalog="my_catalog",
    schema="my_schema",
    use_case_name="hotel_reservations",
    user_dir_path="/Users/user/",
    git_repo="git_repo",
    volume_whl_path="Volumes/users/user/packages/",
    parameters={"learning_rate": 0.01, "n_estimators": 1000, "max_depth": 6},
    num_features={
        "no_of_adults": NumFeature(type="integer", constraints=Constraints(min=0)),
        "avg_price_per_room": NumFeature(type="float", constraints=Constraints(min=0.0)),
    },
    cat_features={
        "type_of_meal_plan": CatFeature(
            type="string", allowed_values=["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"]
        ),
        "required_car_parking_space": CatFeature(type="bool", allowed_values=[True, False], encoding=[1, 0]),
    },
    target="booking_status",
    primary_key="Booking_ID",
)


# Mock DataFrame fixture
@pytest.fixture
def mock_dataframe(spark: SparkSession):
    data = [
        {
            "no_of_adults": 25,
            "avg_price_per_room": 50000,
            "type_of_meal_plan": "Meal Plan 1",
            "required_car_parking_space": True,
            "booking_status": 1,
        },
        {
            "no_of_adults": 30,
            "avg_price_per_room": 60000,
            "type_of_meal_plan": "Meal Plan 2",
            "required_car_parking_space": False,
            "booking_status": 0,
        },
        {
            "no_of_adults": None,
            "avg_price_per_room": 70000,
            "type_of_meal_plan": "Meal Plan 2",
            "required_car_parking_space": None,
            "booking_status": 1,
        },
    ]
    return spark.createDataFrame(data)


# Test case for the __init__ and load function
@patch.object(SparkSession, "read")
def test_data_processor_init(mock_read, mock_dataframe, spark: SparkSession):
    mock_read.table.return_value = mock_dataframe

    processor = DataProcessor(mock_config, spark)

    mock_read.table.assert_called_once_with(
        f"{mock_config.catalog}.{mock_config.db_schema}.{mock_config.use_case_name}"
    )

    assert processor.df == mock_dataframe


# Test the split_data function
@patch.object(SparkSession, "read")
def test_split_data(mock_read, mock_dataframe, spark: SparkSession):
    mock_read.table.return_value = mock_dataframe

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


@patch.object(SparkSession, "read")
def test_split_data_value_error_test_size_low(mock_read, mock_dataframe, spark: SparkSession):
    mock_read.table.return_value = mock_dataframe

    processor = DataProcessor(mock_config, spark)

    with pytest.raises(ValueError, match="test_size must be between 0 and 1, got"):
        processor.split_data(test_size=0)


@patch.object(SparkSession, "read")
def test_split_data_value_error_test_size_high(mock_read, mock_dataframe, spark: SparkSession):
    mock_read.table.return_value = mock_dataframe

    processor = DataProcessor(mock_config, spark)

    with pytest.raises(ValueError, match="test_size must be between 0 and 1, got"):
        processor.split_data(test_size=1.5)


@patch.object(SparkSession, "read")
def test_split_data_missing_target(mock_read, spark):
    data_missing_target = [
        {
            "no_of_adults": 25,
            "avg_price_per_room": 50000,
            "type_of_meal_plan": "Meal Plan 1",
            "required_car_parking_space": True,
            "booking_status": None,
        },
        {
            "no_of_adults": 30,
            "avg_price_per_room": 60000,
            "type_of_meal_plan": "Meal Plan 2",
            "required_car_parking_space": False,
            "booking_status": "",
        },
    ]
    data_non_missing_target = [
        {
            "no_of_adults": None,
            "avg_price_per_room": 50000,
            "type_of_meal_plan": "Meal Plan 1",
            "required_car_parking_space": True,
            "booking_status": "Canceled",
        },
        {
            "no_of_adults": 30,
            "avg_price_per_room": 60000,
            "type_of_meal_plan": "Meal Plan 2",
            "required_car_parking_space": False,
            "booking_status": "Not_Canceled",
        },
    ]

    mock_data = data_missing_target + data_non_missing_target
    sparse_df = spark.createDataFrame(mock_data)
    mock_read.table.return_value = sparse_df

    # Create DataProcessor instance
    processor = DataProcessor(mock_config, spark)

    # Call the split_data function
    train_data, test_data = processor.split_data()

    # Check that the train_data contains only valid rows (non-null and non-empty target)
    assert train_data.count() == len(data_non_missing_target)
