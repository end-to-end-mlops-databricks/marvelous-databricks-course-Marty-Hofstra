import json
from unittest.mock import Mock, mock_open, patch

import pandas as pd
import pytest
import requests
import yaml
from pyspark.sql.functions import col
from pyspark.sql.functions import max as spark_max
from pyspark.sql.functions import min as spark_min
from pyspark.sql.types import DoubleType, IntegerType

from hotel_reservations.types.project_config_types import CatFeature, Constraints, NumFeature
from hotel_reservations.utils import (
    check_repo_info,
    generate_booking_ids_regex,
    generate_synthetic_data,
    get_error_metrics,
    open_config,
)
from test.test_conf import spark_session

spark = spark_session

mock_config: dict = {
    "catalog": "my_catalog",
    "schema": "my_schema",
    "use_case_name": "hotel_reservations",
    "user_dir_path": "/Users/user/",
    "git_repo": "git_repo",
    "volume_whl_path": "Volumes/users/user/packages/",
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
    "primary_key": "Booking_ID",
    "features_to_serve": ["pk", "feature_1", "feature_2"],
}


def test_open_config_file_not_found():
    with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            open_config("path/to/nonexistent_config.yaml", "scope")


def test_open_config_invalid_yaml():
    # Simulate an invalid YAML file by causing yaml.safe_load to raise a YAMLError.
    with patch("builtins.open", mock_open(read_data="not: valid: yaml")):
        with patch("yaml.safe_load", side_effect=yaml.YAMLError("Invalid YAML content")):
            with pytest.raises(ValueError, match="Failed to parse configuration file"):
                open_config("path/to/invalid_config.yaml", "scope")


spark = spark_session


@pytest.fixture
def sample_predictions(spark):
    data = [(3.0, 2.5), (5.0, 5.1), (6.0, 5.9), (8.0, 7.8), (9.0, 9.2)]
    columns = ["label", "prediction"]
    return spark.createDataFrame(data, columns)


def test_get_error_metrics_valid_data(sample_predictions):
    """Test that error metrics are calculated correctly for valid input"""
    error_metrics = get_error_metrics(sample_predictions)
    assert "mse" in error_metrics
    assert "mae" in error_metrics
    assert "r2" in error_metrics

    # Check metric values within expected ranges (these are example thresholds)
    assert 0 <= error_metrics["mse"] < 1, "MSE out of expected range"
    assert 0 <= error_metrics["mae"] < 1, "MAE out of expected range"
    assert 0 <= error_metrics["r2"] <= 1, "R2 out of expected range"


def test_get_error_metrics_missing_columns(spark):
    """Test handling of missing label or prediction column"""
    # Missing 'label' column
    data_missing_label = [(2.5,), (5.1,), (5.9,)]
    df_missing_label = spark.createDataFrame(data_missing_label, ["prediction"])

    with pytest.raises(ValueError):
        get_error_metrics(df_missing_label, label_col_name="label")

    # Missing 'prediction' column
    data_missing_prediction = [(3.0,), (5.0,), (6.0,)]
    df_missing_prediction = spark.createDataFrame(data_missing_prediction, ["label"])

    with pytest.raises(ValueError):
        get_error_metrics(df_missing_prediction, pred_col_name="prediction")


def test_check_repo_info_success():
    # Mock the dbutils object and its method calls
    mock_dbutils = Mock()
    mock_context_json = json.dumps(
        {"extraContext": {"api_url": "https://mock.databricks-instance.com", "api_token": "mock_token"}}
    )
    mock_dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson.return_value = mock_context_json

    # Mock the requests.get response
    mock_response = Mock()
    mock_response.json.return_value = {"repos": [{"branch": "main", "head_commit_id": "abc123def456"}]}

    repo_path = "/Repos/mock-user/mock-repo"

    with patch("requests.get", return_value=mock_response):
        branch, sha = check_repo_info(repo_path, dbutils=mock_dbutils)

        # Assert that the correct API endpoint was called with correct headers and params
        requests.get.assert_called_once_with(
            "https://mock.databricks-instance.com/api/2.0/repos",
            headers={"Authorization": "Bearer mock_token"},
            params={"path_prefix": repo_path},
        )

        # Assert that the branch and SHA returned match the mock data
        assert branch == "main"
        assert sha == "abc123def456"


def test_check_repo_info_no_repo_found():
    # Mock the dbutils object and its method calls
    mock_dbutils = Mock()
    mock_context_json = json.dumps(
        {"extraContext": {"api_url": "https://mock.databricks-instance.com", "api_token": "mock_token"}}
    )
    mock_dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson.return_value = mock_context_json

    # Mock the requests.get response to simulate no repository found
    mock_response = Mock()
    mock_response.json.return_value = {"repos": []}

    repo_path = "/Repos/mock-user/non-existent-repo"

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(IndexError):
            check_repo_info(repo_path, dbutils=mock_dbutils)


def test_check_repo_info_invalid_token():
    # Mock the dbutils object and its method calls
    mock_dbutils = Mock()
    mock_context_json = json.dumps(
        {"extraContext": {"api_url": "https://mock.databricks-instance.com", "api_token": "invalid_token"}}
    )
    mock_dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson.return_value = mock_context_json

    # Mock the requests.get response to simulate an authentication error
    mock_response = Mock()
    mock_response.json.side_effect = requests.exceptions.HTTPError("401 Client Error: Unauthorized for url")

    repo_path = "/Repos/mock-user/mock-repo"

    with patch("requests.get", side_effect=mock_response.json.side_effect):
        with pytest.raises(requests.exceptions.HTTPError):
            check_repo_info(repo_path, dbutils=mock_dbutils)


def test_check_repo_info_rate_limit():
    """Test handling of rate limit response."""
    mock_dbutils = Mock()
    mock_context_json = json.dumps(
        {"extraContext": {"api_url": "https://mock.databricks-instance.com", "api_token": "mock_token"}}
    )
    mock_dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson.return_value = mock_context_json

    mock_response = Mock()
    mock_response.json.side_effect = requests.exceptions.HTTPError("429 Client Error: Too Many Requests")

    repo_path = "/Repos/mock-user/mock-repo"

    with patch("requests.get", side_effect=mock_response.json.side_effect):
        with pytest.raises(requests.exceptions.HTTPError, match="429 Client Error"):
            check_repo_info(repo_path, dbutils=mock_dbutils)


def test_generate_booking_ids_regex_basic():
    existing_ids = ["INN1205", "INN1207"]
    num_new_ids = 3
    prefix = "INN"

    result = generate_booking_ids_regex(existing_ids, num_new_ids, prefix)

    # Expected new IDs based on the given input
    expected = ["INN1208", "INN1209", "INN1210"]
    assert result == expected, f"Expected {expected}, but got {result}"


def test_generate_booking_ids_regex_custom_prefix():
    existing_ids = ["BOOK1001", "BOOK1002"]
    num_new_ids = 2
    prefix = "BOOK"

    result = generate_booking_ids_regex(existing_ids, num_new_ids, prefix)

    expected = ["BOOK1003", "BOOK1004"]
    assert result == expected, f"Expected {expected}, but got {result}"


def test_generate_booking_ids_regex_empty_existing_ids():
    existing_ids = []
    num_new_ids = 3
    prefix = "INN"

    # Should handle empty input gracefully
    with pytest.raises(ValueError):
        generate_booking_ids_regex(existing_ids, num_new_ids, prefix)


def test_generate_booking_ids_regex_large_range():
    existing_ids = ["INN99998", "INN99999"]
    num_new_ids = 3
    prefix = "INN"

    result = generate_booking_ids_regex(existing_ids, num_new_ids, prefix)

    expected = ["INN100000", "INN100001", "INN100002"]
    assert result == expected, f"Expected {expected}, but got {result}"


def test_generate_booking_ids_regex_no_new_ids():
    existing_ids = ["INN01205", "INN01207"]
    num_new_ids = 0
    prefix = "INN"

    result = generate_booking_ids_regex(existing_ids, num_new_ids, prefix)
    expected = []  # No new IDs generated
    assert result == expected, f"Expected {expected}, but got {result}"


def test_generate_booking_ids_regex_variable_padding():
    # Existing IDs with variable padding lengths
    existing_ids = ["INN001", "INN099", "INN100"]
    num_new_ids = 3
    prefix = "INN"

    result = generate_booking_ids_regex(existing_ids, num_new_ids, prefix)

    # Padding length should dynamically adjust to 3 (matching 'INN100')
    expected = ["INN101", "INN102", "INN103"]
    assert result == expected, f"Expected {expected}, but got {result}"


def test_generate_booking_ids_regex_mixed_prefix_lengths():
    # Mixed prefix lengths to ensure function handles prefixes properly
    existing_ids = ["BOOK1", "BOOK12", "BOOK99"]
    num_new_ids = 3
    prefix = "BOOK"

    result = generate_booking_ids_regex(existing_ids, num_new_ids, prefix)

    # Padding length should adjust to 2 (matching 'BOOK99')
    expected = ["BOOK100", "BOOK101", "BOOK102"]
    assert result == expected, f"Expected {expected}, but got {result}"


@pytest.fixture
def mock_input_data(spark):
    data = {
        "no_of_adults": [1, 2, 3],
        "avg_price_per_room": [100.0, 200.0, 150.0],
        "no_of_week_nights": [3, 2, 1],
        "type_of_meal_plan": ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3"],
        "required_car_parking_space": [1, 0, 1],
        "booking_id": ["INN001", "INN002", "INN003"],
    }
    return spark.createDataFrame(pd.DataFrame(data))


def test_generate_synthetic_data_basic(mock_input_data):
    synthetic_data = generate_synthetic_data(mock_input_data, num_rows=50)

    assert synthetic_data.count() == 50
    expected_columns = [
        "Booking_ID",
        "no_of_adults",
        "no_of_children",
        "no_of_weekend_nights",
        "no_of_week_nights",
        "type_of_meal_plan",
        "required_car_parking_space",
        "room_type_reserved",
        "lead_time",
        "arrival_year",
        "arrival_month",
        "arrival_date",
        "market_segment_type",
        "repeated_guest",
        "no_of_previous_cancellations",
        "no_of_previous_bookings_not_canceled",
        "avg_price_per_room",
        "no_of_special_requests",
        "booking_status",
    ]
    assert set(synthetic_data.columns) == set(expected_columns)
    assert synthetic_data.schema["no_of_adults"].dataType == IntegerType()
    assert synthetic_data.schema["avg_price_per_room"].dataType == DoubleType()

    # Add value range validation
    adults_stats = synthetic_data.select(
        spark_min("no_of_adults").alias("min"), spark_max("no_of_adults").alias("max")
    ).collect()[0]
    assert adults_stats.min >= 1, "no_of_adults should be at least 1"
    assert adults_stats.max <= 10, "no_of_adults should not exceed 10"


def test_generate_synthetic_data_num_rows_cap(mock_input_data):
    synthetic_data = generate_synthetic_data(mock_input_data, num_rows=200000)
    # Ensure num_rows is capped at 100,000
    assert synthetic_data.count() == 100000


def test_generate_synthetic_data_with_drift(mock_input_data):
    synthetic_data = generate_synthetic_data(mock_input_data, num_rows=50, drift=True)
    # Check drift modifications
    max_price_in_synthetic = synthetic_data.select(col("avg_price_per_room")).rdd.map(lambda row: row[0]).max()
    max_price_in_input = mock_input_data.select(col("avg_price_per_room")).rdd.map(lambda row: row[0]).max()
    assert max_price_in_synthetic >= max_price_in_input * 1.5


def test_generate_synthetic_data_unique_ids(mock_input_data):
    synthetic_data = generate_synthetic_data(mock_input_data, num_rows=50)
    existing_ids = set(mock_input_data.select("Booking_ID").rdd.flatMap(lambda x: x).collect())
    synthetic_ids = set(synthetic_data.select("Booking_ID").rdd.flatMap(lambda x: x).collect())
    # Check that synthetic IDs are unique and do not overlap with existing IDs
    assert len(synthetic_ids) == 50
    assert synthetic_ids.isdisjoint(existing_ids)
