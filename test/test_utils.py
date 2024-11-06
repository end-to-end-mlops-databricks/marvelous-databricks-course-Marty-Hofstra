import json
import logging
from unittest.mock import Mock, mock_open, patch

import pytest
import requests
import yaml

from hotel_reservations.types.project_config_types import CatFeature, Constraints, NumFeature
from hotel_reservations.utils import (
    check_repo_info,
    get_error_metrics,
    open_config,
)
from test.utils import spark_session

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
}


def test_open_config_success():
    # Convert the mock config to YAML and mock the open function to read this as file content.
    mock_yaml_content = yaml.dump(mock_config)

    with patch("builtins.open", mock_open(read_data=mock_yaml_content)):
        with patch("yaml.safe_load", return_value=mock_config):
            # Call the open_config function to read the configuration
            config = open_config("path/to/config.yaml")

            # Perform assertions to ensure the values match
            assert config.catalog == mock_config["catalog"]
            assert config.db_schema == mock_config["schema"]
            assert config.use_case_name == mock_config["use_case_name"]
            assert config.user_dir_path == mock_config["user_dir_path"]
            assert config.git_repo == mock_config["git_repo"]
            assert config.parameters == mock_config["parameters"]
            assert config.num_features == mock_config["num_features"]
            assert config.cat_features == mock_config["cat_features"]
            assert config.target == mock_config["target"]


def test_open_config_file_not_found():
    with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            open_config("path/to/nonexistent_config.yaml")


def test_open_config_invalid_yaml():
    # Simulate an invalid YAML file by causing yaml.safe_load to raise a YAMLError.
    with patch("builtins.open", mock_open(read_data="not: valid: yaml")):
        with patch("yaml.safe_load", side_effect=yaml.YAMLError("Invalid YAML content")):
            with pytest.raises(ValueError, match="Failed to parse configuration file"):
                open_config("path/to/invalid_config.yaml")


def test_open_config_logging(caplog):
    mock_yaml_content = yaml.dump(mock_config)

    with patch("builtins.open", mock_open(read_data=mock_yaml_content)):
        with patch("yaml.safe_load", return_value=mock_config):
            with caplog.at_level(logging.INFO):
                open_config("path/to/config.yaml")
                assert "Configuration file loaded successfully" in caplog.text


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
