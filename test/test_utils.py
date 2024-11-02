import logging
from unittest.mock import mock_open, patch

import pytest
import yaml

from hotel_reservations.utils import get_error_metrics, open_config
from test.utils import spark_session


def test_open_config_success():
    # Define a valid config dictionary that would match the ProjectConfigType structure.
    mock_config = {
        "catalog": "my_catalog",
        "schema": "my_schema",
        "model_parameters": {"param1": 0.1},
        "numerical_features": ["feature1", "feature2"],
        "categorical_features": ["cat_feature1", "cat_feature2"],
        "target_variable": "target",
    }

    # Convert the mock config to YAML and mock the open function to read this as file content.
    mock_yaml_content = yaml.dump(mock_config)

    with patch("builtins.open", mock_open(read_data=mock_yaml_content)):
        with patch("yaml.safe_load", return_value=mock_config):
            config = open_config("path/to/config.yaml")
            assert config == mock_config  # Check that the config matches the expected dictionary


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
    # Test if the logging occurs as expected when loading the file successfully.
    mock_config = {"catalog": "catalog_name"}  # Minimal valid config for simplicity
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
