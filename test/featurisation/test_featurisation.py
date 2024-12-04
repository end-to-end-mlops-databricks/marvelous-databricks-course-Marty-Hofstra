from unittest.mock import patch

import pytest

from hotel_reservations.featurisation.featurisation import Featurisation
from hotel_reservations.types.project_config_types import CatFeature, Constraints, NumFeature, ProjectConfig
from test.test_conf import spark_session

spark = spark_session


@pytest.fixture
def config():
    return ProjectConfig(
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
        features_to_serve=["pk", "feature_1", "feature_2"],
    )


@pytest.fixture
def feature_data(spark):
    return spark.createDataFrame([{"id": 1, "feature": "value1"}, {"id": 2, "feature": "value2"}])


@pytest.fixture
def featurisation_instance(config, feature_data):
    return Featurisation(config=config, feature_data=feature_data, features_type="test_features", primary_key="id")


def test_initialization(featurisation_instance):
    assert featurisation_instance.feature_table_name == "my_catalog.my_schema.hotel_reservations_test_features"
    assert featurisation_instance.primary_key == "id"


def test_write_feature_table_missing_primary_key(featurisation_instance, spark):
    featurisation_instance.feature_data = featurisation_instance.feature_data.drop("id")
    with pytest.raises(ValueError, match="Primary key column 'id' not found in feature_data"):
        featurisation_instance.write_feature_table(spark)


@patch("pyspark.sql.SparkSession.sql")
def test_enable_change_data_feed(mock_sql, featurisation_instance, spark):
    result = featurisation_instance.enable_change_data_feed(spark)

    # Check if the SQL query called contains the correct statements without exact whitespace matching
    args, _ = mock_sql.call_args
    assert "ALTER TABLE" in args[0]
    assert featurisation_instance.feature_table_name in args[0]
    assert "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)" in args[0]

    assert result == f"Change data feed has been enabled for {featurisation_instance.feature_table_name}."


@patch("pyspark.sql.SparkSession.sql")
def test_check_table_CDF_property_enabled(mock_sql, featurisation_instance, spark):
    # Mock the result of the DESCRIBE DETAIL query to simulate the CDF property being enabled
    mock_sql.return_value.select.return_value.collect.return_value = [
        {"properties": {"delta.enableChangeDataFeed": "true"}}
    ]

    is_cdf_enabled = featurisation_instance.check_table_CDF_property(spark)
    assert is_cdf_enabled is True


@patch("pyspark.sql.SparkSession.sql")
def test_check_table_CDF_property_disabled(mock_sql, featurisation_instance, spark):
    # Mock the result of the DESCRIBE DETAIL query to simulate the CDF property being disabled
    mock_sql.return_value.select.return_value.collect.return_value = [
        {"properties": {"delta.enableChangeDataFeed": "false"}}
    ]

    is_cdf_enabled = featurisation_instance.check_table_CDF_property(spark)
    assert is_cdf_enabled is False
