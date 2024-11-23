from unittest.mock import patch

import pytest
from pyspark.ml.feature import Imputer, OneHotEncoder, StandardScaler, StringIndexer
from pyspark.sql import SparkSession

from hotel_reservations.models.model import Model
from hotel_reservations.types.project_config_types import CatFeature, Constraints, NumFeature, ProjectConfig
from test.test_conf import spark_session

spark = spark_session

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


@patch.object(SparkSession, "read")
def test_create_preprocessing_stages(mock_read, spark: SparkSession):
    """Test the create_preprocessing_stages method of the Model class."""
    mock_read.table.return_value = mock_dataframe
    mock_read.table.return_value = mock_dataframe

    model = Model(config=mock_config, spark=spark)

    stages = model.create_preprocessing_stages()

    # Assertions to verify the preprocessing stages are correctly created
    assert isinstance(stages, list)
    assert len(stages) > 0

    # Check if target_indexer is in the stages (StringIndexer for the target column)
    assert any(isinstance(stage, StringIndexer) for stage in stages)

    # Check if Imputer is created for numeric features
    assert any(isinstance(stage, Imputer) for stage in stages)

    # Check if StandardScaler is created
    assert any(isinstance(stage, StandardScaler) for stage in stages)

    # Check if OneHotEncoder is created for categorical features
    assert any(isinstance(stage, OneHotEncoder) for stage in stages)

    # Ensure no error is raised when preprocessing with valid config
    try:
        model.create_preprocessing_stages()
    except Exception as e:
        pytest.fail(f"create_preprocessing_stages raised an exception: {str(e)}")


@patch.object(SparkSession, "read")
def test_create_preprocessing_stages_no_features(mock_read, spark):
    """Test the create_preprocessing_stages method raises an error when no feature columns are provided."""
    mock_read.table.return_value = mock_dataframe

    # Modify the config to have no features
    mock_config.num_features = {}
    mock_config.cat_features = {}

    # Initialize the Model with the modified mock configuration
    model = Model(config=mock_config, spark=spark)

    # Ensure ValueError is raised when no features are provided
    with pytest.raises(ValueError, match="No feature columns specified in config"):
        model.create_preprocessing_stages()


@patch.object(SparkSession, "read")
def test_create_preprocessing_stages_with_empty_target(mock_read, spark):
    """Test the create_preprocessing_stages method handles missing target gracefully."""
    mock_read.table.return_value = mock_dataframe

    # Modify the config to have no target
    mock_config.target = None

    # Initialize the Model with the modified mock configuration
    model = Model(config=mock_config, spark=spark)

    # Ensure ValueError is raised when no target is specified
    with pytest.raises(ValueError, match="Target column not specified in config"):
        model.create_preprocessing_stages()
