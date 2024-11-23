from unittest.mock import MagicMock, patch

import pytest
from databricks.sdk.service.catalog import MonitorSnapshot

from hotel_reservations.monitoring.monitoring import Monitoring
from hotel_reservations.types.project_config_types import CatFeature, Constraints, NumFeature, ProjectConfig

# Mock configuration for testing
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


# Fixtures
@pytest.fixture
def mock_workspace_client():
    """
    Fixture to mock the WorkspaceClient.
    """
    with patch("hotel_reservations.monitoring.monitoring.WorkspaceClient") as MockClient:
        yield MockClient.return_value


@pytest.fixture
def monitoring(mock_workspace_client):
    """
    Fixture to initialize the Monitoring class with a mock WorkspaceClient.
    """
    return Monitoring(config=mock_config, table_name="test_table")


# Unit Tests
def test_create_lakehouse_monitor_exists(monitoring, mock_workspace_client):
    """
    Test the create_lakehouse_monitor method when the monitor already exists.
    """
    # Mock the behavior of the quality_monitors.get method to simulate an existing monitor
    mock_workspace_client.quality_monitors.get = MagicMock()

    # Execute the method
    monitoring.create_lakehouse_monitor()

    # Assert that get was called and create was not called
    mock_workspace_client.quality_monitors.get.assert_called_once_with("test_table")
    mock_workspace_client.quality_monitors.create.assert_not_called()


def test_create_lakehouse_monitor_create_new(monitoring, mock_workspace_client):
    """
    Test the create_lakehouse_monitor method when the monitor does not exist.
    """
    # Mock the behavior of the quality_monitors.get method to raise an exception
    mock_workspace_client.quality_monitors.get = MagicMock(side_effect=Exception("Monitor does not exist"))
    mock_workspace_client.quality_monitors.create = MagicMock()

    # Execute the method
    monitoring.create_lakehouse_monitor()

    # Assert that get was called and create was called with the correct parameters
    mock_workspace_client.quality_monitors.get.assert_called_once_with("test_table")
    mock_workspace_client.quality_monitors.create.assert_called_once_with(
        table_name="test_table",
        assets_dir=f"/Workspace/{mock_config.user_dir_path}/lakehouse_monitoring/test_table",
        output_schema_name=f"{mock_config.catalog}.{mock_config.db_schema}",
        snapshot=MonitorSnapshot(),
    )


def test_refresh_monitor(monitoring, mock_workspace_client):
    """
    Test the refresh_monitor method.
    """
    # Mock the behavior of the quality_monitors.run_refresh method
    mock_workspace_client.quality_monitors.run_refresh = MagicMock()

    # Execute the method
    monitoring.refresh_monitor()

    # Assert that run_refresh was called with the correct parameters
    mock_workspace_client.quality_monitors.run_refresh.assert_called_once_with(table_name="test_table")
