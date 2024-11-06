from unittest.mock import MagicMock

import pytest

from hotel_reservations.featurisation.featurisation import write_feature_table
from test.utils import spark_session

spark = spark_session


@pytest.fixture
def feature_data():
    """Fixture to create a mock feature data DataFrame."""
    data = MagicMock()  # Mocked DataFrame, could replace with actual data for more specific tests
    data.columns = ["feature1", "feature2", "primary_key"]  # Mock columns for testing
    return data


def test_write_feature_table_primary_key_not_found(spark, feature_data):
    """Test case where the primary key is not found in the feature data."""

    # Remove primary_key column to simulate error
    feature_data.columns = ["feature1", "feature2"]

    with pytest.raises(ValueError, match="Primary key column 'primary_key' not found in feature_data"):
        write_feature_table(
            feature_data=feature_data,
            catalog="my_catalog",
            schema="my_schema",
            use_case_name="hotel_reservation",
            primary_key="primary_key",
            spark=spark,
        )
