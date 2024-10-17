import pytest
from unittest.mock import patch, MagicMock
from databricks.connect import DatabricksSession
from pyspark.sql import DataFrame, Row
from typing import List, Dict
from src.data_processor import DataProcessor
import os

# Sample mock config
mock_config = {
    'catalog': 'my_catalog',
    'schema': 'my_schema',
    'table_name': 'my_table',
    'num_features': ['age', 'income'],
    'cat_features': ['gender', 'city'],
    'target': 'purchased'
}

# Fixture for remote Spark session via Databricks Connect
@pytest.fixture(scope="module")
def spark():
    os.environ["DATABRICKS_HOST"] = "https://adb-4802833999154755.15.azuredatabricks.net"
    os.environ["DATABRICKS_CLUSTER_ID"] = "1020-093808-8j3fu6s5"
    
    # Create a remote Spark session using Databricks Connect
    return DatabricksSession.builder.getOrCreate()

@pytest.fixture
def mock_dataframe(spark):
    # Create a mock dataframe with remote Databricks Spark session
    data = [
        {'age': 25, 'income': 50000, 'gender': 'M', 'city': 'NY', 'purchased': 1},
        {'age': 30, 'income': 60000, 'gender': 'F', 'city': 'LA', 'purchased': 0},
        {'age': None, 'income': 70000, 'gender': 'F', 'city': None, 'purchased': 1}
    ]
    return spark.createDataFrame(data)

# Test case for the __init__ and load function
@patch('src.data_processor.DataProcessor.read_UC_spark')
def test_data_processor_init(mock_read_UC_spark, mock_dataframe):
    mock_read_UC_spark.return_value = mock_dataframe
    
    # Initialize DataProcessor with mocked config
    processor = DataProcessor(mock_config)
    
    # Ensure that the read_UC_spark method is called correctly
    mock_read_UC_spark.assert_called_once_with(mock_config['catalog'], mock_config['schema'], mock_config['table_name'])

    # Assert that DataFrame is loaded correctly
    assert processor.df == mock_dataframe

# Test the split_data function
@patch('src.data_processor.DataProcessor.read_UC_spark')
def test_split_data(mock_read_UC_spark, mock_dataframe):
    mock_read_UC_spark.return_value = mock_dataframe

    processor = DataProcessor(mock_config)
    train, test = processor.split_data(test_size=0.5, random_state=42)

    # Assert that split_data returns two DataFrames
    assert isinstance(train, DataFrame)
    assert isinstance(test, DataFrame)

    # Test split sizes
    total_rows = mock_dataframe.count()
    assert train.count() + test.count() == total_rows

# Test edge case: if no data is available after dropping missing target values
@patch('src.data_processor.DataProcessor.read_UC_spark')
def test_no_data_after_dropping(mock_read_UC_spark, spark):
    # Create a DataFrame where all rows have missing target
    mock_data = [
        Row(age=25, income=50000, gender='M', city='NY', purchased=None),
        Row(age=30, income=60000, gender='F', city='LA', purchased=None)
    ]
    empty_df = spark.createDataFrame(mock_data)
    mock_read_UC_spark.return_value = empty_df

    processor = DataProcessor(mock_config)
    processor.preprocess_data()

    # After dropping rows with missing target, DataFrame should be empty
    assert processor.df.count() == 0

