from pyspark.sql import SparkSession
from pytest import fixture, mark, param
from databricks.connect import DatabricksSession

# pytest.fixture can be used to mark unit tests, this is useful for spark unit tests because it is not always desired/possible to run them locally. This can be done by running `python -m pytest -m "not spark"`
@fixture(
    scope="session",
    params=[param("spark", marks=[mark.spark, mark.filterwarnings("ignore:distutils Version classes are deprecated")])],
)

def spark_session() -> SparkSession:
  """Creates a SparkSession fixture for the entire test session.

    Since we're using databricks-connect, a DatabricksSession is used. Ensure that you've added your cluster_id to the DEFAULT databricks profile in the .databrickscfg file.

    If you do not want to test Spark-related functionality, you can skip these
    tests by running:
        `python -m pytest -m "not spark"`

    This SHOULD disable all tests using this fixture through the mark in the
    fixture's parameters.

    Returns:
        SparkSession: A Spark session to use
    """
  spark = DatabricksSession.builder.getOrCreate()
  return spark