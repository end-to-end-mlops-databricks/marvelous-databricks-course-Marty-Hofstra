from pyspark.sql import SparkSession
from pytest import fixture, mark, param


# pytest.fixture can be used to mark unit tests, this is useful for spark unit tests because it is not always desired/possible to run them locally. This can be done by running `python -m pytest -m "not spark"`
@fixture(
    scope="session",
    params=[param("spark", marks=[mark.spark, mark.filterwarnings("ignore:distutils Version classes are deprecated")])],
)
def spark_session() -> SparkSession:
    """Creates a SparkSession fixture for the entire test session.

    Even though we're using databricks-connect, a SparkSession is used. This is due to the fact that a DatabricksSession has limitations when working with certain PySpark functionality (e.g. pyspark.ml.features)
    If you do not want to test Spark-related functionality, you can skip these
    tests by running:
        `python -m pytest -m "not spark"`

    This SHOULD disable all tests using this fixture through the mark in the
    fixture's parameters.

    Returns:
        SparkSession: A Spark session to use
    """
    return SparkSession.builder.appName("MyApp").master("local[*]").getOrCreate()
