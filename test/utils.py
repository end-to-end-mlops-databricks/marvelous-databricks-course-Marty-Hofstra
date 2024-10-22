from pyspark.sql import SparkSession
from pytest import fixture, mark, param
from pyspark import SparkConf

@fixture(
    scope="session",
    params=[param("spark", marks=[mark.spark, mark.filterwarnings("ignore:distutils Version classes are deprecated")])],
)
def spark_session() -> SparkSession:
    """Creates a SparkSession fixture for the entire test session.

    This is setup such that that SparkSession should only be initialised once.
    Since it can take a long time for Spark to start, this should save us time.

    If you do not want to test Spark-related functionality, you can skip these
    tests by running:
        `python -m pytest -m "not spark"`

    This SHOULD disable all tests using this fixture through the mark in the
    fixture's parameters.

    Returns:
        SparkSession: A Spark session to use, shared between tests to save
            boot time.
    """
    print("Launching Spark session: please wait...")
    config = (
        SparkConf()
        .set("spark.sql.execution.arrow.enabled", "False")
        .set("spark.sql.execution.arrow.pyspark.enabled", "False")
    )
    return SparkSession.builder.config(conf=config).getOrCreate()
    