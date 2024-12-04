from typing import Optional, Tuple

from pyspark.ml import Pipeline
from pyspark.sql import DataFrame, SparkSession

from hotel_reservations.types.project_config_types import ProjectConfig


class DataProcessor:
    """A class to preprocess the input data

    Attributes
    ----------
    config: ProjectConfig
        Project configuration file containing the catalog and schema where the data resides. Moreover, it contains the model parameters, numerical features, categorical features and the target variables.
    spark: SparkSession
        The Spark session is required for running Spark functionality outside of Databricks.

    Methods
    -------
    split_data:
        Splits the DataFrame into training and test sets
    """

    def __init__(self, config: ProjectConfig, spark: SparkSession, drift: bool = False) -> None:
        """Constructs all the necessary attributes for the preprocessing object

        Args:
            config (ProjectConfig): Project configuration file converted to dict, containing the catalog and schema where the data resides. Moreover, it contains the model parameters, numerical features, categorical features and the target variables.
            spark (SparkSession): The spark session is required for running Spark functionality outside of Databricks.
            drift (bool): True if synthetic drift data has to be processed.
        """
        self.config: ProjectConfig = config
        self.suffix: str = "_skewed" if drift else ""
        self.df: DataFrame = spark.read.table(
            f"{config.catalog}.{config.db_schema}.{config.use_case_name}{self.suffix}"
        )
        self.X: Optional[DataFrame] = None
        self.y: Optional[DataFrame] = None
        self.preprocessor: Optional[Pipeline] = None

    def drop_missing_target(self) -> None:
        """Drops rows with missing target values"""
        target: str = self.config.target
        self.df = self.df.dropna(subset=[target])

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[DataFrame, DataFrame]:
        """Splits the DataFrame into training and test sets, the missing target values are dropped.

        Args:
            test_size (float, optional): Proportion of the input data to be part of the test set. Defaults to 0.2.
            random_state (int, optional): Value of the state. Defaults to 42.

        Raises:
            ValueError: If `test_size` is not between 0 and 1.

        Returns:
            train_data (DataFrame): Data used for training the model
            test_data (DataFrame): Data used for testing the model
        """
        target: str = self.config.target
        self.df = self.df.dropna(subset=[target])

        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

        if self.df.isEmpty():
            raise ValueError("Cannot split an empty DataFrame")

        self.drop_missing_target()

        train_data: DataFrame
        test_data: DataFrame
        train_data, test_data = self.df.randomSplit([1.0 - test_size, test_size], seed=random_state)
        return train_data, test_data
