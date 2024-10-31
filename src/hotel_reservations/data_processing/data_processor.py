from typing import List, Tuple

from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame, SparkSession

from hotel_reservations.types.project_config_types import ProjectConfigType


class DataProcessor:
    """A class to preprocess the input data

    ...

    Attributes
    ----------
    config : ProjectConfigType
        Project configuration file containing the catalog and schema where the data resides. Moreover, it contains the model parameters, numerical features, categorical features and the target variables.

    Methods
    -------
    read_UC_spark:
        Reads from Unity Catalog as a Spark Dataframe, the naming of tables in Databricks consists of three levels: catalog, schema and table name.
    preprocess_data:
        Preprocesses the data by handling missing values, scaling numeric features, and encoding categorical features
    split_data:
        Splits the DataFrame into training and test sets
    """

    def __init__(self, config: ProjectConfigType, spark: SparkSession | None = None) -> None:
        """Constructs all the necessary attributes for the preprocessing object

        Args:
            config ProjectConfigType: Project configuration file containing the catalog and schema where the data resides. Moreover, it contains the model parameters, numerical features, categorical features and the target variables.
            spark (SparkSession | None): The spark session is required for running Spark functionality outside of Databricks. Defaults to None.
        """
        self.df: DataFrame = self.read_UC_spark(config["catalog"], config["schema"], config["table_name"], spark)
        self.config: ProjectConfigType = config
        self.X: DataFrame = None
        self.y: DataFrame = None
        self.preprocessor: Pipeline = None

    def read_UC_spark(self, catalog: str, schema: str, table_name: str, spark: SparkSession | None = None) -> DataFrame:
        """Reads from Unity Catalog as a Spark Dataframe, the naming of tables in Databricks consists of three levels: catalog, schema and table name.

        Args:
            catalog (str): Catalog in which the table to dead from resides.
            schema (str): Schema/database in which the table to dead from resides,
            table_name (str): The name of the table to read from
            spark (SparkSession | None): The spark session is required for running Spark functionality outside of Databricks. Defaults to None.
        Returns:
            DataFrame: The data in PySpark format
        """
        three_level_table_name = f"{catalog}.{schema}.{table_name}"
        return spark.read.table(three_level_table_name)  # type: ignore

    def preprocess_data(self) -> None:
        """Preprocessing of data, consisting of the following steps:
        - Imputation of missing values
        - Handling of categorical features
        - Use of the VectorAssembler to combine numeric and categorical features
        - Combine all numeric and categorical features into one feature column
        - Build the preprocessing pipeline
        """
        target: str = self.config["target"]
        self.df = self.df.dropna(subset=[target])

        numeric_imputer: Imputer = Imputer(
            inputCols=self.config["num_features"], outputCols=[f"{c}_imputed" for c in self.config["num_features"]]
        )

        scaler: StandardScaler = StandardScaler(inputCol="features_num", outputCol="scaled_features")

        indexers: List[StringIndexer] = [
            StringIndexer(inputCol=col, outputCol=f"{col}_indexed") for col in self.config["cat_features"]
        ]
        encoders: List[OneHotEncoder] = [
            OneHotEncoder(inputCol=f"{col}_indexed", outputCol=f"{col}_encoded") for col in self.config["cat_features"]
        ]
        assembler_numeric: VectorAssembler = VectorAssembler(
            inputCols=[f"{c}_imputed" for c in self.config["num_features"]], outputCol="features_num"
        )

        assembler_categorical: VectorAssembler = VectorAssembler(
            inputCols=[f"{col}_encoded" for col in self.config["cat_features"]], outputCol="features_cat"
        )

        assembler_all: VectorAssembler = VectorAssembler(
            inputCols=["scaled_features", "features_cat"], outputCol="features"
        )

        stages: List = (
            [numeric_imputer, assembler_numeric, scaler] + indexers + encoders + [assembler_categorical, assembler_all]
        )
        self.preprocessor = Pipeline(stages=stages)

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[DataFrame, DataFrame]:
        """Splits the DataFrame into training and test sets

        Args:
            test_size (float, optional): Proportion of the input data to be part of the test set. Defaults to 0.2.
            random_state (int, optional): Value of the state. Defaults to 42.

        Returns:
            Tuple[DataFrame, DataFrame]: The input data split up into a training and test set
        """
        train: DataFrame
        test: DataFrame
        train, test = self.df.randomSplit([1.0 - test_size, test_size], seed=random_state)
        return train, test
