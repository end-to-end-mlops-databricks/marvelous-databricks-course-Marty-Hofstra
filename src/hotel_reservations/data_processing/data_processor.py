from typing import List, Optional, Tuple

from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame, SparkSession


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

    def __init__(self, config: dict, spark: SparkSession) -> None:
        """Constructs all the necessary attributes for the preprocessing object

        Args:
            config (dict): Project configuration file converted to dict, containing the catalog and schema where the data resides. Moreover, it contains the model parameters, numerical features, categorical features and the target variables.
            spark (SparkSession): The spark session is required for running Spark functionality outside of Databricks.
        """
        self.config: dict = config
        self.df: DataFrame = self.read_UC_spark(config["catalog"], config["db_schema"], config["table_name"], spark)
        self.X: Optional[DataFrame] = None
        self.y: Optional[DataFrame] = None
        self.preprocessor: Optional[Pipeline] = None

    def read_UC_spark(self, catalog: str, schema: str, table_name: str, spark: SparkSession) -> DataFrame:
        """Reads from Unity Catalog as a Spark Dataframe, the naming of tables in Databricks consists of three levels: catalog, schema and table name.

        Args:
            catalog (str): Catalog from which the table is read.
            schema (str): Schema/database from which the table is read.
            table_name (str): The name of the table to read from
            spark (SparkSession): The spark session is required for running Spark functionality outside of Databricks.

        Raises:
            ValueError: If the table in UC cannot be read

        Returns:
            DataFrame: The data in PySpark format
        """
        three_level_table_name = f"{catalog}.{schema}.{table_name}"
        try:
            return spark.read.table(three_level_table_name)
        except Exception as e:
            raise ValueError(f"Failed to read table '{three_level_table_name}': {str(e)}") from e

    def preprocess_data(self) -> List:
        """Preprocessing of data, consisting of the following steps:
        - Imputation of missing values
        - Handling of categorical features
        - Use of the VectorAssembler to combine numeric and categorical features
        - Combine all numeric and categorical features into one feature column
        - Build the preprocessing pipeline

        Returns:
            List: Preprocessing stages required for the PySpark ML pipeline
        """
        target: str = self.config["target"]
        self.df = self.df.dropna(subset=[target])

        # Extracting input column names from the config
        num_feature_cols = list(self.config["num_features"].keys())
        cat_feature_cols = list(self.config["cat_features"].keys())

        target_indexer: StringIndexer = StringIndexer(inputCol=target, outputCol="label")

        # Set up the imputer
        numeric_imputer: Imputer = Imputer(
            inputCols=num_feature_cols, outputCols=[f"{c}_imputed" for c in num_feature_cols]
        )

        # Create the scaler
        scaler: StandardScaler = StandardScaler(inputCol="features_num", outputCol="scaled_features")

        # StringIndexer and OneHotEncoder for categorical features
        indexers: List[StringIndexer] = [
            StringIndexer(inputCol=col, outputCol=f"{col}_indexed") for col in cat_feature_cols
        ]
        encoders: List[OneHotEncoder] = [
            OneHotEncoder(inputCol=f"{col}_indexed", outputCol=f"{col}_encoded") for col in cat_feature_cols
        ]

        # Assemble numeric features
        assembler_numeric: VectorAssembler = VectorAssembler(
            inputCols=[f"{c}_imputed" for c in num_feature_cols], outputCol="features_num"
        )

        # Assemble categorical features
        assembler_categorical: VectorAssembler = VectorAssembler(
            inputCols=[f"{col}_encoded" for col in cat_feature_cols], outputCol="features_cat"
        )

        # Combine numeric and categorical features
        assembler_all: VectorAssembler = VectorAssembler(
            inputCols=["scaled_features", "features_cat"], outputCol="features"
        )

        # Building the pipeline
        preprocessing_stages: List = (
            [target_indexer, numeric_imputer, assembler_numeric, scaler]
            + indexers
            + encoders
            + [assembler_categorical, assembler_all]
        )

        return preprocessing_stages

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[DataFrame, DataFrame]:
        """Splits the DataFrame into training and test sets

        Args:
            test_size (float, optional): Proportion of the input data to be part of the test set. Defaults to 0.2.
            random_state (int, optional): Value of the state. Defaults to 42.

        Raises:
            ValueError: If `test_size` is not between 0 and 1.

        Returns:
            Tuple[DataFrame, DataFrame]: The input data split up into a training and test set
        """
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

        if self.df.isEmpty():
            raise ValueError("Cannot split an empty DataFrame")

        train: DataFrame
        test: DataFrame
        train, test = self.df.randomSplit([1.0 - test_size, test_size], seed=random_state)
        return train, test
