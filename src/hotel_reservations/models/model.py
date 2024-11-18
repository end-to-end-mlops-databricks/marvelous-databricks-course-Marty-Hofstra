from pyspark.ml.feature import Imputer, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession

from hotel_reservations.types.project_config_types import ProjectConfig


class Model:
    """A class for creating a PySpark ML model

    Attributes
    ----------
    config: ProjectConfig
        Project configuration file containing the catalog and schema where the data resides. Moreover, it contains the model parameters, numerical features, categorical features and the target variables.
    spark: SparkSession
        The Spark session is required for running Spark functionality outside of Databricks.

    Methods
    -------
    create_preprocessing_stages:
        Creates ML Pipeline preprocessing stages
    """

    def __init__(self, config: ProjectConfig, spark: SparkSession) -> None:
        """Constructs all the necessary attributes for the modelling object

        Args:
            config (ProjectConfig): Project configuration file converted to dict, containing the catalog and schema where the data resides. Moreover, it contains the model parameters, numerical features, categorical features and the target variables.
            spark (SparkSession): The spark session is required for running Spark functionality outside of Databricks.
        """
        self.config: ProjectConfig = config
        try:
            self.train_data = spark.read.table(f"{config.catalog}.{config.db_schema}.{config.use_case_name}_train_data")
            self.test_data = spark.read.table(f"{config.catalog}.{config.db_schema}.{config.use_case_name}_test_data")
        except Exception as e:
            raise RuntimeError("Failed to read training or testing data tables") from e

    def create_preprocessing_stages(self) -> list:
        """Creates the following ML Pipeline preprocessing stages:
        - Handling of categorical features
        - Use of the VectorAssembler to combine numeric and categorical features
        - Combine all numeric and categorical features into one feature column
        - Build the preprocessing pipeline

        Returns:
            List: Preprocessing stages required for the PySpark ML pipeline
        """
        target: str = self.config.target

        if target is None:
            raise ValueError("Target column not specified in config")

        # Extracting input column names from the config
        num_feature_cols = list(self.config.num_features.keys())
        cat_feature_cols = list(self.config.cat_features.keys())

        if not num_feature_cols and not cat_feature_cols:
            raise ValueError("No feature columns specified in config")

        target_indexer: StringIndexer = StringIndexer(inputCol=target, outputCol="label")

        # Set up the imputer
        numeric_imputer: Imputer = Imputer(
            inputCols=num_feature_cols, outputCols=[f"{c}_imputed" for c in num_feature_cols]
        )

        # Create the scaler
        scaler: StandardScaler = StandardScaler(inputCol="features_num", outputCol="scaled_features")

        # StringIndexer and OneHotEncoder for categorical features
        indexers: list[StringIndexer] = [
            StringIndexer(inputCol=col, outputCol=f"{col}_indexed") for col in cat_feature_cols
        ]
        encoders: list[OneHotEncoder] = [
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
        preprocessing_stages: list = (
            [target_indexer, numeric_imputer, assembler_numeric, scaler]
            + indexers
            + encoders
            + [assembler_categorical, assembler_all]
        )

        return preprocessing_stages
