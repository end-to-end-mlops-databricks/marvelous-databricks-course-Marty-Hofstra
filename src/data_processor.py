from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from typing import List, Tuple, Dict

class DataProcessor:
    def __init__(self, config: Dict[str, List[str]]) -> None:
        self.df: DataFrame = self.read_UC_spark(config['catalog'], config['schema'], config['table_name'])
        self.config: Dict[str, List[str]] = config
        self.X: DataFrame = None  # Placeholder for feature matrix"
        self.y: DataFrame = None  # Placeholder for target variable
        self.preprocessor: Pipeline = None  # Placeholder for preprocessing pipeline

    def read_UC_spark(self, catalog: str, schema: str, table_name: str) -> DataFrame:
        """Loads data from a CSV file into a PySpark DataFrame."""
        three_level_table_name = f"{catalog}.{schema}.{table_name}"
        return spark.read.table(three_level_table_name)

    def preprocess_data(self) -> None:
        """Preprocesses the data by handling missing values, scaling numeric features, and encoding categorical features."""
        target: str = self.config['target']
        self.df = self.df.dropna(subset=[target])

        # Handle missing numeric values and scaling
        numeric_imputer: Imputer = Imputer(
            inputCols=self.config['num_features'], 
            outputCols=[f'{c}_imputed' for c in self.config['num_features']]
        )

        scaler: StandardScaler = StandardScaler(inputCol='features_num', outputCol='scaled_features')

        # Handle categorical features
        indexers: List[StringIndexer] = [StringIndexer(inputCol=col, outputCol=f'{col}_indexed') for col in self.config['cat_features']]
        encoders: List[OneHotEncoder] = [OneHotEncoder(inputCol=f'{col}_indexed', outputCol=f'{col}_encoded') for col in self.config['cat_features']]

        # VectorAssembler to combine numeric and categorical features
        assembler_numeric: VectorAssembler = VectorAssembler(
            inputCols=[f'{c}_imputed' for c in self.config['num_features']],
            outputCol='features_num'
        )

        assembler_categorical: VectorAssembler = VectorAssembler(
            inputCols=[f'{col}_encoded' for col in self.config['cat_features']],
            outputCol='features_cat'
        )

        # Combine all numeric and categorical features into one feature column
        assembler_all: VectorAssembler = VectorAssembler(
            inputCols=['scaled_features', 'features_cat'],
            outputCol='features'
        )

        # Build the preprocessing pipeline
        stages: List = [numeric_imputer, assembler_numeric, scaler] + indexers + encoders + [assembler_categorical, assembler_all]
        self.preprocessor = Pipeline(stages=stages)

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[DataFrame, DataFrame]:
        """Splits the DataFrame into training and test sets."""
        train: DataFrame
        test: DataFrame
        train, test = self.df.randomSplit([1.0 - test_size, test_size], seed=random_state)
        return train, test