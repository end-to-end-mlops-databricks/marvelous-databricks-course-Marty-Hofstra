from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import DataFrame, SparkSession

from hotel_reservations.types.project_config_types import ProjectConfig


class Featurisation:
    def __init__(self, config: ProjectConfig, feature_data: DataFrame, features_type: str, primary_key: str) -> None:
        """Constructs all the necessary attributes for the featurisation object

        Args:
            feature_data (DataFrame): Dataframe containing feature data to write to the Feature Store
            features_type (str): Type of features for clear naming of the feature table, e.g. 'predictions_features'
            primary_key (str): Name of the column to use as PK in the Feature table
        """
        self.config: ProjectConfig = config
        self.feature_data: DataFrame = feature_data
        self.catalog: str = config.catalog
        self.schema: str = config.db_schema
        self.use_case_name: str = config.use_case_name
        self.features_type: str = features_type
        self.primary_key: str = primary_key

    def write_feature_table(
        self,
        spark: SparkSession,
    ) -> str:
        """Write feature data to the databricks Feature Store. If the table already exists, the data will be upserted. If not, then a table will be created in the Feature Store.

        Args:
            spark(SparkSession): The SparkSession used for writing to the FS

        Returns:
            str: Message on succesful writing of data to UC
        """
        if self.primary_key not in self.feature_data.columns:
            raise ValueError(f"Primary key column '{self.primary_key}' not found in feature_data")

        feature_table_name = f"{self.catalog}.{self.schema}.{self.use_case_name}_{self.features_type}"
        fe = FeatureEngineeringClient()

        try:
            fe = FeatureEngineeringClient()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FeatureEngineeringClient: {str(e)}") from e

        try:
            if spark.catalog.tableExists(feature_table_name):
                fe.write_table(
                    name=feature_table_name,
                    df=self.feature_data,
                    mode="merge",
                )
                None if self.check_table_CDF_config(feature_table_name, spark) else self.enable_change_data_feed(
                    feature_table_name, spark
                )
                return f"The feature data has been succesfully upserted into {feature_table_name}"
            else:
                fe.create_table(
                    name=feature_table_name,
                    df=self.feature_data,
                    primary_keys=self.primary_key,
                    description="Hotel reservation feature data",
                )
                self.enable_change_data_feed(feature_table_name, spark)
                return f"Table {feature_table_name} has been created in the Feature Store successfully."
        except Exception as e:
            raise RuntimeError(f"Failed to write to Feature Store: {str(e)}") from e

    def enable_change_data_feed(self, feature_table_name: str, spark: SparkSession) -> str:
        try:
            spark.sql(f"""
                ALTER TABLE {feature_table_name}
                SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
            """)
            return f"Change data feed has been enabled for {feature_table_name}."
        except Exception as e:
            raise RuntimeError(f"Failed to enable change data feed for {feature_table_name}: {str(e)}") from e

    def check_table_CDF_config(self, feature_table_name: str, spark: SparkSession) -> bool:
        try:
            table_CDF = (
                spark.sql(f"DESCRIBE DETAIL delta.`{feature_table_name}`")
                .select("properties")
                .collect()[0]["properties"]
                .get("delta.enableChangeDataFeed", "false")
                .lower()
                == "true"
            )
            return table_CDF
        except Exception as e:
            raise RuntimeError(f"Failed to enable change data feed for {feature_table_name}: {str(e)}") from e
