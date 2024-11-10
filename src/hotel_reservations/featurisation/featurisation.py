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
        self.feature_table_name: str = f"{config.catalog}.{config.db_schema}.{config.use_case_name}_{features_type}"
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

        fe = FeatureEngineeringClient()

        try:
            fe = FeatureEngineeringClient()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FeatureEngineeringClient: {str(e)}") from e

        try:
            if spark.catalog.tableExists(self.feature_table_name):
                fe.write_table(
                    name=self.feature_table_name,
                    df=self.feature_data,
                    mode="merge",
                )
                None if self.check_table_CDF_property(spark) else self.enable_change_data_feed(spark)
                return f"The feature data has been succesfully upserted into {self.feature_table_name}"
            else:
                fe.create_table(
                    name=self.feature_table_name,
                    df=self.feature_data,
                    primary_keys=self.primary_key,
                    description="Hotel reservation feature data",
                )
                self.enable_change_data_feed(spark)
                return f"Table {self.feature_table_name} has been created in the Feature Store successfully."
        except Exception as e:
            raise RuntimeError(f"Failed to write to Feature Store: {str(e)}") from e

    def enable_change_data_feed(self, spark: SparkSession) -> str:
        """Enable the change data feed property on the feature table.

        Args:
            spark (SparkSession): The spark session is required for running Spark functionality outside of Databricks.

        Raises:
            RuntimeError: If the table cannot be found or if it is not a Delta table

        Returns:
            str: Message on succesfull enabling of the change datat feed property on the feature table
        """
        try:
            spark.sql(f"""
                ALTER TABLE {self.feature_table_name}
                SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
            """)
            return f"Change data feed has been enabled for {self.feature_table_name}."
        except Exception as e:
            raise RuntimeError(f"Failed to enable change data feed for {self.feature_table_name}: {str(e)}") from e

    def check_table_CDF_property(self, spark: SparkSession) -> bool:
        """Checks if the change data feed (CDF) property is already enable on the table

        Args:
            spark (SparkSession): The spark session is required for running Spark functionality outside of Databricks.

        Raises:
            RuntimeError: If the table cannot be found or if it is not a Delta table

        Returns:
            bool: True when the change data feed property is already enabled
        """
        try:
            table_CDF = (
                spark.sql(f"DESCRIBE DETAIL {self.feature_table_name}")
                .select("properties")
                .collect()[0]["properties"]
                .get("delta.enableChangeDataFeed", "false")
                .lower()
                == "true"
            )
            return table_CDF
        except Exception as e:
            raise RuntimeError(f"Failed to enable change data feed for {self.feature_table_name}: {str(e)}") from e
