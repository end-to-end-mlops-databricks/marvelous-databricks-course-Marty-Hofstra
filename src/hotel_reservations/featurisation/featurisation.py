from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import DataFrame, SparkSession


def write_feature_table(
    feature_data: DataFrame,
    catalog: str,
    schema: str,
    use_case_name: str,
    features_type: str,
    primary_key: str,
    spark: SparkSession,
) -> str:
    """Write feature data to the databricks Feature Store. If the table already exists, the data will be upserted. If not, then a table will be created in the Feature Store.

    Args:
        feature_data (DataFrame): Dataframe containing feature data to write to the Feature Store
        catalog (str): Catalog in which to write the feature data
        schema (str): Schema/database in which to write the feature data
        use_case_name (str): Name of the use case
        features_type (str): Type of features for clear naming of the feature table, e.g. 'predictions_features'
        primary_key (str): Name of the column to use as PK in the Feature table
        spark(SparkSession): The SparkSession used for writing to the FS

    Returns:
        str: Message on succesful writing of data to UC
    """
    if primary_key not in feature_data.columns:
        raise ValueError(f"Primary key column '{primary_key}' not found in feature_data")

    feature_table_name = f"{catalog}.{schema}.{use_case_name}_{features_type}"
    fe = FeatureEngineeringClient()

    if spark.catalog.tableExists(feature_table_name):
        fe.write_table(
            name=feature_table_name,
            df=feature_data,
            mode="merge",
        )
        return f"The feature data has been succesfully upserted into {feature_table_name}"
    else:
        fe.create_table(
            name=feature_table_name,
            df=feature_data,
            primary_keys=primary_key,
            description="Hotel reservation feature data",
        )
        return f"Table {feature_table_name} has been created in the Feature Store successfully."
