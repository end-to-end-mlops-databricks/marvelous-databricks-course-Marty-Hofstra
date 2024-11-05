from pyspark.sql import SparkSession

from hotel_reservations.utils import open_config, write_feature_table


def featurisation():
    spark = SparkSession.builder.getOrCreate()
    config = open_config("../../../../project_config.yaml").dict()

    hotel_reservation_data = spark.read.table(f"{config['catalog']}.{config['db_schema']}.{config['table_name']}")

    write_feature_table(hotel_reservation_data, config["catalog"], config["db_schema"], config["table_name"], spark)


if __name__ == "__main__":
    featurisation()
