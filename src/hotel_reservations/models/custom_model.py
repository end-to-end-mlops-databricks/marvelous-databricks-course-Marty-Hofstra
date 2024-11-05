import mlflow
from mlflow.pyfunc import load_model
from pyspark.sql import DataFrame, SparkSession

from hotel_reservations.utils import adjust_predictions


class HotelReservationsModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_path = context.artifacts["my_model_path"]
        self.model = load_model(model_path)

    def predict(self, context, model_input: DataFrame, spark: SparkSession):
        if isinstance(model_input, DataFrame):
            columns = list(model_input.columns)
            predict_udf = mlflow.pyfunc.spark_udf(spark, self.model)
            predictions = model_input.withColumn("prediction", predict_udf(*columns))
            predictions_custom_model = {"Prediction": adjust_predictions(predictions)}
            return predictions_custom_model
        else:
            raise ValueError("Input must be a Spark DataFrame.")
