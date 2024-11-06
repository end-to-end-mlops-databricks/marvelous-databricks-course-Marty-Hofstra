import mlflow
from pyspark.sql import DataFrame

from hotel_reservations.utils import adjust_predictions


class HotelReservationsModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def load_context(self, context):
        """Load model from MLflow context."""
        self.model = mlflow.pyfunc.spark_udf(context.artifacts["model_path"])

    def predict(self, context, model_input: DataFrame):
        if isinstance(model_input, DataFrame):
            columns = list(model_input.columns)
            predictions = model_input.withColumn("prediction", self.model(*columns))
            predictions_custom_model = adjust_predictions(predictions)
            return predictions_custom_model
        else:
            raise ValueError("Input must be a Spark DataFrame.")
