import mlflow
from mlflow.pyfunc import load_model
from pyspark.sql import DataFrame

from hotel_reservations.utils import adjust_predictions


class HotelReservationsModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_path = context.artifacts["my_model_path"]
        self.model = load_model(model_path)

    def transform(self, model_input: DataFrame):
        if isinstance(model_input, DataFrame):
            columns = list(model_input.columns)
            predictions = model_input.withColumn("prediction", self.model(*columns))
            predictions = {"Prediction": adjust_predictions(predictions)}
            return predictions
        else:
            raise ValueError("Input must be a Spark DataFrame.")
