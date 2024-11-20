import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

logging.basicConfig(level=logging.ERROR)


class Serving:
    """A class to serve the predictions

    Attributes
    ----------
    serving_endpoint_name: str
        Name used for creation of the serving endpoint, e.g. assigning the value 'hotel-reservations-feature-serving' wil create the following endpoint: `f"https://{self.host}/serving-endpoints/hotel-reservations-feature-serving/invocations"`
    num_requests: int
        Number of requests that you require to sent to the endpoint
    host: str
        Name of the host for the serving endpoint, can be retrieved in Databricks by `host = spark.conf.get("spark.databricks.workspaceUrl")`
    token: str
        Token required for the serving endpoint, can be retrieved in Databricks by `token = (dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()`
    primary_key: str
        PK used for retrieving primary key values from the feature table

    Methods
    -------
    create_serving_endpoint:
        Reads from Unity Catalog as a Spark Dataframe, the naming of tables in Databricks consists of three levels: catalog, schema and table name.
    send_request:
        Sends a request to the endpoint with the primary key value as input
    send_request_random_id:
        Sends a request to the endpoint with a random pk value from the given list
    execute_and_profile_requests:
        Executes the requests and profiles them, specifically the total execution time and the average latency
    """

    def __init__(self, serving_endpoint_name: str, num_requests: int, host: str, token: str, primary_key: str) -> None:
        """Constructs all the necessary attributes for the serving object

        Args:
            serving_endpoint_name (str): Name used for creation of the serving endpoint, e.g. assigning the value 'hotel-reservations-feature-serving' wil create the following endpoint: `f"https://{self.host}/serving-endpoints/hotel-reservations-feature-serving/invocations"`
            num_requests (int): Number of requests that you require to sent to the endpoint
            host (str): Name of the host for the serving endpoint, can be retrieved in Databricks by `host = spark.conf.get("spark.databricks.workspaceUrl")`
            token (str): Token required for the serving endpoint, can be retrieved in Databricks by `token = (dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()`
            primary_key (str): PK used for retrieving primary key values from the feature table
        """
        self.workspace = WorkspaceClient()
        self.serving_endpoint_name = serving_endpoint_name
        self.num_requests = num_requests
        self.host = host
        self.token = token
        self.primary_key = primary_key

    def create_serving_endpoint(self, feature_spec_name: str, workload_size: str = "Small"):
        """Creates a serving endpoint in Databricks

        Args:
            feature_spec_name (str): Name of the FeatureSpec in Databricks, a FeatureSpec is a user-defined set of features and functions.
            workload_size (str, optional): The workload size of the served entity. The workload size corresponds to a range of provisioned
            concurrency that the compute autoscales between. A single unit of provisioned concurrency can
            process one request at a time. Valid workload sizes are "Small" (4 - 4 provisioned concurrency),
            "Medium" (8 - 16 provisioned concurrency), and "Large" (16 - 64 provisioned concurrency). If
            scale-to-zero is enabled, the lower bound of the provisioned concurrency for each workload size
            is 0.. Defaults to "Small".
        """
        try:
            self.workspace.serving_endpoints.get(self.serving_endpoint_name)
            logging.info(f"Serving endpoint {self.serving_endpoint_name} already exists. Skipping creation.")
        except Exception as e:
            print(f"{e} Creating the serving endpoint {self.serving_endpoint_name}")
            self.workspace.serving_endpoints.create(
                name=self.serving_endpoint_name,
                config=EndpointCoreConfigInput(
                    served_entities=[
                        ServedEntityInput(
                            entity_name=feature_spec_name,
                            scale_to_zero_enabled=True,
                            workload_size=workload_size,
                        )
                    ]
                ),
            )

    def send_request(self, pk_value: str) -> tuple[int, str, float]:
        """Sends a request to the endpoint with the primary key value as input

        Args:
            pk_value (str): Value of the primary key for which the prediction is requested

        Returns:
            reponse_status (int): Status code of the response
            reponse_text (str): The content of the response
            latency (float): Latency in seconds

        Raises:
            requests.exceptions.RequestException: If a request error occurs.
        """
        try:
            start_time = time.time()
            serving_endpoint = f"https://{self.host}/serving-endpoints/{self.serving_endpoint_name}/invocations"
            response = requests.post(
                f"{serving_endpoint}",
                headers={"Authorization": f"Bearer {self.token}"},
                json={"dataframe_records": [{self.primary_key: pk_value}]},
            )
            end_time = time.time()
            latency = end_time - start_time

            response_status = response.status_code
            response_text = response.text

            return response_status, response_text, latency
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to send an endpoint request: {str(e)}") from e

    def send_request_random_id(self, pk_list: list[str]) -> tuple[int, str, float]:
        """Sends a request to the endpoint with a random pk value from the given list

        Args:
            pk_list (list[str]): Value of the primary key for which the prediction is requested

        Returns:
            reponse_status (int): Status code of the response
            reponse_text (str): The content of the response
            latency (float): Latency in seconds

        Raises:
            ValueError: If `pk_list` is empty.
        """
        if not pk_list:
            raise ValueError("ID list cannot be empty")

        random_id = random.choice(pk_list)

        response_status, response_text, latency = self.send_request(random_id)

        return response_status, response_text, latency

    def execute_and_profile_requests(self, pk_list: list[str], max_workers: int = 100) -> tuple[float, float]:
        """Executes the requests and profiles them, specifically the total execution time and the average latency

        Args:
            pk_list (list[str]): Value of the primary key for which the prediction is requested
            max_workers (int, optional): The ThreadPoolExecutor uses a pool of at most max_workers to execute calls asynchronously. Defaults to 100.

        Returns:
            total_execution_time (float): Total time it takes in seconds to execute the calls
            average_latency (float): Average latency in seconds per call
        """
        total_start_time = time.time()
        latencies: list[float] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.send_request_random_id, pk_list) for _ in range(self.num_requests)]

            for future in as_completed(futures):
                try:
                    latency = future.result()[2]
                    latencies.append(latency)
                except Exception as e:
                    print(f"An error occurred during request execution: {e}")

        total_end_time = time.time()
        total_execution_time = total_end_time - total_start_time

        average_latency = sum(latencies) / len(latencies)

        return total_execution_time, average_latency
