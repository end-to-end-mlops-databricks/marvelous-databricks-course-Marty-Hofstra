import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput


class Serving:
    def __init__(self, serving_endpoint_name: str, num_requests: int, host: str, token: str, primary_key: str) -> None:
        self.workspace = WorkspaceClient()
        self.serving_endpoint_name = serving_endpoint_name
        self.num_requests = num_requests
        self.host = host
        self.token = token
        self.primary_key = primary_key

    def create_serving_endpoint(self, feature_spec_name: str, workload_size: str = "Small"):
        try:
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
        except Exception as e:
            print(f"Failed to create serving endpoint '{self.serving_endpoint_name}': {e}")

    def send_request(self, pk_value: str) -> tuple[int, str, float]:
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

    def send_request_random_id(self, id_list: list[str]) -> tuple[int, str, float]:
        random_id = random.choice(id_list)

        response_status, response_text, latency = self.send_request(random_id)

        return response_status, response_text, latency

    def execute_and_profile_requests(self, id_list: list[str], max_workers: int = 100) -> tuple[float, float]:
        total_start_time = time.time()
        latencies: list[float] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.send_request_random_id, id_list) for _ in range(self.num_requests)]

            for future in as_completed(futures):
                latency = future.result()[2]
                latencies.append(latency)

        total_end_time = time.time()
        total_execution_time = total_end_time - total_start_time

        average_latency = sum(latencies) / len(latencies)

        return total_execution_time, average_latency
