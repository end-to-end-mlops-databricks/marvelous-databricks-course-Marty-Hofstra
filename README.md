<h1 align="center">
Hotel reservations package by Marty-Hofstra

## Description
This repo contains functionality for modelling the hotel reservations dataset, this can be [found here](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset)

## Setup
### Secrets
The Databricks user directory and volume path where the .whl is stored are not assigned in the config yaml in order to prevent them from being exposed. by following the steps in the `create_databricks_secrets.py` you can add these to the Databricks secrets. This has to be done before you can open the config.

### Virtual environment
In order to set your venv, run `make init` in the terminal. UV is used as Python package installer and resolver, it can be installed by running `brew install uv`. For local testing, the optional test dependencies have to be installed by running `make init extra=test`, this is required for running PySpark locally. In order to use DBConnect, the optional dev dependencies have to be installed by running `make init extra=dev`

Note: It might occur that after `make init extra=test`, the PySpark version patched from DBconnect is still used, this makes the unit tests fail because it requires a remote Spark session. If this happens, run `pip uninstall pyspark` and `pip uninstall pyspark==3.5.0`.

### Databricks
Install the Databricks extension in Visual Studio Code and follow the steps visible in the extension. Run `databricks auth login - configure-cluster - host <workspace-url>` in the terminal, this should create a `databrickscfg` file that looks as follows:
```
[DEFAULT]
host      = <WORKSPACE_URL>
auth_type = databricks-cli
cluster_id = <CLUSTER_ID>
```
Note: the DBR has to be 15.4 or higher, otherwise it will conflict with the `databricks-connect version`.

### Package
#### Creation
The `hotel_reservations` package (as a .whl) can be created and stored in DBFS by running `make build_and_store_whl dbfs_path=${dbfs_path}`, where `${dbfs_path}` is the path to the volume in which you want to store the whl.

#### Usage
In the cluster configuration, click on `Libraries` and then `Install new`, select `Volumes` and navigate to the path where you stored the wheel. The package functionality can be imported with `import hotel_reservations`.

*Data processing*
An example of usage of the data processing functions is as follows:

```
from pyspark.sql import SparkSession
from hotel_reservations.data_processing.data_processor import DataProcessor
from hotel_reservations.utils import open_config

spark = SparkSession.builder.getOrCreate()

# Load config from a relative path and add your Databricks secret scope here
config = open_config("project_config.yaml", scope="databricks_secret_scope")

data_preprocessor = DataProcessor(config, spark)

data_preprocessor.preprocess_data()

train, test = data_preprocessor.split_data()

X_features = list(set(config.cat_features) | set(config.num_features))

X_train = train.select(X_features)
X_test = test.select(X_features)
Y_train = train.select(config.target)
Y_test = test.select(config.target)

display(X_train)
```


*Featurisation*
This example shows how to register a table as Feature table.
```
from pyspark.sql import SparkSession

from hotel_reservations.featurisation.featurisation import Featurisation
from hotel_reservations.utils import open_config

spark = SparkSession.builder.getOrCreate()

# Load config from a relative path and add your Databricks secret scope here
config = open_config("project_config.yaml", scope="databricks_secret_scope")

input_data = spark.read.table(f"{config.catalog}.{config.db_schema}.{config.use_case_name}")

# Initiate the Featurisation instance with the config, feature data, feature type (for table naming) and the primary key
featurisation_instance = Featurisation(config, input_data, "input_features", "primary_key")

# Register the table as a table in teh Databricks Feature Store
featurisation_instance.write_feature_table(spark)
```

*Feature serving*
An example on how to use Feature Serving
```
from hotel_reservations.featurisation.featurisation import Featurisation
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Load config from a relative path and add your Databricks secret scope here
config = open_config("project_config.yaml", scope="databricks_secret_scope")

prediction_df = spark.read.table(f"{config.catalog}.{config.db_schema}.{config.use_case_name}_predictions")

token = (
        dbutils.notebook.entry_point.getDbutils()  # type: ignore # noqa: F821
        .notebook()
        .getContext()
        .apiToken()
        .get()
)

host = spark.conf.get("spark.databricks.workspaceUrl")

serving_instance = Serving("serving_endpoint_name", 10, host, token, "primary_key")

# The serving endpoint is created with a Feature Spec
serving_instance.create_serving_endpoint("feature_spec_name")

pk_list = predictions_df.select("primary_key").rdd.flatMap(lambda x: x).collect()

# Send a request for a random pk value
response_status, response_text, latency = serving_instance.send_request(random.choice(pk_list))

print("Response status:", response_status)
print("Reponse text:", response_text)
print("Execution time:", latency, "seconds")

# Bunch of requests
total_execution_time, average_latency = serving_instance.execute_and_profile_requests(pk_list)

print("\nTotal execution time:", total_execution_time, "seconds")
print("Average latency per request:", average_latency, "seconds")
```
