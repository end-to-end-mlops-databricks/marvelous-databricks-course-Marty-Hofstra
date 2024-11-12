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
In the cluster configuration, click on `Libraries` and then `Install new`, select `Volumes` and navigate to the path where you stored the wheel. The package functionality can be imported with `import hotel_reservations`. An example of usage of the data processing functions is as follows:

```
from pyspark.sql import SparkSession
from hotel_reservations.data_processing.data_processor import DataProcessor
from hotel_reservations.utils import open_config

config = open_config("../../../../project_config.yaml", scope="marty-MLOPs-cohort")

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
