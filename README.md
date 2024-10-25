<h1 align="center">
Marvelous MLOps End-to-end MLOps with Databricks Marty-Hofstra repo

## Description
This repo contains functionality for modelling the hotel reservations dataset, this can be [found here](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset)
## Setup
### Virtual environment
In order to set your venv, run `make init` in the terminal. UV is used as Python package installer and resolver, it can be installed by running `brew install uv`.

### Databricks
Install the Databricks extension in Visual Studio Code and follow the steps visible in the extension. Make sure to add your cluster_id to the `.databrickscfg` file.
Note: the DBR has to be 15.4 or higher, otherwise it will conflict with the `databricks-connect version`.

### Package
The `housing_prices` package (as a .whl) can be created and stored in DBFS by running `make build_and_store_whl dbfs_path=${dbfs_path}`, where `${dbfs_path}` is the path to the volume in which you want to store the whl.