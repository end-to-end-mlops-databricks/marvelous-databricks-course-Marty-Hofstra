# Databricks notebook source

# COMMAND ----------
# MAGIC %pip install databricks_cli

# COMMAND ----------
# imports
from databricks_cli.sdk.api_client import ApiClient
from databricks_cli.secrets.api import SecretApi

# get databricks API url and login
api_client = ApiClient(
    host=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None),  # type: ignore # noqa: F821
    token=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None),  # type: ignore # noqa: F821
)

# Create secret client
secret_api = SecretApi(api_client)
# more info on this client here: https://github.com/databricks/databricks-cli/blob/main/databricks_cli/secrets/api.py

# COMMAND ----------
# Create new secret scope
scope = "marty-MLOPs-cohort"  # PLEASE CHANGE THIS NAME to your own name. a scope contains all your secrets
if scope not in [dict_entry["name"] for dict_entry in secret_api.list_scopes()["scopes"]]:
    secret_api.create_scope(
        scope,
        initial_manage_principal=None,
        scope_backend_type="DATABRICKS",
        backend_azure_keyvault=None,
    )

secret_api.list_scopes()

# COMMAND ----------
# Check who has access to this scope
secret_api.list_acls(scope)

# COMMAND ----------
# check available secrets in scope
secret_api.list_secrets(scope=scope)

# COMMAND ----------
# Add a secret to the scope
your_secret = "blabla"
secret_api.put_secret(scope=scope, key="test", string_value=your_secret, bytes_value=None)
# secret_api.put_secret(scope=scope, key="blabla", string_value=your_secret, bytes_value=None)

# COMMAND ----------
# Read secrets from scope, printing secrets is redacted
print(dbutils.secrets.get(scope=scope, key="test"))  # type: ignore # noqa: F821
print(
    " ".join(x for x in dbutils.secrets.get(scope=scope, key="test"))  # type: ignore # noqa: F821
)  # you can still get it done in this way.
