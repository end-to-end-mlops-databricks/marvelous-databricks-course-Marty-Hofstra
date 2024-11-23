# Databricks notebook source

# COMMAND ----------
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql
from hotel_reservations.utils import open_config
import time

w = WorkspaceClient()

# COMMAND ----------
config = open_config("../project_config.yaml", scope="marty-MLOPs-cohort")

# COMMAND ----------
alert_query = """
/* This query detects prediction drift by:
 * 1. Calculating the percentage difference between current and previous average predictions
 * 2. Comparing against a configured threshold
 * 3. Returns 1 if drift detected, 0 otherwise
 */
SELECT FIRST(CASE WHEN percentage > 50.0 THEN 1 ELSE 0 END) AS prediction_drift FROM(
SELECT avg, lead(avg, 1) OVER(ORDER BY window DESC) AS lead_avg_prediction, ROUND((avg - lead(avg, 1) OVER(ORDER BY window DESC))* 100.0 / lead(avg, 1) OVER(ORDER BY window DESC), 1) AS percentage, window from {config.catalog}.{config.schema}.{config.use_case_name}_preds_profile_metrics
where column_name = 'prediction'
ORDER BY window DESC
)
"""

# COMMAND ----------
srcs = w.data_sources.list()

# COMMAND ----------
query = w.queries.create(
    query=sql.CreateQueryRequestQuery(
        display_name=f"hotel-reservations-alert-query-{time.time_ns()}",
        warehouse_id=srcs[0].warehouse_id,
        description="Alert on hotel reservation predictions",
        query_text=alert_query,
    )
)

# COMMAND ----------
alert = w.alerts.create(
    alert=sql.CreateAlertRequestAlert(
        condition=sql.AlertCondition(
            operand=sql.AlertConditionOperand(column=sql.AlertOperandColumn(name="prediction_drift")),
            op=sql.AlertOperator.EQUAL,
            threshold=sql.AlertConditionThreshold(value=sql.AlertOperandValue(double_value=1)),
        ),
        display_name=f"hotel-reservations-alert-query-{time.time_ns()}",
        query_id=query.id,
    )
)