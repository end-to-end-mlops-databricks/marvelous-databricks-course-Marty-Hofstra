"""This script evaluates the newest experiment run with the run related to the current model version based on the MAE,
and if the newest model version is better it is registered. If only one experiment run is available, due to the model
being registered for the first time in the training task, no new model version will be registered.
Namely, that would just create a second model version that is exactly the same."""

import re

import mlflow

from hotel_reservations.utils import check_repo_info, open_config


def evaluate_model_task():
    config = open_config("../../../../project_config.yaml", scope="marty-MLOPs-cohort")

    git_sha = dbutils.jobs.taskValues.get(taskKey="train_model", key="git_sha", debugValue="")  # type: ignore # noqa: F821
    job_run_id = dbutils.jobs.taskValues.get(taskKey="train_model", key="job_run_id", debugValue="")  # type: ignore # noqa: F821
    model_uri = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_uri", debugValue="")  # type: ignore # noqa: F821

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    git_branch, git_sha = check_repo_info(
        f"/Workspace/{config.user_dir_path}/{config.git_repo}",
        dbutils,  # type: ignore # noqa: F821
    )

    mlflow_client = mlflow.tracking.MlflowClient()
    current_model_run_id = mlflow_client.search_model_versions(
        f"name='{config.catalog}.{config.db_schema}.{config.use_case_name}_model_basic'"
    )[0].run_id
    new_run_id = re.search(r"runs:/([^/]+)/", model_uri).group(1)

    current_model_mae = mlflow.search_runs(
        experiment_names=[f"/{config.user_dir_path}/{config.use_case_name}"],
        filter_string=f"run_id='{current_model_run_id}'",
    )["metrics.mae"][0]

    new_run_mae = mlflow.search_runs(
        experiment_names=[f"/{config.user_dir_path}/{config.use_case_name}"], filter_string=f"run_id='{new_run_id}'"
    )["metrics.mae"][0]

    if new_run_mae < current_model_mae:
        print("New model is better based on MAE")
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=f"{config.catalog}.{config.db_schema}.{config.use_case_name}_model_basic",
            tags={"git_sha": git_sha, "branch": git_branch, "job_run_id": job_run_id},
        )

        print("New model registered with version:", model_version.version)
        dbutils.jobs.taskValues.set(key="model_version", value=model_version.version)  # type: ignore # noqa: F821
        dbutils.jobs.taskValues.set(key="model_update", value=1)  # type: ignore # noqa: F821
    else:
        print("The model has not improved based on the MAE")
        dbutils.jobs.taskValues.set(key="model_update", value=0)  # type: ignore # noqa: F821


if __name__ == "__main__":
    evaluate_model_task()
