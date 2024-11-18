"""This script evaluates the newest experiment run with the run related to the current model version based on the MAE, and if the newest model version is better it is registered."""

import mlflow

from hotel_reservations.utils import check_repo_info, open_config


def evaluate_model_task():
    config = open_config("../../../../project_config.yaml", scope="marty-MLOPs-cohort")

    git_sha = dbutils.jobs.taskValues.get(taskKey="train_model", key="git_sha", debugValue="")  # type: ignore # noqa: F821
    job_run_id = dbutils.jobs.taskValues.get(taskKey="train_model", key="job_run_id", debugValue="")  # type: ignore # noqa: F821
    model_uri = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_uri", debugValue="")  # type: ignore # noqa: F821

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")

    git_branch, git_sha = check_repo_info(
        f"/Workspace/{config.user_dir_path}/{config.git_repo}",
        dbutils,  # type: ignore # noqa: F821
    )

    try:
        previous_run_mae = mlflow.search_runs(
            experiment_names=[f"/{config.user_dir_path}/{config.use_case_name}"],
            filter_string=f"tags.branch='{git_branch}'",
        )["metrics.mae"][1]
    except IndexError:
        previous_run_mae = float("inf")

    current_run_mae = mlflow.search_runs(
        experiment_names=[f"/{config.user_dir_path}/{config.use_case_name}"],
        filter_string=f"tags.branch='{git_branch}'",
    )["metrics.mae"][0]

    if current_run_mae < previous_run_mae:
        print("New model is better based on MAE.")
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=f"{config.catalog}.{config.db_schema}.{config.use_case_name}_model_basic",
            tags={"git_sha": git_sha, "branch": git_branch, "job_run_id": job_run_id},
        )

        print("New model registered with version:", model_version.version)
        dbutils.jobs.taskValues.set(key="model_version", value=model_version.version)  # type: ignore # noqa: F821
        dbutils.jobs.taskValues.set(key="model_update", value=1)  # type: ignore # noqa: F821
    else:
        print("Old model is better based on MAE.")
        dbutils.jobs.taskValues.set(key="model_update", value=0)  # type: ignore # noqa: F821


if __name__ == "__main__":
    evaluate_model_task()
