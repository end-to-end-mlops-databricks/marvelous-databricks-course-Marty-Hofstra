import logging

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorSnapshot

from hotel_reservations.types.project_config_types import ProjectConfig

logging.basicConfig(level=logging.ERROR)


class Monitoring:
    def __init__(self, config: ProjectConfig, table_name: str) -> None:
        """Constructs all the necessary attributes for the serving object

        Args:
            config (ProjectConfig): Project configuration file converted to dict, containing the catalog and schema where the data resides. Moreover, it contains the model parameters, numerical features, categorical features and the target variables.
            table_name (str): The name of the table for which to create the lakehouse monitor.
        """
        self.config: ProjectConfig = config
        self.workspace = WorkspaceClient()
        self.table_name = table_name

    def create_lakehouse_monitor(self):
        """Creates a Lakehouse data monitor if it does not exists yet. A snapshot type is chosen here because the predictions for the hotel-reservations are done batch wise."""
        try:
            self.workspace.quality_monitors.get(self.table_name)
            logging.info(f"Lakehouse monitor {self.table_name} already exists. Skipping creation.")
        except Exception as e:
            print(f"{e} Creating the data monitor for {self.table_name}")
            self.workspace.quality_monitors.create(
                table_name=self.table_name,
                assets_dir=f"/Workspace/{self.config.user_dir_path}/lakehouse_monitoring/{self.table_name}",
                output_schema_name=f"{self.config.catalog}.{self.config.db_schema}",
                snapshot=MonitorSnapshot(),
            )

    def refresh_monitor(self):
        """Refreshes the monitor, this is done after new predictions have been written to UC."""
        self.workspace.quality_monitors.run_refresh(table_name=self.table_name)
