import logging

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorSnapshot

from hotel_reservations.types.project_config_types import ProjectConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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

    def create_lakehouse_monitor(self) -> None:
        """Creates a Lakehouse data monitor if it does not exists yet. A snapshot type is chosen here because the predictions for the hotel-reservations are done batch wise.

        Returns:
            None

        Raises:
            ValueError: If assets directory path is invalid
            RuntimeError: If monitor creation fails
        """
        assets_dir = f"/Workspace/{self.config.user_dir_path}/lakehouse_monitoring/{self.table_name}"
        if not assets_dir.startswith("/Workspace/"):
            raise ValueError("assets_dir must be under /Workspace/")

        try:
            self.workspace.quality_monitors.get(self.table_name)
            logger.info(f"Lakehouse monitor {self.table_name} already exists. Skipping creation.")
            return
        except Exception as e:
            logger.info(f"Monitor not found: {e}. Creating new monitor for {self.table_name}")
        try:
            self.workspace.quality_monitors.create(
                table_name=self.table_name,
                assets_dir=assets_dir,
                output_schema_name=f"{self.config.catalog}.{self.config.db_schema}",
                snapshot=MonitorSnapshot(),
            )
            logger.info(f"Successfully created monitor for {self.table_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to create monitor: {e}") from e

    def refresh_monitor(self) -> None:
        """Refreshes the monitor, this is done after new predictions have been written to UC.

        Returns:
            None

        Raises:
            RuntimeError: If monitor refresh fails
        """
        try:
            logger.info(f"Starting monitor refresh for {self.table_name}")
            self.workspace.quality_monitors.run_refresh(table_name=self.table_name)
            logger.info(f"Successfully refreshed monitor for {self.table_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to refresh monitor: {e}") from e
