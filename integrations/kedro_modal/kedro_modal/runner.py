from abc import abstractmethod
from typing import Optional
from kedro.runner import AbstractRunner, ParallelRunner
from kedro.pipeline import Pipeline
from kedro.io import DataCatalog, AbstractDataSet
from pluggy import PluginManager


class ModalRunner(ParallelRunner):
    @abstractmethod
    def _run(
        self,
        pipeline: Pipeline,
        catalog: DataCatalog,
        hook_manager: PluginManager,
        session_id: Optional[str] = None,
    ) -> None:
        pass

    @abstractmethod
    def create_default_data_set(self, ds_name: str) -> AbstractDataSet:
        pass
