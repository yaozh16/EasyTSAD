import logging
from logging import Logger
from TSCDD.Config.PathManager import PathManager
from TSCDD.DataFactory.DataStore import DataStore, DataStoreViewIndex
from TSCDD.Methods import BaseMethod
from abc import ABCMeta, abstractmethod
from ...Config.Options import TestSchemaOptions
from ...Config import TestSchemaConfig
from .TestResults import TestResults
from ...utils import Progress
from typing import Union


class BaseTestSchemaMeta(ABCMeta):
    """
    Attributes:
        registry (dict): Registry to store the registered schemas.
    """
    registry = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if name != 'BaseTestSchema':
            BaseTestSchemaMeta.registry[name] = cls


class BaseTestSchema(metaclass=BaseTestSchemaMeta):

    def __init__(self, data_store: DataStore, test_view: DataStoreViewIndex, method: BaseMethod,
                 test_schema_config: TestSchemaConfig):
        """
        Initializes an instance of the BaseSchema class.

        Args:
            - `data_store` (DataStore): Data reference.
            - `test_view` (DataStoreViewIndex): Data view.
            - `method` (BaseMethods): The model object.
            - `cfg_path` (str, optional): Path to a custom configuration file. Defaults to None.
        """
        self.logger: Union[Logger, Progress] = logging.getLogger("logger")
        self.data_store = data_store
        self.test_view = test_view
        self.method = method
        self.pm = PathManager.get_instance()
        self.config = test_schema_config

    def set_schema_progress(self, progress: Progress):
        self.logger = progress

    @classmethod
    @abstractmethod
    def option_mark(cls) -> TestSchemaOptions:
        raise NotImplementedError("Schema option_mark unimplemented")

    @abstractmethod
    def run_test(self) -> TestResults:
        """
        Performs the experiment test.
        """
        raise NotImplementedError("Schema method run_test unimplemented")









