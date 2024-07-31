from abc import ABCMeta, abstractmethod
from TSCDD.Config.Options.ExpSchemaOptions import ExpSchemaOptions
from ...DataFactory.DataStore import DataStore, DataStoreViewIndex
from typing import Union
from ...Config import ExpSchemaConfig
from ..TestSchema import TestResults


class ExpResult:
    def __init__(self, ds_view: DataStoreViewIndex, test_results: TestResults):
        self.ds_view = ds_view
        self.test_results = test_results

    def to_json(self) -> dict:
        return {
            "ds_view": self.ds_view.to_json(),
            "test_results": self.test_results.to_json()
        }

    @staticmethod
    def from_dict(result_json) -> "ExpResult":
        ds_view: DataStoreViewIndex = DataStoreViewIndex.from_json(result_json["ds_view"])
        test_results: TestResults = TestResults.from_json(result_json["test_results"])
        return ExpResult(ds_view, test_results)


class ExpResults:
    def __init__(self):
        self.results: list[ExpResult] = list()

    def set(self, ts_view: DataStoreViewIndex, test_results: TestResults):
        self.results.append(ExpResult(ts_view, test_results))

    def to_json(self):
        return [e.to_json() for e in self.results]

    @staticmethod
    def from_dict(results_json: list) -> "ExpResults":
        results = ExpResults()
        for result in results_json:
            results.results.append(ExpResult.from_dict(result))
        return results


class BaseExpSchemaMeta(ABCMeta):
    """
    Attributes:
        registry (dict): Registry to store the registered schemas.
    """
    registry = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if name != 'BaseExpSchema':
            BaseExpSchemaMeta.registry[name] = cls


class BaseExpSchema(metaclass=BaseExpSchemaMeta):
    def __init__(self, data_store: DataStore, method_name: str, cfg: ExpSchemaConfig, hparams: dict = None):
        self.data_store = data_store
        self.cfg: ExpSchemaConfig = cfg
        self.method_name = method_name
        self.hparams = hparams if isinstance(hparams, dict) else {}
        self.exp_results = ExpResults()

    @classmethod
    @abstractmethod
    def option_mark(cls) -> ExpSchemaOptions:
        raise NotImplementedError("ExpSchema option_mark is not implemented.")

    @abstractmethod
    def run(self):
        raise NotImplementedError("ExpSchema run is not implemented.")

    def get_exp_results(self) -> ExpResults:
        return self.exp_results


class ExpSchemaFactory:
    @classmethod
    def exp_schema_names(cls):
        return BaseExpSchemaMeta.registry.keys()

    @classmethod
    def exp_schema_options(cls):
        return set([e.option_mark() for e in BaseExpSchemaMeta.registry.values()])

    @classmethod
    def fetch_by_name(cls, key: Union[str, list[str]], default=None) -> Union[None, BaseExpSchema, list[BaseExpSchema]]:
        if isinstance(key, str):
            return BaseExpSchemaMeta.registry.get(key, default)
        elif isinstance(key, list):
            return [BaseExpSchemaMeta.registry.get(e, default) for e in key]

    @classmethod
    def fetch_by_option(cls, exp_schema_option: ExpSchemaOptions, default=None) -> Union[None, BaseExpSchema]:
        for e in BaseExpSchemaMeta.registry.values():
            if e.option_mark() == exp_schema_option:
                return e
        return default

    @classmethod
    def construct_instance(cls, schema_cls, data_store: DataStore, method_name: str, cfg: ExpSchemaConfig,
                           hparams: dict) -> BaseExpSchema:
        return schema_cls(data_store, method_name, cfg, hparams)
