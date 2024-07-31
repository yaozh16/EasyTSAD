from TSCDD.DataFactory.DataStore import DataStore, DataStoreViewIndex
from TSCDD.Methods.BaseMethods import BaseMethod, BaseOnlineMethod, BaseOfflineMethod
from .BaseTestSchema import BaseTestSchema, BaseTestSchemaMeta
from TSCDD.Config.Options import TestSchemaOptions
from TSCDD.Config import TestSchemaConfig
from .OnlineTestSchema import OnlineTestSchema
from .OfflineTestSchema import OfflineTestSchema
from typing import Dict


class TestSchemaFactory:

    @classmethod
    def construct_by_name(cls, schema_name: str, data_store: DataStore, test_view: DataStoreViewIndex,
                            method: BaseMethod, test_schema_config: TestSchemaConfig) -> BaseTestSchema:
        schema_cls = BaseTestSchemaMeta.registry.get(schema_name, default=None)
        if schema_cls is not None:
            return schema_cls(data_store, test_view, method, test_schema_config)
        else:
            raise KeyError(f"TestSchema {schema_name} not registered")

    @classmethod
    def construct_by_option(cls, option: TestSchemaOptions, data_store: DataStore, test_view: DataStoreViewIndex,
                            method: BaseMethod, test_schema_config: TestSchemaConfig) -> BaseTestSchema:
        if option is TestSchemaOptions.Online:
            schema_cls = OnlineTestSchema
        elif option is TestSchemaOptions.Offline:
            schema_cls = OfflineTestSchema
        else:
            if isinstance(method, BaseOnlineMethod):
                schema_cls = OnlineTestSchema
            elif isinstance(method, BaseOfflineMethod):
                schema_cls = OfflineTestSchema
            else:
                raise TypeError(f"Unknown option {option} to construct test schema (Method [{method}])")

        if schema_cls is not None:
            return schema_cls(data_store, test_view, method, test_schema_config)
        else:
            raise KeyError(f"TestSchema option {option} not registered")

