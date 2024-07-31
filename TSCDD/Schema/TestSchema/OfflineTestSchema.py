from ...DataFactory.DataStore import DataStore, DataStoreViewIndex
from ...Methods import MethodTestResults
from ...Methods.BaseMethods import BaseOfflineMethod
from .BaseTestSchema import BaseTestSchema
from .TestResults import TestResults, TestResult
from TSCDD.Config.Options.TestSchemaOptions import TestSchemaOptions
from TSCDD.Config import TestSchemaConfig
from tqdm import tqdm
from datetime import datetime


class OfflineTestSchema(BaseTestSchema):
    @classmethod
    def option_mark(cls) -> TestSchemaOptions:
        return TestSchemaOptions.Offline

    def __init__(self, data_store: DataStore, test_view: DataStoreViewIndex, method: BaseOfflineMethod,
                 test_schema_config: TestSchemaConfig):
        super().__init__(data_store, test_view, method, test_schema_config)
        self.method = method
        assert isinstance(method, BaseOfflineMethod), f"Method [{method.__class__}] does not support offline test"

    def run_test(self) -> TestResults:
        test_results = TestResults()
        model: BaseOfflineMethod = self.method
        test_curves = self.data_store[self.test_view]

        desc = test_curves.items()
        for key, tsdata_view in desc:
            log_prefix = f"OfflineTest {key} : {tsdata_view.size()} step(s). "
            if not self.config.quiet:
                self.logger.info(log_prefix)
            test_result = TestResult()
            ts_view = tsdata_view.get_timeseries()
            model.offline_initialize()
            curve_test_result: MethodTestResults = model.offline_test(ts_view)
            test_result.set_result(curve_test_result)
            test_results.update({key: test_result})
        return test_results
