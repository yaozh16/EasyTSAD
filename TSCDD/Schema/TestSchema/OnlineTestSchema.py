import logging

from TSCDD.Config import TestSchemaConfig
from TSCDD.Config.Options.TestSchemaOptions import TestSchemaOptions
from TSCDD.Methods.BaseMethods import BaseOnlineMethod
from .BaseTestSchema import BaseTestSchema
from .TestResults import OnlineTestResult, TestResults
from ...DataFactory.DataStore import DataStore, DataStoreViewIndex
from ...DataFactory.TimeSeries.TimeSeries import TimeSeriesView
from tqdm import tqdm
from datetime import datetime


class OnlineTestSchema(BaseTestSchema):
    @classmethod
    def option_mark(cls) -> TestSchemaOptions:
        return TestSchemaOptions.Online

    def __init__(self, data_store: DataStore, test_view: DataStoreViewIndex, method: BaseOnlineMethod,
                 test_schema_config: TestSchemaConfig):
        super().__init__(data_store, test_view, method, test_schema_config)
        self.method = method
        assert isinstance(method, BaseOnlineMethod), f"Method [{method.__class__}] does not support online test"

    def run_test(self) -> TestResults:
        test_results = TestResults()
        retrain_window_size = self.config.online_retrain_window_size
        online_sliding_window_size = self.config.online_sliding_window_size
        online_step_size = self.config.online_step_size
        model: BaseOnlineMethod = self.method
        test_curves = self.data_store[self.test_view]
        desc = test_curves.items()
        for key, tsdata_view in desc:
            test_result = OnlineTestResult()
            ts_view: TimeSeriesView = tsdata_view.get_timeseries()
            model.online_initialize()
            sliding_window: TimeSeriesView = ts_view.head(online_sliding_window_size)
            for i in range(ts_view.size()):
                if model.need_retrain():
                    test_result.record_retrain()
                    if not self.config.quiet:
                        self.logger.info(f" Retrain {test_result.get_retrain_count()} times in total")
                    retrain_timeseries: TimeSeriesView = ts_view.safe_slice(i, i + retrain_window_size)
                    model.online_retrain(retrain_timeseries)
                model.online_step(sliding_window)
                sliding_window.step(online_step_size)
            curve_test_result = model.online_collect_results()
            test_result.set_result(curve_test_result)
            test_results.update({key: test_result})
        return test_results
