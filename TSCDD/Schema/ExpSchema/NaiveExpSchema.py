from ...Schema.ExpSchema.BaseExpSchema import BaseExpSchema
from ...Config.Options.ExpSchemaOptions import ExpSchemaOptions
from ...Methods import BaseMethod
from ...Methods.BaseMethods import BaseOfflineMethod
from ...DataFactory.DataStore import DataStoreViewIndex
from ...Methods import MethodFactory
from ...Methods.OnlineMethodWrapper import OnlineMethodWrapper
from ...Schema.TestSchema import TestSchemaFactory, BaseTestSchema
from ...Schema.TestSchema import TestResults
from ...Config.Options import TestSchemaOptions
import copy
from ...utils import Progress


class NaiveExpSchema(BaseExpSchema):
    @classmethod
    def option_mark(cls) -> ExpSchemaOptions:
        return ExpSchemaOptions.Naive

    def run(self):
        method_name = self.method_name
        train_ratio = self.cfg.train_ratio
        test_ratio = self.cfg.test_ratio
        test_schema_option = self.cfg.test_schema_config.option

        curve_keys = self.data_store.get_dataset_curve_keys()
        progress = Progress(curve_keys, None)
        for curve_key in progress:
            progress.info(f"Exp {curve_key}", keyword="NaiveExpSchema")
            ts_datastore_view = self.data_store.get_single_view(curve_key)
            train_set_view, _ = ts_datastore_view.split_by_fractions([train_ratio, 1.0-train_ratio])
            _, test_set_view = ts_datastore_view.split_by_fractions([1.0 - test_ratio, test_ratio])
            hparams = self.cfg.method_hparams.get_hparams(method_name, self.hparams, curve_key, cascading=True)
            model: BaseMethod = MethodFactory.construct_by_name(method_name, hparams)

            if test_schema_option is TestSchemaOptions.Online and isinstance(model, BaseOfflineMethod):
                model = OnlineMethodWrapper(model, self.cfg.test_schema_config.offline_test_period,
                                            self.cfg.test_schema_config.offline_test_size)
            test_schema: BaseTestSchema = TestSchemaFactory.construct_by_option(test_schema_option, self.data_store,
                                                                                test_set_view, model,
                                                                                self.cfg.test_schema_config)
            test_schema.set_schema_progress(progress)
            model.train_valid(train_set_view)
            test_results: TestResults = test_schema.run_test()

            self.exp_results.set(test_set_view, test_results)














