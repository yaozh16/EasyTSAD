import glob
import json
import os
import shutil
import time
from typing import Union, Tuple
import pandas as pd

from .logger import setup_logger
from ..Config import GlobalConfig, DataConfig
from ..Config.Options import VisOption
from ..Config.PathManager import PathManager
from ..DataFactory.DataStore import DataStore
from ..DataFactory.LabelStore import LabelType
from ..Evaluations.EvalInterface import EvalInterface
from ..Evaluations.Evaluator import Evaluator
from ..Methods.BaseMethods import BaseMethodMeta, BaseOnlineMethod
from ..Schema.ExpSchema import ExpSchemaFactory, BaseExpSchema
from ..Visualization import Visualizer


class Controller4CD:
    '''
    Controller4CD class represents a controller that manages global configuration and logging.

    Attributes:
    '''

    def __init__(self, cfg_path=None, log_path=None, log_level="info") -> None:
        """
        Controller4CD class represents a controller that manages global configuration and logging.

        Args:
            cfg_path (str, optional): Path to the configuration file. If provided, the configuration will be applied
            from this file. Defaults to None (Not Recommanded).
            log_path (str, optional): Path to the log file. If not provided, a default log file named "TSADEval.log"
            will be built in current workspace. Defaults to None.
            log_level (str, optional): Log level to set for the logger. Options: "debug", "info", "warning", "error".
            Defaults to "info".
        """

        self.logger = setup_logger(log_path, level=log_level)

        origin_file_path = os.path.abspath(__file__)
        origin_directory = os.path.dirname(origin_file_path)
        origin_cfg_path = os.path.join(origin_directory, "GlobalCfg.toml")

        self.cfg: GlobalConfig = GlobalConfig(origin_cfg_path)

        if cfg_path is not None:
            self.apply_cfg(cfg_path)

        PathManager.del_instance()
        self.pm = PathManager(self.cfg)

        self.ds: Union[DataStore, None] = None
        self.exp_schema: Union[BaseExpSchema, None] = None
        self.evaluator: Union[Evaluator, None] = None

    def load_dataset(self, custom_data_config: DataConfig = None):
        if custom_data_config is not None:
            self.ds = DataStore(self.cfg.data_config.copy(custom_data_config))
        else:
            self.ds = DataStore(self.cfg.data_config)
        self.ds.load()

    def run_exps(self, method: str, exp_config_path: str = None, hparams=None):
        """
        Run experiments using the specified method and training schema.

        Args:
            method (str): The method being used.
            exp_config_path (str, optional): Path to a custom experiment configuration file. Defaults to None.
            hparams (dict, optional): Hyperparameters for the model. Defaults to None.

        Returns:
            None
        """

        if self.ds is None:
            self.load_dataset()
        self.logger.info("Run Experiments. Method[{}].".format(method))

        if exp_config_path is not None:
            exp_schema_config = self.cfg.exp_schema_config.copy(exp_config_path)
        else:
            exp_schema_config = self.cfg.exp_schema_config
        exp_schema_cls = ExpSchemaFactory.fetch_by_option(self.cfg.exp_schema_config.option, None)
        if exp_schema_cls is None:
            raise NotImplementedError(f"{self.cfg.exp_schema_config.option} has no corresponding schema implementation")

        self.exp_schema = ExpSchemaFactory.construct_instance(exp_schema_cls, self.ds, method, exp_schema_config,
                                                              hparams)
        self.exp_schema.run()
        if self.cfg.exp_schema_config.save_test_results:
            self.logger.info(f"Save Experiment Test Outputs. Method[{method}] "
                             f"Schema[{self.cfg.exp_schema_config.option.name}]")
            for exp_result in self.exp_schema.get_exp_results().results:
                for curve_key, (start, end, dims) in exp_result.ds_view.get_items():
                    dataset_type, dataset_name, curve_name = curve_key
                    current_timestamp = str(int(time.time()))
                    output_path = self.pm.get_test_output_reference_view_path(method_name=method,
                                                                              schema_name=self.cfg.exp_schema_config.
                                                                              option.name,
                                                                              dataset_type=dataset_type,
                                                                              dataset_name=dataset_name,
                                                                              timestamp=current_timestamp,
                                                                              curve_name=curve_name)
                    with open(output_path, "w") as f:
                        f.write(json.dumps({"start": start, "end": end, "dims": dims}, indent=2))
                for curve_key, test_result in exp_result.test_results.items():
                    dataset_type, dataset_name, curve_name = curve_key
                    output_path = self.pm.get_test_output_result_path(method_name=method,
                                                                      schema_name=self.cfg.exp_schema_config.option.name,
                                                                      dataset_type=dataset_type,
                                                                      dataset_name=dataset_name,
                                                                      timestamp=current_timestamp,
                                                                      curve_name=curve_name)
                    with open(output_path, "w") as f:
                        f.write(json.dumps(test_result.to_json(), indent=2))

    def set_evals(self, evals):
        '''
        Registers the evaluation protocols used for performance evaluations.

        Args:
            evals (list[EvalInterface]): The evaluation instances inherited from EvalInterface.
        '''
        self.logger.info("Register evaluations")
        self.evals = evals

    def run_evals(self, method: str, method_label_type: Union[str, LabelType] = None,
                  measurements: list[EvalInterface] = None):
        """
        Performing evaluations. The result will be saved in Results/Evals, including the detailed evaluation results
        and the average evaluation results.

        Args:
            method (str): The method being used.
            method_label_type (Union[str, LabelType]): The method label being used.
            measurements (list[EvalInterface]): Measurements used to evaluate the model output
        """
        if self.ds is None:
            self.load_dataset()
        if self.evaluator is None or not self.evaluator.compatible(method=method,
                                                                   schema_name=self.cfg.exp_schema_config.option.name,
                                                                   data_store=self.ds):
            self.evaluator = Evaluator(method=method, schema_name=self.cfg.exp_schema_config.option.name,
                                       data_store=self.ds, eval_config=self.cfg.eval_config)
        if isinstance(method_label_type, str):
            method_label_type = LabelType.get_type(method_label_type)

        if measurements is None:
            measurements = self.evals
        self.evaluator.run_eval(measurements=measurements, method_label_type=method_label_type)

    def plots(self, options: list[VisOption], methods: Union[str, list[str]], schemas: Union[str, list[str]],
              method_label_types: Union[str, list[str], LabelType, list[LabelType]] = None):
        """
        Generate plots for the specified method and training schema. The plots are located in Results/Plots.

        Args:
            options (list[VisOption]): Plot options
            methods (Union[str, list[str]]): Method name.
            schemas (Union[str, list[str]]): Training schema name.
            method_label_types:  The type of labels used in plot, default None (using all types of labels)

        Returns:
            None

        """
        if isinstance(methods, str):
            methods = [methods]
        if isinstance(schemas, str):
            schemas = [schemas]
        if isinstance(method_label_types, list):
            method_label_types = LabelType.get_types(method_label_types)
        if isinstance(method_label_types, LabelType):
            method_label_types = [method_label_types]
        self.logger.info("Plotting. Method[{}], Schema[{}].".format(methods, schemas))
        if self.ds is None:
            self.load_dataset()
        vis = Visualizer(self.ds, vis_config=self.cfg.vis_config)
        vis.plot_all(set(options), methods, schemas, method_label_types)

    def summary(self, method_name="*", schema_name="*") -> Tuple[pd.DataFrame, pd.DataFrame]:
        eval_agg_paths = self.pm.get_eval_agg_by_curve_path(method_name=method_name, schema_name=schema_name,
                                                            dataset_type="*", dataset_name="*", curve_name="*",
                                                            label_name="*", cur_timestamp="*", safe_dir=False)
        data = []
        data_expand = []
        for path in glob.glob(eval_agg_paths):
            with open(path) as f:
                eval_format_result = dict()

                schema_name, method_name, dataset_type, dataset_name, curve_name, timestamp, label_name = \
                    self.pm.split_eval_agg_by_curve_path(path)
                eval_format_result[("Attribute", "schema_name")] = schema_name
                eval_format_result[("Attribute", "method_name")] = method_name
                eval_format_result[("Attribute", "dataset_type")] = dataset_type
                eval_format_result[("Attribute", "dataset_name")] = dataset_name
                eval_format_result[("Attribute", "curve_name")] = curve_name
                eval_format_result[("Attribute", "label_name")] = label_name
                eval_format_result[("Attribute", "online")] = "Yes" if \
                    issubclass(BaseMethodMeta.registry[method_name], BaseOnlineMethod) else "No"

                eval_format_prefix = dict(eval_format_result)
                eval_result: dict = json.load(f)

                for eval_name, eval_detail in eval_result.items():

                    if isinstance(eval_detail, dict):
                        for eval_sub_item_key, eval_sub_item_value in eval_detail.items():
                            data_expand.append(dict(eval_format_prefix))
                            data_expand[-1][("Eval", "eval_item")] = (eval_name, eval_sub_item_key)
                            data_expand[-1][("Eval", "value")] = eval_sub_item_value
                            eval_format_result[("Eval", (eval_name, eval_sub_item_key))] = eval_sub_item_value
                    else:
                        data_expand.append(dict(eval_format_prefix))
                        data_expand[-1][("Eval", "eval_item")] = (eval_name, )
                        data_expand[-1][("Eval", "value")] = eval_detail
                        eval_format_result[("Eval", (eval_name,))] = eval_detail

                data.append(eval_format_result)
        return pd.DataFrame(data), pd.DataFrame(data_expand)

    def apply_cfg(self, path=None):
        """
        Applies configuration from a file.

        This method reads a configuration file from the specified path and overrides the corresponding default values.

        NOTE:
            If no path is provided, it uses a default configuration.

        Args:
            path (str, optional):
                The path to the configuration file. If None, default configuration is used. Defaults to None.

        Returns:
            None
        """
        if path is None:
            self.logger.warning("Using Default Config %(origin_cfg_path)s.\n\
                ")
            return

        path = os.path.abspath(path)
        self.cfg.update(path)

        self.logger.info("Reload Config Successfully.")
        self.logger.debug(json.dumps(self.cfg.items(), indent=4))

    def clear_result(self, keep: int = 1):
        test_output_path_format = self.pm.get_test_output_dir(method_name="*", schema_name="*",
                                                              dataset_type="*", dataset_name="*",
                                                              curve_name="*", timestamp="*",
                                                              safe_dir=False)
        all_test_output_files = glob.glob(test_output_path_format)

        def get_unique_key(tgt_path):
            schema_name, method_name, dataset_type, dataset_name, curve_name, timestamp = \
                self.pm.split_test_output_result_path(os.path.join(tgt_path, "."))
            return int(timestamp), (schema_name, method_name, dataset_type, dataset_name, curve_name)

        path_df = pd.DataFrame({"paths": all_test_output_files})
        path_df["ukey"] = path_df["paths"].map(get_unique_key)
        path_df["timestamp"] = path_df["ukey"].map(lambda x: x[0])
        path_df["key"] = path_df["ukey"].map(lambda x: x[1])
        keep_df = path_df.groupby('key', group_keys=False).apply(lambda x: x.nlargest(keep, 'timestamp'))
        keep_ukeys = set(keep_df["ukey"].values)
        remv_df = path_df[~path_df.index.isin(keep_df.index)]
        remove_count = 0
        for index, row in remv_df.iterrows():
            timestamp, (schema_name, method_name, dataset_type, dataset_name, curve_name) = row["ukey"]
            timestamp = str(timestamp)
            tgt_path = self.pm.get_test_output_dir(method_name=method_name, schema_name=schema_name,
                                                   dataset_type=dataset_type, dataset_name=dataset_name,
                                                   curve_name=curve_name, timestamp=timestamp, safe_dir=False)
            for folder_path in glob.glob(tgt_path):
                shutil.rmtree(folder_path)
                remove_count += 1

        eval_output_path_format = self.pm.get_eval_agg_by_curve_path(method_name="*", schema_name="*",
                                                                     dataset_type="*", dataset_name="*",
                                                                     curve_name="*", label_name="*",
                                                                     cur_timestamp="*",
                                                                     safe_dir=False)
        eval_output_paths = glob.glob(eval_output_path_format)
        for eval_output_path in eval_output_paths:
            schema_name, method_name, dataset_type, dataset_name, curve_name, timestamp, label_name = \
                self.pm.split_eval_agg_by_curve_path(eval_output_path)
            unique_key = (int(timestamp), (schema_name, method_name, dataset_type, dataset_name, curve_name))
            if unique_key in keep_ukeys:
                continue
            folder_path = os.path.dirname(eval_output_path)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)
                remove_count += 1

        self.logger.info(f"Remove directory number: {remove_count}")
