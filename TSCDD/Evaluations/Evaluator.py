import json
import logging

from ..Config.PathManager import PathManager
from .EvalInterface import EvalInterface
from ..Config.EvalConfig import EvalConfig
from ..DataFactory.DataStore import DataStore
from ..DataFactory.LabelStore import LabelType
from ..Evaluations import Performance, MetricInterface
from ..Schema.TestResultLoader import TestResultLoader
from ..utils import Progress
from typing import Union
import time
from ..Config.Options import EvalLoadOption


class Evaluator:
    def __init__(self, method: str, schema_name: str, data_store: DataStore, eval_config: EvalConfig):
        self.method = method
        self.schema_name = schema_name
        self.config = eval_config
        self.ds = data_store
        self.logger = logging.getLogger("logger")
        self.result_loader = TestResultLoader(self.method, self.schema_name, self.ds, eval_config.eval_load_option)

    def compatible(self, method: str, schema_name: str, data_store: DataStore):
        return self.method == method and schema_name == schema_name and self.ds == data_store

    def run_eval(self, measurements: list[EvalInterface], method_label_type: LabelType = None):
        pm = PathManager.get_instance()

        progress = Progress(self.result_loader)

        self.logger.info(f"Method[{self.method}] Schema[{self.schema_name}] Measurement"
                         f"{[e.name for e in measurements]} {self.result_loader.size()} curve(s) in total.")
        dataset_eval_results = {}
        # evaluate each curve
        for timestamp, curve_key, ground_truth_labels, method_output_labels in progress:
            dataset_type, dataset_name, curve_name = curve_key

            progress.info(f"Curve[{dataset_name}, {curve_name}] {len(ground_truth_labels)} labels "
                          f"{len(method_output_labels)} outputs")
            if method_label_type is not None:
                method_output_labels = [e for e in method_output_labels
                                        if e.label_type() == method_label_type]
            if len(method_output_labels) == 0:
                continue

            curve_eval_results: dict[str, list[MetricInterface]] = {}
            # multiple labeller
            for ground_truth_label in ground_truth_labels:
                # multiple samples / outputs
                for method_output_label in method_output_labels:
                    performance = Performance(method_output_label, ground_truth_label, self.config.margins)
                    result = performance.perform_eval(measurements)
                    if result is None:
                        break
                    eval_metric_list: list[MetricInterface] = result[0]
                    for eval_metric in eval_metric_list:
                        if eval_metric.name not in curve_eval_results:
                            curve_eval_results[eval_metric.name] = [eval_metric]
                        else:
                            curve_eval_results[eval_metric.name].append(eval_metric)
            if len(curve_eval_results) == 0:
                continue
            eval_detail_path = pm.get_eval_detail_path(method_name=self.method, schema_name=self.schema_name,
                                                       dataset_type=dataset_type, dataset_name=dataset_name,
                                                       curve_name=curve_name, label_name=method_label_type.name,
                                                       cur_timestamp=timestamp, safe_dir=True)
            eval_agg_path = pm.get_eval_agg_by_curve_path(method_name=self.method, schema_name=self.schema_name,
                                                          dataset_type=dataset_type, dataset_name=dataset_name,
                                                          curve_name=curve_name, label_name=method_label_type.name,
                                                          cur_timestamp=timestamp, safe_dir=True)
            aggregated = {name: metrics[0].aggregate(metrics) for name, metrics in curve_eval_results.items()}
            with open(eval_detail_path, "w") as f:
                srl = {name: [m.to_dict() for m in metrics] for name, metrics in curve_eval_results.items()}

                f.write(json.dumps(srl, indent=2))
            with open(eval_agg_path, "w") as f:
                srl = {}
                for name, metric in aggregated.items():
                    srl.update(metric.to_dict())
                f.write(json.dumps(srl, indent=2))

            if (dataset_type, dataset_name) not in dataset_eval_results:
                dataset_eval_results[(dataset_type, dataset_name)] = dict()
            for name, metric in aggregated.items():
                if name not in dataset_eval_results[(dataset_type, dataset_name)]:
                    dataset_eval_results[(dataset_type, dataset_name)][name] = [metric]
                else:
                    dataset_eval_results[(dataset_type, dataset_name)][name].append(metric)
        # aggregate all evaluation by dataset
        for (dataset_type, dataset_name), metric_dict in dataset_eval_results.items():
            eval_agg_path = pm.get_eval_by_dataset_path(method_name=self.method, schema_name=self.schema_name,
                                                        dataset_type=dataset_type, dataset_name=dataset_name,
                                                        safe_dir=True)
            aggregated = {name: metrics[0].aggregate(metrics) for name, metrics in metric_dict.items()}
            with open(eval_agg_path, "w") as f:
                srl = {name: metric.to_dict() for name, metric in aggregated.items()}
                f.write(json.dumps(srl, indent=2))
