import os
import glob
import pandas as pd
import logging
import json
from typing import Tuple
from TSCDD.Config.PathManager import PathManager
from TSCDD.DataFactory.DataStore import DataStoreViewIndex, DataStore
from TSCDD.DataFactory.TSData import TSDataView
from TSCDD.DataFactory.LabelStore import LabelView
from TSCDD.Schema.TestSchema.TestResults import TestResult
from collections.abc import Iterable
from ..Config.Options import EvalLoadOption


class TestResultLoader(Iterable):
    def __init__(self, method_name: str, schema_name: str, data_store: DataStore, eval_load_option: EvalLoadOption):
        self._logger = logging.getLogger("logger")
        self._ds: DataStore = data_store
        pm = PathManager.get_instance()

        def get_curve_keys(paths):
            keys = []
            ukeys = []
            for path in paths:
                sn, mn, dataset_type, dataset_name, curve_name, timestamp = \
                    pm.split_test_output_result_path(path)
                keys.append((dataset_type, dataset_name, curve_name))
                ukeys.append((int(timestamp), (dataset_type, dataset_name, curve_name)))
            return keys, ukeys

        test_view_paths = glob.glob(pm.get_test_output_reference_view_path(method_name=method_name,
                                                                           schema_name=schema_name,
                                                                           dataset_type="*",
                                                                           dataset_name="*",
                                                                           curve_name="*",
                                                                           timestamp="*",
                                                                           safe_dir=False))

        result_paths = glob.glob(
            pm.get_test_output_result_path(method_name=method_name, schema_name=schema_name, dataset_type="*",
                                           dataset_name="*", curve_name="*", timestamp="*", safe_dir=False))

        def filter_outputs(path_list: list, column_name: str):
            curve_keys, ukeys = get_curve_keys(path_list)
            path_df = pd.DataFrame({column_name: path_list,
                                    "ukeys": ukeys,
                                    "curve_keys": curve_keys})
            path_df = path_df.sort_values(["ukeys"])
            if eval_load_option is EvalLoadOption.Last:
                path_df = path_df.groupby(["curve_keys"]).last()
            path_df = path_df.set_index("ukeys")
            path_df = path_df[[column_name]]
            return path_df

        test_view_path_df = filter_outputs(test_view_paths, "test_view_path")
        result_path_df = filter_outputs(result_paths, "result_path")

        self._curves = pd.concat([
            test_view_path_df,
            result_path_df,
        ], axis=1).dropna()

    def __len__(self):
        return self.size()

    def size(self):
        return len(self._curves.index)

    def __iter__(self) -> Tuple[Tuple, list[LabelView], list[LabelView]]:
        # evaluate each curve
        for row_key, row in self._curves.iterrows():
            timestamp, curve_key = row_key
            dataset_type, dataset_name, curve_name = curve_key
            test_view_index = DataStoreViewIndex()

            try:
                with open(row["test_view_path"], "r") as f:
                    ground_truth_view_json = json.load(f)
                    test_view_index.add_item(key=curve_key, start_index=ground_truth_view_json["start"],
                                             end_index=ground_truth_view_json["end"], dims=ground_truth_view_json["dims"])
                test_data_view: TSDataView = self._ds[test_view_index][curve_key]
                with open(row["result_path"], "r") as f:
                    result_json = json.load(f)
                    test_result = TestResult.from_json(result_json)
                ground_truth_labels: list[LabelView] = test_data_view.get_labels().get_labels()
                method_output_labels: list[LabelView] = test_result.get_result().label_stores.get_labels()
            except KeyError:
                continue
            yield timestamp, curve_key, ground_truth_labels, method_output_labels






