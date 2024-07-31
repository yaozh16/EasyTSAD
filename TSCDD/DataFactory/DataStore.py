import logging
import os.path
import random
from itertools import accumulate
from typing import List, Dict
from typing import Union, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from .TSData import TSData, TSDataView
from ..Config import DataConfig
from ..Config.PathManager import PathManager


class DataStoreViewIndex:
    def __init__(self):
        self._items: Dict[Tuple[str, str, str], Tuple[int, int, list[int]]] = {}

    def add_item(self, key: Tuple[str, str, str], start_index: int, end_index: int,
                 dims: Union[list[int], int]):
        if isinstance(dims, int):
            dims = list(range(dims))
        self._items[key] = (start_index, end_index, dims)

    def get_keys(self):
        return self._items.keys()

    def get_items(self):
        return self._items.items()

    def __and__(self, other: "DataStoreViewIndex"):
        view = DataStoreViewIndex()
        for key, (start, end, dims) in self._items:
            if key in other._items:
                start_other, end_other, dims_other = other._items[key]
                if isinstance(dims, list):
                    if isinstance(dims_other, list):
                        inter_dims = [d for d in dims if d in dims_other]
                        if len(inter_dims) == 0:
                            continue
                    else:
                        inter_dims = dims
                elif isinstance(dims_other, list):
                    inter_dims = dims_other
                else:
                    inter_dims = None
                if start >= end_other or end <= start_other:
                    continue
                else:
                    inter_start = int(max(start, start_other))
                    inter_end = int(min(end, end_other))
                    view.add_item(key, inter_start, inter_end, inter_dims)
        return view

    def split_by_fractions(self, fractions: Union[float, list[float]]) -> list["DataStoreViewIndex"]:
        if isinstance(fractions, float):
            fractions = min(max(fractions, 0.0), 1.0)
            fractions = [fractions, 1 - fractions]

        views = [DataStoreViewIndex() for i in range(len(fractions))]
        glb_fractions = np.array(fractions) / sum(fractions)
        glb_fractions = [0] + list(accumulate(glb_fractions))[:-1] + [1]
        glb_fractions = np.array(glb_fractions)
        for curve_key, (start, end, dims) in self._items.items():
            segment_indexes = glb_fractions * (end - start) + start
            segment_indexes = [int(e) for e in segment_indexes]
            for i, (seg_start, seg_end) in enumerate(list(zip(segment_indexes[:-1], segment_indexes[1:]))):
                views[i].add_item(curve_key, seg_start, seg_end, dims)
        return views

    def get_single_key_view(self, key: Tuple[str, str, str]) -> "DataStoreViewIndex":
        view_index = DataStoreViewIndex()
        view_index._items[key] = self._items[key]
        return view_index

    def to_json(self):
        return self._items

    @staticmethod
    def from_json(json_obj):
        view_index = DataStoreViewIndex()
        view_index._items.update(json_obj)
        return view_index


class DataStore:
    def __init__(self, data_config: DataConfig):
        self._config = data_config
        self._ts_data: Dict[Tuple[str, str, str], TSData] = {}
        self._logger = logging.getLogger("logger")
        self._dataset_curves = pd.DataFrame([], columns=["dataset_type", "dataset", "curve", "key"])

    @classmethod
    def build(cls, data_config: DataConfig):
        return cls(data_config)

    def load(self):
        data_dir = self._config.dataset_dir
        dataset_type = self._config.dataset_type
        datasets = self._config.datasets
        # define dataset directory
        if data_dir is None:
            raise ValueError("Missing Dataset Directory Path. \n"
                             "Please specify the dataset directory path.")

        pm = PathManager.get_instance()
        if not os.path.exists(data_dir):
            raise FileNotFoundError("Dataset Directory %s does not exist." % data_dir)

        pm.load_dataset_path(data_dir)
        self._logger.info("Dataset Directory has been loaded.")

        if isinstance(datasets, str):
            datasets = [datasets]
        if datasets is None:
            self._logger.info(f"Try loading all datasets from {dataset_type}.")
            datasets = os.listdir(os.path.join(data_dir, dataset_type))
            datasets = [e for e in datasets if os.path.isdir(e)]

        # check if datasets exists
        dataset_curves = []
        for dataset in datasets:
            if isinstance(dataset, str):
                dataset_path = pm.get_dataset_path(dataset_type, dataset)
                if not os.path.exists(dataset_path):
                    raise FileNotFoundError("%s does not exist. Please Check the directory path." % dataset_path)
                dataset_curves.extend(
                    [[dataset_type, dataset, curve] for curve in pm.get_dataset_curves(dataset_type, dataset)])
            else:
                try:
                    dataset_key, curve_names = dataset
                except Exception:
                    raise ValueError("%s is not valid format. Check the directory path." % dataset)
                for curve in curve_names:
                    curve_path = pm.get_dataset_one_curve(dataset_type, dataset_key, curve)
                    if not os.path.exists(curve_path):
                        raise FileNotFoundError("%s does not exist. Please Check the directory path." % curve_path)
                    dataset_curves.append([dataset_type, dataset_key, curve])
        self._dataset_curves = DataFrame(dataset_curves, columns=["dataset_type", "dataset", "curve"])
        self._dataset_curves["key"] = list(zip(self._dataset_curves["dataset_type"],
                                               self._dataset_curves["dataset"],
                                               self._dataset_curves["curve"]))
        for key in self._dataset_curves["key"]:
            dataset_type, dataset, curve_name = key
            self._ts_data[key] = TSData.buildfrom(dataset_type, dataset, curve_name, self._config.label_types,
                                                  self._config.preprocess)
        self._dataset_curves["n_obs"] = self._dataset_curves["key"].map(lambda k: self._ts_data[k].size())
        self._dataset_curves["n_dim"] = self._dataset_curves["key"].map(lambda k: self._ts_data[k].get_dim())

    def filter(self, key_filter_callback) -> DataStoreViewIndex:
        view = DataStoreViewIndex()
        for key in self.get_dataset_curve_keys():
            if key_filter_callback(key):
                ts_data = self.get_ts_data(key)
                view.add_item(key, 0, ts_data.size(), ts_data.get_dim())
        return view

    def univariate_curves(self):
        view = DataStoreViewIndex()
        for key in self.get_dataset_curve_keys():
            if self._ts_data[key].get_dim() == 1:
                ts_data = self.get_ts_data(key)
                view.add_item(key, 0, ts_data.size(), ts_data.get_dim())
        return view

    def multivariate_curves(self):
        view = DataStoreViewIndex()
        for key in self.get_dataset_curve_keys():
            if self._ts_data[key].get_dim() > 1:
                ts_data = self.get_ts_data(key)
                view.add_item(key, 0, ts_data.size(), ts_data.get_dim())
        return view

    def get_full_view(self) -> DataStoreViewIndex:
        view_index = DataStoreViewIndex()
        for curve_key, ts_data in self._ts_data.items():
            view_index.add_item(curve_key, 0, ts_data.size(), ts_data.get_dim())
        return view_index

    def split_views_by_curve_keys(self, fractions: list[float], shuffle=True):
        keys = list(self.get_dataset_curve_keys())
        if shuffle:
            random.shuffle(keys)
        fractions = np.array(fractions) / sum(fractions) * len(keys)
        glb_fractions = [0] + list(accumulate(fractions))[:-1] + [len(keys)]
        views = []
        for start_index, end_index in zip(glb_fractions[:-1], glb_fractions[1:]):
            views.append(DataStoreViewIndex())
            for key in keys[start_index: end_index]:
                ts_data = self.get_ts_data(key)
                views[-1].add_item(key, 0, ts_data.size(), ts_data.get_dim())
        return views

    def get_dataset_curves_paths(self) -> DataFrame:
        return self._dataset_curves

    def get_dataset_curve_keys(self) -> List[Tuple[str, str, str]]:
        return self._dataset_curves["key"].tolist()

    def get_ts_data(self, key: Tuple[str, str, str]) -> Union[TSData, None]:
        return self._ts_data.get(key, None)

    def get_single_view(self, key: Tuple[str, str, str]) -> DataStoreViewIndex:
        view = DataStoreViewIndex()
        ts_data = self.get_ts_data(key)
        view.add_item(key, 0, ts_data.size(), ts_data.get_dim())
        return view

    def __getitem__(self, view: DataStoreViewIndex) -> Dict[Tuple[str, str, str], TSDataView]:
        views = {}
        for key, (start_index, end_index, dims) in view.get_items():
            ts_data = self.get_ts_data(key)
            if ts_data is not None:
                views[key] = ts_data.get_view(start_index, end_index, dims)
        return views

