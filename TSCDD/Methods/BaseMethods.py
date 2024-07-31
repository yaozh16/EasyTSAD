import json
import logging
import os
from abc import ABC
from abc import ABCMeta, abstractmethod
from typing import Union

import numpy as np
import toml

from ..utils import update_nested_dict
from ..DataFactory.DataStore import DataStoreViewIndex
from ..DataFactory.LabelStore import LabelStore
from ..DataFactory.TSData import TimeSeriesView


class MethodTestResults:
    def __init__(self, label_store: LabelStore = None):
        self.label_stores = label_store

    def set_label_store(self, label_stores: LabelStore = None):
        self.label_stores = label_stores

    def get_label_store(self):
        return self.label_stores

    def to_json(self):
        return self.label_stores.to_json()

    @classmethod
    def from_json(cls, json_obj):
        return cls(label_store=LabelStore.from_json(json_obj))


class BaseMethodMeta(ABCMeta):
    """
    Metaclass register implemented methods automatically. This allows the usage of runtime arguments to specify the method to run experiments.

    Attributes:
        registry (dict): Registry to store the registered methods.

    """
    registry = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if name != 'BaseMethod':
            BaseMethodMeta.registry[name] = cls


class BaseMethod(metaclass=BaseMethodMeta):
    @classmethod
    @abstractmethod
    def _method_file_path(cls) -> str:
        raise NotImplementedError("BaseMethod method default_config_path is not implemented.")

    @classmethod
    def method_default_config_path(cls) -> Union[None, str]:
        config_dir = os.path.dirname(cls._method_file_path())
        for method_default_config_path in [
            os.path.join(config_dir, f"{cls.__name__}_config.toml"),
            os.path.join(config_dir, f"{cls.__name__}_config.json"),
            os.path.join(config_dir, f"config.toml"),
            os.path.join(config_dir, f"config.json"),
        ]:
            if os.path.exists(method_default_config_path):
                return method_default_config_path
        return None

    @classmethod
    def _load_default_config(cls) -> dict:
        default_config_path = cls.method_default_config_path()
        if isinstance(default_config_path, str) and os.path.exists(default_config_path):
            if default_config_path.endswith(".toml"):
                return toml.load(default_config_path)
            elif default_config_path.endswith(".json"):
                with open(default_config_path, "r") as f:
                    return json.load(f)
            else:
                raise NotImplementedError(f"Unknown default config type: {default_config_path}")
        else:
            return {}

    def __init__(self, hparams: dict = None):
        self.logger = logging.getLogger("logger")
        hparams = hparams if isinstance(hparams, dict) else {}
        self.hparams = update_nested_dict(self._load_default_config(), hparams)
        self.test_results = MethodTestResults()

    def get_results(self):
        return self.test_results

    @abstractmethod
    def train_valid(self, data_store_view_index: DataStoreViewIndex):
        raise NotImplementedError(f"{self.__class__} train_valid_phase is not implemented")

    @abstractmethod
    def test(self, timeseries: TimeSeriesView) -> MethodTestResults:
        raise NotImplementedError(f"{self.__class__} test_phase is not implemented")

    def param_statistic(self, save_file):
        pass


class BaseOfflineMethod(BaseMethod, ABC):
    def test(self, timeseries: TimeSeriesView) -> MethodTestResults:
        return self.offline_test(timeseries)

    @abstractmethod
    def offline_initialize(self):
        raise NotImplementedError("OfflineMethod offline_initialize is not implemented.")

    @abstractmethod
    def offline_test(self, timeseries: TimeSeriesView) -> MethodTestResults:
        raise NotImplementedError("OfflineMethod offline_test is not implemented.")

    @classmethod
    def internal_online(cls) -> bool:
        """
        override this if the method runs in an online style but implemented as an BaseOfflineMethod
        """
        return False


class BaseOnlineMethod(BaseMethod, ABC):
    class OnlineState:
        def __init__(self):
            self.test_indexes = None
            self.step = 0
            self.need_retrain = True
            self.custom_dict = dict()

        def initialize(self):
            self.test_indexes = set()
            self.need_retrain = True
            self.step = 0

        def update(self, custom_dict: dict):
            self.custom_dict.update(custom_dict)

        def record_index(self, index_arr: np.ndarray):
            self.test_indexes |= set(index_arr)

    def __init__(self, hparams):
        super().__init__(hparams)
        self.online_state = self.OnlineState()

    def test(self, timeseries: TimeSeriesView):
        return self.online_step(timeseries)

    @abstractmethod
    def online_initialize(self):
        raise NotImplementedError("OnlineMethod initialize is not implemented.")

    @abstractmethod
    def online_retrain(self, timeseries: TimeSeriesView):
        raise NotImplementedError("OnlineMethod retrain is not implemented.")

    @abstractmethod
    def need_retrain(self) -> bool:
        raise NotImplementedError("OnlineMethod need_retrain is not implemented.")

    @abstractmethod
    def online_step(self, timeseries: TimeSeriesView):
        raise NotImplementedError("OnlineMethod online_step is not implemented.")

    @abstractmethod
    def online_collect_results(self) -> MethodTestResults:
        raise NotImplementedError("OnlineMethod collect_results is not implemented.")


class MethodFactory:
    @classmethod
    def construct_by_name(cls, method_name: str, hparam: dict = None) \
            -> Union[None, BaseMethod]:
        model_cls = BaseMethodMeta.registry.get(method_name, None)
        if model_cls is not None:
            return model_cls(hparam)
        else:
            return None
