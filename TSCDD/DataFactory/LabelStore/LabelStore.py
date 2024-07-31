from typing import Dict, Union

import numpy as np

from .Label import Label, LabelFactory, LabelView
from .LabelType import LabelType
from abc import ABC, abstractmethod
import pandas as pd
import logging
from ..ObjectView import ObjectView


logger = logging.getLogger("logging")


class LabelStore:
    def __init__(self, labels: Union[None, Label, list[Label]] = None):
        self.labels: list[Label] = []
        self._len = 0
        self._count = 0
        self._label_types = set()

        if labels is None:
            return
        elif isinstance(labels, Label):
            self.set_label(labels)
        elif isinstance(labels, list):
            for label in labels:
                self.set_label(label)

    def load(self, label_path: str, label_type: LabelType):
        try:
            labels = pd.read_pickle(label_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Label file not found: {label_path}")
        except Exception:
            raise Exception(f"Label file can not be read: {label_path}")

        label_dict = LabelFactory.construct_from_dataframe(labels, label_type)
        for annotator, label in label_dict.items():
            self.set_label(label)

    def label_length(self):
        return self._len

    def label_counts(self):
        return len(self.labels)

    def get_label_types(self) -> frozenset:
        return frozenset(self._label_types)

    def set_label(self, label: Label):
        if not isinstance(label, Label):
            raise TypeError("Input object is not a Label")
        self.labels.append(label)
        self._label_types.add(label.label_type)
        self._len = int(max(self._len, label.size()))

    def get_labels(self, start_index: int = None, end_index: int = None, label_type: LabelType = None) -> \
            list[LabelView]:
        if label_type is not None:
            return [LabelView(lb, start_index, end_index) for lb in self.labels if label_type == lb.label_type]
        else:
            return [LabelView(lb, start_index, end_index) for lb in self.labels]

    def to_json(self):
        return [label.to_json() for label in self.labels]

    @classmethod
    def from_json(cls, json_obj: list):
        return cls(labels=[LabelFactory.from_json(e) for e in json_obj])


class LabelStoreView(ObjectView):
    def _inherit(self, parent: "LabelStoreView"):
        pass

    def __init__(self, labels: LabelStore, start_index: int, end_index: int):
        super().__init__(labels, start_index, end_index)
        self._obj: LabelStore = labels
        self._iter_index = 0
        self._label_values: list[LabelView] = []

    def __iter__(self):
        self._iter_index = 0
        self._label_values:  list[LabelView] = self.get_labels()
        return self

    def __next__(self) -> LabelView:
        if self._iter_index >= len(self._label_values):
            raise StopIteration
        self._iter_index += 1
        return self._label_values[self._iter_index]

    def get_labels(self) -> list[LabelView]:
        return self._obj.get_labels(self.start_index, self.end_index)



