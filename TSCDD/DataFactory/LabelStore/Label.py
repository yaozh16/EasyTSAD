from abc import ABC, abstractmethod, ABCMeta
import numpy as np
from pandas import DataFrame
from .LabelType import LabelType
from ..ObjectView import ObjectView
from typing import Dict
from typing import Union


class LabelMeta(ABCMeta):
    registry = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if name != 'Label':
            LabelMeta.registry[name] = cls


class Label(metaclass=LabelMeta):
    def __init__(self, labels: np.ndarray, annotator: str = None):
        self.label_values = labels
        self.annotator = annotator

    @property
    def label_type(self) -> LabelType:
        return LabelType.Default

    def __len__(self) -> int:
        return len(self.label_values)

    def size(self) -> int:
        return len(self.label_values)

    def empty(self) -> bool:
        return len(self) == 0

    def get(self, start_index: int = None, end_index: int = None) -> np.ndarray:
        if start_index is None:
            if end_index is None:
                return self.label_values
            else:
                return self.label_values[:end_index]
        elif end_index is None:
            return self.label_values[start_index:]
        else:
            return self.label_values[start_index:end_index]

    def __getitem__(self, item):
        return self.label_values[item]

    @classmethod
    def build_labels_from_dataframe(cls, dataframe: DataFrame):
        label_dict = {}
        for annotator in dataframe.columns:
            assert isinstance(annotator, str), "Label column (annotator) is not string."
            labels = dataframe[annotator].values
            label_dict[annotator] = cls(labels, annotator)
        return label_dict

    def head(self, n: int):
        return self.__class__(self.label_values[:n], annotator=self.annotator)

    def last(self, n: int):
        return self.__class__(self.label_values[-n:], annotator=self.annotator)

    def to_json(self):
        return {
            "labels": self.label_values.tolist(),
            "label_type": self.label_type.name,
            "annotator": self.annotator,
        }

    @classmethod
    @abstractmethod
    def connect_from(cls, labels: list["Label"], overlap: int = 0, annotator_check: bool = True) -> "Label":
        raise NotImplementedError("Label connection not implemented")


class ScoreLabel(Label):
    @classmethod
    def connect_from(cls, labels: list["ScoreLabel"], overlap: int = 0, annotator_check: bool = True) -> "ScoreLabel":
        assert len(labels) > 0, "Label connection from empty list"

        annotator = labels[0].annotator
        label_type = labels[0].label_type
        for lbl in labels:
            assert label_type == lbl.label_type, "Label connection from different types"
            assert annotator == lbl.annotator, "Label connection from different annotators"
        label_values = cls._connect_label_values([lbl.label_values for i, lbl in enumerate(labels)], overlap)
        return cls(label_values, annotator=annotator)

    @classmethod
    def _connect_label_values(cls, values_list: list[np.ndarray], overlap: int = 0):
        values_list = [e if i == 0 else e[overlap:] for i, e in enumerate(values_list)]
        return np.concatenate(values_list)


class BinaryLabel(ScoreLabel):
    def __init__(self, labels: np.ndarray, annotator: str = None):
        labels = labels.astype(int)
        super().__init__(labels, annotator)
        self.labels = labels
        self.annotator = annotator

    @classmethod
    def from_point_list(cls, non_zero_point_indexes: Union[list[int], np.ndarray], sequence_length: int,
                        annotator: str = None) \
            -> "BinaryLabel":
        label = np.zeros(sequence_length)
        label[list(non_zero_point_indexes)] = 1
        return cls(label, annotator)


class ChangePointLabel(BinaryLabel):
    @property
    def label_type(self):
        return LabelType.ChangePoint

    def __init__(self, labels: np.ndarray, annotator: str = None):
        super().__init__(labels, annotator)


class ReportPointLabel(BinaryLabel):
    @property
    def label_type(self):
        return LabelType.ReportPoint

    def __init__(self, labels: np.ndarray, annotator: str = None):
        super().__init__(labels, annotator)


class AnomalyLabel(BinaryLabel):
    @property
    def label_type(self):
        return LabelType.AnomalyPoint


class RunLengthLabel(ScoreLabel):
    @property
    def label_type(self):
        return LabelType.RunLength

    def __init__(self, labels: np.ndarray, annotator: str = None):
        super().__init__(labels, annotator)

    @classmethod
    def from_change_point_indexes(cls, change_point_indexes: Union[list, np.ndarray], seq_length: int,
                                  annotator: str = None):
        change_point_indexes = list(sorted(change_point_indexes))
        run_lengths = np.arange(seq_length)
        for index in change_point_indexes:
            if index >= 0:
                if index < seq_length:
                    run_lengths[index:] -= run_lengths[index]
            else:
                run_lengths -= run_lengths[0] + index
        return cls(labels=run_lengths, annotator=annotator)

    @classmethod
    def _connect_label_values(cls, values_list: list[np.ndarray], overlap: int = 0):
        try:
            for i in range(len(values_list) - 1):
                v_pre = values_list[i].astype(np.float64)
                v_post = values_list[i + 1].astype(np.float64)
                v_post_changes = np.where(v_post[1:] - v_post[:-1] < 0)[0]
                v_post_change_first = v_post_changes[0] + 1 if len(v_post_changes) > 0 else len(v_post)
                if overlap == 0:
                    move_value = v_pre[-1] + 1 - v_post[0]
                else:
                    move_value = np.mean(v_pre[-overlap:] - v_post[:overlap])
                v_post[:v_post_change_first] += move_value
                v_post[v_post < 0] = 0
                values_list[i+1] = v_post[overlap:]
            result = np.concatenate(values_list)
        except Exception as e:
            raise e
        return result


class ChangeScoreLabel(ScoreLabel):
    @property
    def label_type(self):
        return LabelType.ChangeScore

    def __init__(self, labels: np.ndarray, annotator: str = None):
        super().__init__(labels, annotator)


class LabelView(ObjectView):
    def _inherit(self, parent: "LabelView"):
        pass

    def __init__(self, label: Label, start_index: int, end_index: int, safety_check: bool = False):
        super().__init__(label, start_index, end_index, safety_check)
        self._obj: Label = label

    def label_type(self) -> LabelType:
        return self._obj.label_type

    def get(self) -> np.ndarray:
        return self._obj.get(self.start_index, self.end_index)

    def annotator(self) -> str:
        return self._obj.annotator if isinstance(self._obj.annotator, str) else "UNKNOWN"


class LabelFactory:
    @classmethod
    def _label_cls_mapping(cls) -> Dict[LabelType, type[Label]]:
        return {
            LabelType.Default: Label,
            LabelType.ChangePoint: ChangePointLabel,
            LabelType.RunLength: RunLengthLabel,
            LabelType.ReportPoint: ReportPointLabel,
            LabelType.ChangeScore: ChangeScoreLabel,
        }

    @classmethod
    def construct_label(cls, label: np.ndarray, annotator: str = None, label_type: LabelType = LabelType.Default) -> \
            Label:
        return cls._label_cls_mapping()[label_type](label, annotator)

    @classmethod
    def construct_label_by_name(cls, label: np.ndarray, annotator: str = None,
                                label_type: str = "Default") -> Label:
        for e in LabelType:
            if e.name == label_type:
                return cls._label_cls_mapping()[e](label, annotator)

    @classmethod
    def construct_from_dataframe(cls, dataframe: DataFrame, label_type: LabelType = LabelType.Default) -> \
            dict[str, Label]:
        return cls._label_cls_mapping()[label_type].build_labels_from_dataframe(dataframe)

    @classmethod
    def from_json(cls, json_obj) -> Label:
        return LabelFactory.construct_label_by_name(np.array(json_obj["labels"]), json_obj["annotator"],
                                                    json_obj["label_type"])
