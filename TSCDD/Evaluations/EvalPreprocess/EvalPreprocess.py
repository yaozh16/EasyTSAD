from enum import Enum
from abc import ABC
import numpy as np
from typing import Tuple, Dict
from ...DataFactory.LabelStore import LabelView
from ...DataFactory.LabelStore.LabelType import  LabelType
from ...DataFactory.LabelStore.LabelTools import LabelTools
from copy import deepcopy
import logging


class Align(Enum):
    No = 0
    Left = 1
    Right = 2


class ScoreNorm(Enum):
    No = 0
    NonNeg = 1


class MarginExpand(Enum):
    No = 0
    Yes = 1


class RunLengthCvt(Enum):
    No = 0
    KeepHeadSegment = 1
    DropHeadSegment = 2


class EvalPreprocessOption(Enum):

    Raw = frozenset()
    AlignRight = frozenset([Align.Right])
    Default = frozenset([Align.Right, ScoreNorm.NonNeg, MarginExpand.Yes])
    RunLengthWithHead = frozenset([Align.Right, RunLengthCvt.KeepHeadSegment])


class EvalPreprocess:

    def __init__(self, score: LabelView, label: LabelView, margins: Tuple[int, int]):
        self._logger = logging.getLogger("logger")
        self._raw_score = score
        self._raw_label = label
        self._margins = (margins[0], margins[1]) if margins is not None else (0, 0)
        self._prepared_inputs: Dict[EvalPreprocessOption, Tuple[np.ndarray, np.ndarray]] = dict()
        self._prepared_inputs[EvalPreprocessOption.Raw] = (score.get(), label.get())

    def get_eval_input(self, option: EvalPreprocessOption) -> Tuple[np.ndarray, np.ndarray]:
        if option not in self._prepared_inputs:
            score, label = self._prepared_inputs[EvalPreprocessOption.Raw]

            option_items = option.value

            if Align.Left in option_items:
                label = label[-len(score):]  # use the later slice of the label sequence
            elif Align.Right in option_items:
                label = label[:len(score)]  # use the earlier slice of the label sequence

            if ScoreNorm.NonNeg in option_items:
                score = score - np.min(score)

            if MarginExpand.Yes in option_items:
                label = self._expand_by_margin(label)

            elif RunLengthCvt.KeepHeadSegment in option_items:
                if not self._raw_label.label_type() == LabelType.RunLength:
                    if self._raw_label.label_type() == LabelType.ChangePoint:
                        label = self._run_length_convert(label, False)
                    else:
                        raise NotImplementedError(f"convert from {self._raw_label.label_type()} to "
                                                  f"RunLength is not defined yet.")
                if not self._raw_score.label_type() == LabelType.RunLength:
                    if self._raw_score.label_type() == LabelType.ChangePoint:
                        score = self._run_length_convert(score, False)
                    else:
                        raise NotImplementedError(f"convert from {self._raw_score.label_type()} to "
                                                  f"RunLength is not defined yet.")

            self._prepared_inputs[option] = (score, label)
        return self._prepared_inputs[option]

    def _expand_by_margin(self, label: np.ndarray) -> np.ndarray:
        pre_margin, post_margin = self._margins
        if pre_margin == 0 and post_margin == 0:
            return label
        label = deepcopy(label)
        for i in range(pre_margin):
            label[:-1] |= ((~np.roll(label, 1, axis=0)) & np.roll(label, -1, axis=0))[:-1]
        for i in range(post_margin):
            label[1:] |= ((~np.roll(label, -1, axis=0)) & np.roll(label, 1, axis=0))[1:]
        return label

    @staticmethod
    def _run_length_convert(binary_label: np.ndarray, drop_head_seg: bool = True) -> np.ndarray:
        run_length, (seg_starts, seg_ends) = LabelTools.convert_binary_to_run_length(binary_label)

        if drop_head_seg:
            if len(seg_starts) > 0:
                run_length = run_length[seg_starts[0]:]
            else:
                raise ValueError("Label without change points cannot be convert to run length.")
        return run_length
