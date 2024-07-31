from typing import Tuple
import numpy as np
from ...EvalPreprocess import EvalPreprocessOption
from ...EvalInterface import EvalInterface
from ...Metrics import SingleValueMetric


class RunLengthMSRE(EvalInterface):
    """
    Using Run Length MSRE to evaluate the models.
    """

    def __init__(self, eval_head: bool = True, need_label_convert: bool = True) -> None:
        if not need_label_convert:
            super().__init__("Run Length MSRE", EvalPreprocessOption.Raw)
        elif eval_head:
            super().__init__("Run Length MSRE", EvalPreprocessOption.RunLengthWithHead)
        else:
            super().__init__("Run Length MSRE", EvalPreprocessOption.RunLengthWithoutHead)

    def calc(self, scores: np.ndarray, labels: np.ndarray, margins) -> SingleValueMetric:
        assert scores.ndim == 1, f"Score dimension should be 1 (current score dimension: {scores.ndim}."
        assert labels.ndim == 1, f"Label dimension should be 1 (current label dimension: {labels.ndim}."

        error = scores - labels
        error = error * error
        error = sum(error) / len(error)
        error = np.sqrt(error)
        return SingleValueMetric(
            name=self.name,
            value=error
        )

