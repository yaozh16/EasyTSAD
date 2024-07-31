import numpy as np

from ...EvalInterface import EvalInterface
from ...EvalPreprocess import EvalPreprocessOption
from ...Metrics import SingleValueMetric


class RunLengthMSE(EvalInterface):
    """
    Using Run Length MSE to evaluate the models.
    """

    def __init__(self, need_label_convert: bool = True) -> None:
        if not need_label_convert:
            super().__init__("Run Length MSE", EvalPreprocessOption.Raw)
        else:
            super().__init__("Run Length MSE", EvalPreprocessOption.RunLengthWithHead)

    def calc(self, scores: np.ndarray, labels: np.ndarray, margins) -> SingleValueMetric:
        assert scores.ndim == 1, f"Score dimension should be 1 (current score dimension: {scores.ndim}."
        assert labels.ndim == 1, f"Label dimension should be 1 (current label dimension: {labels.ndim}."

        error = scores - labels
        error = error * error
        error = sum(error) / len(error)
        return SingleValueMetric(
            name=self.name,
            value=error
        )

