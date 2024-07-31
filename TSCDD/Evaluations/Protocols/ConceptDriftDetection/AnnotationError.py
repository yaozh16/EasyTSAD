import numpy as np
from ...EvalPreprocess import EvalPreprocessOption
from ...EvalInterface import EvalInterface
from ...Metrics import SingleValueMetric


class AnnotationError(EvalInterface):
    """
    Using Annotation Error to evaluate the models.
    """

    def __init__(self) -> None:
        super().__init__("Annotation Error", EvalPreprocessOption.Default)

    def calc(self, scores: np.ndarray, labels: np.ndarray, margins) -> SingleValueMetric:
        assert scores.ndim == 1, f"Score dimension should be 1 (current score dimension: {scores.ndim}."
        assert labels.ndim == 1, f"Label dimension should be 1 (current label dimension: {labels.ndim}."

        label_change_point_count = np.count_nonzero(labels & ~np.roll(labels, -1))
        score_change_point_count = np.count_nonzero(scores & ~np.roll(scores, -1))

        return SingleValueMetric(
            name=self.name,
            value=np.abs(label_change_point_count - score_change_point_count)
        )


