from typing import Tuple
import numpy as np
from ...EvalPreprocess import EvalPreprocessOption
from ...MetricInterface import MetricInterface
from ...EvalInterface import EvalInterface
from ...Metrics import SingleValueMetric


class RandIndex(EvalInterface):
    """
    Using Rand Index to evaluate the models.
    """

    def __init__(self) -> None:
        super().__init__("Rand Index", EvalPreprocessOption.AlignRight)

    @staticmethod
    def __get_partition_matrix(binary_arr: np.ndarray) -> np.ndarray:
        arr_len = len(binary_arr)
        arr_ext = np.concatenate([np.ones_like(binary_arr[:1]), binary_arr, np.ones_like(binary_arr[:1])])
        seg_starts, = np.where(arr_ext[:-1] & ~arr_ext[1:])
        seg_ends, = np.where(~arr_ext[:-1] & arr_ext[1:])
        splits = np.hstack([[0] if seg_starts[0] > 0 else [],
                            np.vstack([seg_starts, seg_ends]).T.flatten(),
                            [arr_len] if seg_ends[-1] < arr_len else []]).astype(np.int32)

        partition_matrix = np.zeros((arr_len, arr_len))
        for i in range(len(splits)-1):
            start = splits[i]
            end = splits[i+1]
            partition_matrix[start:end, start:end] = 1
        return partition_matrix

    def calc(self, scores: np.ndarray, labels: np.ndarray, margins) -> SingleValueMetric:
        assert scores.ndim == 1, f"Score dimension should be 1 (current score dimension: {scores.ndim}."
        assert labels.ndim == 1, f"Label dimension should be 1 (current label dimension: {labels.ndim}."
        L = len(labels)

        score_partition_matrix = self.__get_partition_matrix(scores)
        label_partition_matrix = self.__get_partition_matrix(labels)

        disagreement = score_partition_matrix != label_partition_matrix

        rand_index = 1 - np.sum(disagreement) / (2 * L * (L - 1))
        return SingleValueMetric(
            name=self.name,
            value=rand_index
        )


