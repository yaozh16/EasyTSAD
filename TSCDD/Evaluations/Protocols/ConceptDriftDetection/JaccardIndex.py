from typing import Tuple
import numpy as np
from ...EvalPreprocess import EvalPreprocessOption
from ...EvalInterface import EvalInterface
from ...Metrics import SingleValueMetric


class SegmentJaccardIndex(EvalInterface):
    """
    Using Jaccard Index to evaluate the models.
    """

    def __init__(self) -> None:
        super().__init__("Jaccard Index", EvalPreprocessOption.Default)


    @staticmethod
    def __partion_jaccard_index(ground_truth: list[Tuple[int, int]], test_partition: list[Tuple[int, int]]) -> float:
        seqs = []
        for s1 in ground_truth:
            seqs.append([])
            for s2 in test_partition:
                seqs[-1].append(SegmentJaccardIndex.__segment_jaccard_index(s1, s2))
        seqs = np.array(seqs)
        seqs = np.max(seqs, axis=1)
        seg1_len = np.array(ground_truth)[:, 1] - np.array(ground_truth)[:, 0]
        return np.average(seqs, weights=seg1_len)

    @staticmethod
    def __segment_jaccard_index(segment1: Tuple[int, int], segment2: Tuple[int, int]):
        s11, s12 = segment1
        s21, s22 = segment2
        if s11 >= s22 or s21 >= s12:
            return 0
        elif (s21 - s11) * (s22 - s12) <= 0:
            L1 = s12 - s11
            L2 = s22 - s21
            return min(L1, L2) / max(L1, L2)
        else:
            L1 = s22 - s11
            L2 = s12 - s21
            return min(L1, L2) / max(L1, L2)


    @staticmethod
    def __get_segments(binary_arr: np.ndarray) -> list[Tuple[int, int]]:
        arr_ext = np.concatenate([np.ones_like(binary_arr[:1]), binary_arr, np.ones_like(binary_arr[:1])])
        seg_starts, = np.where(arr_ext[:-1] & ~arr_ext[1:])
        seg_ends, = np.where(~arr_ext[:-1] & arr_ext[1:])
        return list(zip(seg_starts, seg_ends))

    def calc(self, scores, labels, margins) -> SingleValueMetric:
        assert scores.ndim == 1, f"Score dimension should be 1 (current score dimension: {scores.ndim}."
        assert labels.ndim == 1, f"Label dimension should be 1 (current label dimension: {labels.ndim}."

        score_segments = self.__get_segments(scores)
        label_segments = self.__get_segments(labels)

        ji = self.__partion_jaccard_index(label_segments, score_segments)

        return SingleValueMetric(
            name=self.name,
            value=ji
        )
