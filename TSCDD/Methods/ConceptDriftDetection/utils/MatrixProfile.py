import numpy as np
from enum import Enum
from collections import deque
from itertools import islice


class DistanceOption(Enum):
    Euclidean = "euclidean"
    EuclideanNorm = "euclidean_norm"
    Cosine = "cosine"


class IAC:
    _mpi_right_iac = dict()
    _mpi_iac = dict()

    @classmethod
    def get_mpi_right_iac(cls, L, count_head: bool = False):
        if L not in cls._mpi_right_iac:
            iac_probability_curve = np.zeros(L)
            for i in range(L - 1):
                if count_head:
                    iac_probability_curve[i:-1] += np.flip(np.arange(1, L - i) / (L - i - 1))
                else:
                    iac_probability_curve[i + 1:-1] += np.flip(np.arange(1, L - i - 1) / (L - i - 1))
            iac_probability_curve *= L / (L - 1)
            iac_probability_curve[iac_probability_curve <= 0] = 1e-10

            cls._mpi_right_iac[L] = iac_probability_curve
        return cls._mpi_right_iac[L]

    @classmethod
    def get_mpi_iac(cls, L):
        if L not in cls._mpi_iac:
            arr = - np.square(np.linspace(-L / 2, L / 2, L)) * 2 / L + L / 2
            arr[arr <= 0] = 1e-10
            cls._mpi_iac[L] = arr
        return cls._mpi_iac[L]


class AC:
    @classmethod
    def calc_ac(cls, mpi, margin_ignore: int = 3):
        L = len(mpi)
        valid_start = 0 if mpi[0] > 0 else 1
        valid_end = L if mpi[L-1] > 0 else L-1
        assert valid_end > valid_start, "the mpi array is too short"

        n = len(mpi)
        ri = np.arange(n)

        arc_starts = np.minimum(ri, mpi)[valid_start: valid_end]
        arc_ends = np.maximum(ri, mpi)[valid_start: valid_end]
        arc_increase = np.bincount(arc_starts, minlength=n) - np.bincount(arc_ends, minlength=n)
        ac = np.cumsum(arc_increase)
        ac[:margin_ignore] = 1.
        ac[-margin_ignore:] = 1.
        return ac


class MatrixProfileRightStream:
    def __init__(self, arr_1d: np.ndarray, window: int = 3, ignore_distance: int = 1,
                 distance_option: DistanceOption = DistanceOption.EuclideanNorm, egress_oldest: bool = True,
                 index_rolling: bool = True):
        self._egress_oldest = egress_oldest
        if self._egress_oldest:
            self._arr_1d: deque = deque(arr_1d)
        else:
            self._arr_1d: deque = deque(arr_1d, maxlen=len(arr_1d))
        self._index_rolling = index_rolling
        self._window = window
        self._ignore_distance = ignore_distance
        self._distance_option = distance_option

        assert arr_1d.ndim == 1, f"Array dimension should be 1 (currently {arr_1d.ndim})"
        assert ignore_distance >= 1, f"Ignore_distance should be positive (currently {ignore_distance})"
        N = len(arr_1d)
        L = N - window + 1
        assert L > 1, "array not long enough"
        self._L = L

        mp_right = np.ones(L, dtype=np.float64) * np.inf
        mpi_right = np.ones(L, dtype=np.int32) * -1

        self._subsequences = deque([self._process_subsequence(arr_1d[i:i + window]).astype(float) for i in range(L)],
                                   maxlen=L)

        for i in range(L - 1):
            for j in range(i + ignore_distance, L):
                distance = self._distance(self._subsequences[i], self._subsequences[j])
                if distance < mp_right[i]:
                    mpi_right[i] = j
                    mp_right[i] = distance

        self._mp_right = deque(mp_right, maxlen=L)
        self._mpi_right = deque(mpi_right, maxlen=L)
        self._update_count = 0

    def get_L(self):
        return self._L

    def get_mpi_right(self) -> np.ndarray:
        if self._index_rolling:
            return np.array(self._mpi_right) - self._update_count
        else:
            return np.array(self._mpi_right)

    def get_mpi_right_raw(self) -> deque:
        return self._mpi_right

    def get_mp_right(self) -> deque:
        return self._mp_right

    def _process_subsequence(self, subsequence: np.ndarray):
        if self._distance_option is DistanceOption.EuclideanNorm:
            subsequence -= np.mean(subsequence, keepdims=True)
        elif self._distance_option is DistanceOption.Cosine:
            subsequence /= np.sqrt(np.sum(np.square(subsequence), keepdims=True))
        else:
            raise NotImplementedError("Unknown distance yet")
        return subsequence

    def _distance(self, sq1: np.ndarray, sq2: np.ndarray):
        if self._distance_option is DistanceOption.Euclidean or self._distance_option is DistanceOption.EuclideanNorm:
            return np.sqrt(np.sum(np.square(sq1 - sq2)))
        elif self._distance_option is DistanceOption.Cosine:
            return np.dot(sq1, sq2)
        else:
            raise NotImplementedError("Unknown distance yet")

    def update(self, v):
        self._arr_1d.append(float(v))
        current_index = self._L + self._update_count
        current_sequence = np.array(list(islice(self._arr_1d, len(self._arr_1d) - self._window, len(self._arr_1d))))
        current_sequence = self._process_subsequence(current_sequence)

        # push forward
        self._mp_right.append(np.inf)
        self._mpi_right.append(-1)
        self._subsequences.append(current_sequence)
        # the new added element is at [self._L + self._update_count]
        for i in range(self._L - self._ignore_distance):
            distance = self._distance(self._subsequences[i], current_sequence)
            if distance < self._mp_right[i]:
                self._mpi_right[i] = current_index
                self._mp_right[i] = distance

        self._update_count += 1


class MatrixProfile:

    @classmethod
    def matrix_profile_1d(cls, arr_1d: np.ndarray, window: int = 3, ignore_distance: int = 1,
                          distance_option: DistanceOption = DistanceOption.Euclidean):
        """

        Args:
            arr_1d: sequence
            window: subsequence length
            ignore_distance: number of points ignored around each point
            distance_option: distance used

        Returns:
            mp, mpi, mp_left, mpi_left, mp_right, mpi_right
            Note the first element of mp_left and mpi_left is meaningless, and
            the last element of mp_right and mpi_right is meaningless.
        """
        assert arr_1d.ndim == 1, f"Array dimension should be 1 (currently {arr_1d.ndim})"
        assert ignore_distance >= 1, f"Ignore_distance should be positive (currently {ignore_distance})"
        N = len(arr_1d)
        L = N - window + 1
        assert L > 1, "array not long enough"

        mp_left = np.ones(L, dtype=np.float64) * np.inf
        mp_right = np.ones(L, dtype=np.float64) * np.inf
        mpi_left = np.ones(L, dtype=np.int32) * -1
        mpi_right = np.ones(L, dtype=np.int32) * -1

        subsequences = np.vstack([arr_1d[i:i + window] for i in range(L)]).astype(float)
        subsequences = cls._process_subsequence(subsequences, distance_option)

        for i in range(L - 1):
            for j in range(i + ignore_distance, L):
                distance = cls._distance(sq1=subsequences[i], sq2=subsequences[j], distance_option=distance_option)
                if distance < mp_right[i]:
                    mpi_right[i] = j
                    mp_right[i] = distance
                if distance < mp_left[j]:
                    mpi_left[j] = i
                    mp_left[j] = distance

        mp_use_left = mp_left < mp_right

        mp = np.where(mp_use_left, mp_left, mp_right)
        mpi = np.where(mp_use_left, mpi_left, mpi_right)

        return mp, mpi, mp_left, mpi_left, mp_right, mpi_right

    @classmethod
    def _process_subsequence(cls, subsequences: np.ndarray, distance_option: DistanceOption):
        if distance_option is DistanceOption.EuclideanNorm:
            if subsequences.ndim == 1:
                subsequences -= np.mean(subsequences, keepdims=True)
            elif subsequences.ndim == 2:
                subsequences -= np.mean(subsequences, axis=1, keepdims=True)
            else:
                raise NotImplementedError("Unknown sequence shape")
        elif distance_option is DistanceOption.Cosine:
            if subsequences.ndim == 1:
                subsequences /= np.sqrt(np.sum(np.square(subsequences), keepdims=True))
            elif subsequences.ndim == 2:
                subsequences /= np.sqrt(np.sum(np.square(subsequences), axis=1, keepdims=True))
            else:
                raise NotImplementedError("Unknown sequence shape")
        elif distance_option is DistanceOption.Euclidean:
            return subsequences
        else:
            raise NotImplementedError("Unknown distance yet")
        return subsequences

    @classmethod
    def _distance(cls, sq1: np.ndarray, sq2: np.ndarray, distance_option: DistanceOption):
        if distance_option is DistanceOption.Euclidean:
            return np.sqrt(np.sum(np.square(sq1 - sq2)))
        elif distance_option is DistanceOption.EuclideanNorm:
            return np.sqrt(np.sum(np.square(sq1 - sq2)))
        elif distance_option is DistanceOption.Cosine:
            return np.dot(sq1, sq2)
        else:
            raise NotImplementedError("Unknown distance yet")



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    import glob

    # file = '../../../../../datasets/TS4CD/Synthetic\\PeriodicSpike\\values.pkl'
    file = '../../../../../datasets/TS4CD/Synthetic\\Shift\\values.pkl'
    # file = '../../../../../datasets/TS4CD/Synthetic\\Gradual\\values.pkl'

    def plot_subgraph(data, s, e, ax):
        c_data = data[s:e]
        ret = MatrixProfile.matrix_profile_1d(c_data, distance_option=DistanceOption.Euclidean, window=10)
        mp, mpi, mp_left, mpi_left, mp_right, mpi_right = ret
        ac = AC.calc_ac(mpi, margin_ignore=3)
        iac = IAC.get_mpi_iac(len(mpi))
        cac = ac / iac
        cac[cac > 1] = 1.
        cac[-3:] = 1.
        cac[:3] = 1.
        ax.plot(np.arange(s, e)[:len(cac)], cac, label=f"[{s},{e}]")
    plt.figure(figsize=(10,10))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    data = pd.read_pickle(file)
    data = data.values[100:, 0]
    ax1.plot(data)
    plot_subgraph(data, 0, len(data), ax2)
    plot_subgraph(data, 0, 200, ax2)
    plot_subgraph(data, 200, 400, ax2)
    plot_subgraph(data, 350, 550, ax2)
    plot_subgraph(data, 400, 600, ax2)
    plot_subgraph(data, 600, 800, ax2)
    plt.ylabel("cac score")
    plt.legend()

    plt.savefig("mp_result1.png")



