from scipy.signal import find_peaks
from ....DataFactory.TimeSeries import TimeSeriesView
from ..BaseMethod4CD import BaseOfflineMethod4CD, BaseOnlineMethod4CD
from ... import MethodTestResults
from ....DataFactory.LabelStore import LabelStore, ChangePointLabel, ReportPointLabel, RunLengthLabel
import numpy as np
from ..utils.MatrixProfile import MatrixProfile
from scipy.ndimage import gaussian_filter
from collections import OrderedDict


class WCACState:
    _iac = dict()

    @classmethod
    def get_iac(cls, n: int):
        if n not in cls._iac:
            # compute the inverted parabola IAC curve with 1/2n height according to floss paper
            arr = - np.square(np.linspace(-n / 2, n / 2, n)) * 2 / n + n / 2
            arr[arr <= 0] = 1e-10
            cls._iac[n] = arr
        return cls._iac[n]

    def __init__(self, timeseries: np.ndarray, margin_ignore=3, m=5, smooth_sigma: float = None):
        self._L, self._D = timeseries.shape
        self._m = m
        self._margin_ignore = margin_ignore
        self._k = None
        self._smooth_sigma = float(smooth_sigma) if smooth_sigma is not None else None
        self._ts = timeseries
        self._update_count = 0
        self._collect_change_points()

    def get_L(self):
        return self._L

    def update(self, obs: np.ndarray):
        obs = obs.reshape((-1, self._D))
        L, D = obs.shape
        self._update_count += L
        if self._L >= L:
            self._ts[:L] = obs
            self._ts = np.roll(self._ts, -L, axis=0)
        else:
            self._ts = obs[-self._L:]
        self._collect_change_points()

    def get_change_points(self) -> np.ndarray:
        return self._change_points

    def _collect_change_points(self):
        candidates: list[np.ndarray] = []
        for d in range(self._D):
            MP, MPI, mp_left, mpi_left, mp_right, mpi_right = MatrixProfile.matrix_profile_1d(self._ts[:, d], self._m,
                                                                                              ignore_distance=1)
            # mp = stumpy.stump(self._ts[:, d], m=self._m, ignore_trivial=True)
            # MP: np.ndarray = mp[:, 0]
            # MPI: np.ndarray = mp[:, 1]
            WCAC_norm = self._extract_1d_wcac_norm(MP, MPI)
            if isinstance(self._smooth_sigma, float):
                WCAC_norm = gaussian_filter(WCAC_norm, sigma=self._smooth_sigma)
            change_point_indexes = self._find_local_minimals(WCAC_norm)
            candidates.append(change_point_indexes)
        if self._k is None:
            self._estimate_k(candidates)
        best_entropy, best_change_points = np.inf, []
        for candidate in candidates:
            change_points, entropy, entropies = self._greedy_entropy_seg(self._ts, candidate, self._k)
            if entropy < best_entropy:
                best_entropy = entropy
                best_change_points = change_points
        self._change_points: np.ndarray = np.array(sorted(best_change_points))

    def _extract_1d_wcac_norm(self, MP: np.ndarray, MPI: np.ndarray, extend_thresh: float = None):
        if extend_thresh is None:
            extend_thresh = MP.mean() * 3
        n = len(MP)
        ri = np.arange(n)
        iac = self.get_iac(n)
        arcs = {(s, e): MP[s] for s, e in zip(ri, MPI)}
        new_arcs = {(s, MPI[e]): d + MP[e] for (s, e), d in arcs.items() if
                    s != MPI[e] and d + MP[e] < extend_thresh and (s, MPI[e]) not in arcs}
        while len(new_arcs) > 0:
            arcs.update(new_arcs)
            new_arcs = {(s, MPI[e]): d + MP[e] for (s, e), d in new_arcs.items() if
                        s != MPI[e] and d + MP[e] < extend_thresh and (s, MPI[e]) not in arcs}
        extended_arcs = arcs

        e_arc_keys = np.array([list(key) for key in extended_arcs.keys()])
        e_arc_starts = np.minimum(e_arc_keys[:, 0], e_arc_keys[:, 1])
        e_arc_ends = np.maximum(e_arc_keys[:, 0], e_arc_keys[:, 1])
        e_weights = np.array([extended_arcs[key] for key in extended_arcs.keys()]) / (e_arc_ends - e_arc_starts)

        e_arc_increase_total_weights = np.bincount(e_arc_starts, weights=e_weights, minlength=n) - \
            np.bincount(e_arc_ends, weights=e_weights, minlength=n)
        WCAC_raw = np.cumsum(e_arc_increase_total_weights)
        WCAC_norm = WCAC_raw / iac * np.percentile(iac, 90) / np.percentile(WCAC_raw, 90)
        WCAC_norm[WCAC_norm > 1.] = 1.
        WCAC_norm[-self._margin_ignore:] = 1
        WCAC_norm[:self._margin_ignore] = 1
        return WCAC_norm

    def _find_local_minimals(self, arr_1d: np.ndarray) -> np.ndarray:
        peaks, properties = find_peaks(arr_1d, distance=self._m)
        return peaks

    def _estimate_k(self, candidates: list[np.ndarray]):
        k_range = [len(c) for c in candidates]
        min_k = int(np.min(k_range))
        max_k = int(np.max(k_range))
        k_range = list(range(min_k, max_k + 1))
        L = []

        all_candidate_entropies = []
        for dim, candidate in enumerate(candidates):
            change_points, entropy, entropies = self._greedy_entropy_seg(self._ts, change_points=candidate,
                                                                         k=len(candidate), fast_forward=False)
            all_candidate_entropies.append(entropies)

        for k in k_range:
            entropies_all_dims = [es[k] if k < len(es) else es[-1] for es in all_candidate_entropies]
            L.append(np.min(entropies_all_dims))

        L = np.array(L)

        if len(L) <= 3:
            self._k = k_range[np.argmin(L)]
        else:
            diff = (L[1:-1] - L[:-2]) / ((L[2:] - L[1:-1]) + 1e-10)
            self._k = k_range[np.argmax(diff) + 1]

    def _greedy_entropy_seg(self, timeseries: np.ndarray, change_points: np.ndarray, k: int,
                            fast_forward: bool = True):
        if k == 0:
            return change_points, 0, [0]
        if fast_forward and len(change_points) <= k:
            ig = self._ig_gain(timeseries, sorted(list(change_points))[:k])
            return change_points, ig, [ig]

        selected_change_points: list[int] = list()
        left_change_points = [c for c in change_points]
        best_entropy = 0
        best_entropies = []
        for i in range(k):
            if len(left_change_points) <= 0:
                break
            cp_scores = []
            for next_c in left_change_points:
                next_cp_candidate = selected_change_points + [next_c]
                cp_scores.append(self._ig_gain(timeseries, sorted(next_cp_candidate)))
            select_cp_index = np.argmin(cp_scores)
            best_entropy = cp_scores[select_cp_index]
            best_entropies.append(best_entropy)
            select_cp: int = left_change_points[select_cp_index]
            selected_change_points.append(select_cp)
            left_change_points.remove(select_cp)
        return selected_change_points, best_entropy, best_entropies

    def _ig_gain(self, timeseries: np.ndarray, split_indexes: list[int]):
        """
        Information Gain. Implemented as described in
            https://github.com/cruiseresearchgroup/IGTS-python and
            https://github.com/cruiseresearchgroup/ESPRESSO
        """
        if len(split_indexes) == 0:
            return 0
        assert timeseries.ndim == 2
        N, dims = timeseries.shape
        if split_indexes[0] != 0:
            split_indexes = [0] + split_indexes

        IG = 0
        for s, e in zip(split_indexes[:-1], split_indexes[1:]):
            dif_d_dim = timeseries[s, :] - timeseries[e, :]
            IG -= (e - s) * self._get_sha_entropy(np.abs(dif_d_dim))
        return IG / N

    @classmethod
    def _get_sha_entropy(cls, dif_d_dim: np.ndarray) -> float:
        dif_d_dim = dif_d_dim[(dif_d_dim != 0)]
        if len(dif_d_dim) > 0:
            p = np.true_divide(dif_d_dim, np.sum(dif_d_dim))
            return -1 * sum(p * np.log(p))
        else:
            return 0


class ESPRESSO(BaseOfflineMethod4CD):
    def offline_initialize(self):
        pass

    def offline_test(self, timeseries: TimeSeriesView) -> MethodTestResults:
        margin_ignore = self.hparams.get("margin_ignore", 5)
        window_size = self.hparams.get("window_size", 5)
        smooth_sigma = self.hparams.get("smooth_sigma", 1.0)

        wcac = WCACState(timeseries.get_values(), margin_ignore, m=window_size, smooth_sigma=smooth_sigma)

        change_points = wcac.get_change_points()
        return MethodTestResults(LabelStore([
            ChangePointLabel.from_point_list(list(change_points), sequence_length=timeseries.size(),
                                             annotator="ESPRESSO(CP)"),
            ReportPointLabel.from_point_list([timeseries.size() - 1] if len(change_points) > 0 else [],
                                             sequence_length=timeseries.size(),
                                             annotator="ESPRESSO(RP)"),
            RunLengthLabel.from_change_point_indexes(list(change_points),
                                                     seq_length=timeseries.size(),
                                                     annotator="ESPRESSO(RL)"),
        ]))

    @classmethod
    def _method_file_path(cls) -> str:
        return __file__


class ESPRESSOOnline(BaseOnlineMethod4CD):
    def online_initialize(self):
        self.online_state.initialize()
        self.online_state.update({
            "wcac": None,
            "need_retrain": True,
            "last_index": None,
            "run_lengths": None,
        })

    def online_retrain(self, timeseries: TimeSeriesView):
        margin_ignore = self.hparams.get("margin_ignore", 1)
        window_size = self.hparams.get("window_size", 5)
        smooth_sigma = self.hparams.get("smooth_sigma", 1.0)
        wcac = WCACState(timeseries.get_values(), margin_ignore, m=window_size, smooth_sigma=smooth_sigma)
        self.online_state.update({
            "wcac": wcac,
            "need_retrain": False,
            "last_index": timeseries.get_indexes()[-1],
            "run_lengths": OrderedDict(),
        })

    def need_retrain(self) -> bool:
        return self.online_state.custom_dict.get("need_retrain", True)

    def online_step(self, timeseries: TimeSeriesView):
        self.online_state.record_index(timeseries.get_indexes())
        wcac: WCACState = self.online_state.custom_dict.get("wcac", None)
        last_index: int = self.online_state.custom_dict.get("last_index", None)
        end_index: int = timeseries.end_index
        if last_index < end_index:
            wcac.update(timeseries.get_values()[-(end_index - last_index):])
            self.online_state.update({
                "last_index": end_index,
            })
            change_points = wcac.get_change_points()
            if len(change_points) > 0:
                self.online_state.custom_dict["run_lengths"][end_index - 1] = end_index - 1 - change_points[-1]

    def online_collect_results(self) -> MethodTestResults:
        online_run_lengths: OrderedDict = self.online_state.custom_dict["run_lengths"]

        indexes = np.array(sorted(list(self.online_state.test_indexes)), dtype=np.int32)
        start_index = int(np.min(indexes))

        run_length_values = np.arange(len(indexes))

        for end_index, run_length in online_run_lengths.items():
            run_length_values[end_index - start_index:] -= run_length_values[end_index - start_index] - run_length
        run_length_label = RunLengthLabel(run_length_values, annotator="ESPRESSOOnline(RL)")

        report_point_indexes = np.where(run_length_values[1:] - run_length_values[:-1] < -1)[0] + 1
        report_point_label = ReportPointLabel.from_point_list(list(report_point_indexes), len(indexes),
                                                              annotator="ESPRESSOOnline(RP)")

        change_point_indexes = report_point_indexes - run_length_values[report_point_indexes]
        change_point_indexes[change_point_indexes < 0] = 0
        change_point_label = ChangePointLabel.from_point_list(change_point_indexes, len(indexes),
                                                              annotator="ESPRESSOOnline(CP)")

        label_store = LabelStore([change_point_label, run_length_label, report_point_label])
        self.test_results = MethodTestResults(label_store)
        return self.test_results

    @classmethod
    def _method_file_path(cls) -> str:
        return __file__
