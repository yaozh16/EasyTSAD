import numpy as np

from ....DataFactory.TimeSeries import TimeSeriesView
from ....Methods.ConceptDriftDetection.BaseMethod4CD import BaseOnlineMethod4CD
from ... import MethodTestResults
from ....DataFactory.LabelStore.Label import ChangePointLabel, ReportPointLabel, RunLengthLabel
from ....DataFactory.LabelStore import LabelStore


class MSSAState:
    def __init__(self, lag: int, T: int, view: TimeSeriesView, test_view: TimeSeriesView):
        assert lag <= T and T % lag == 0, f"MSSA state initialization failed: [lag] {lag} [T] {T}"
        assert view.size() == T, f"MSSA state initialization failed: [view size] {view.size()} [T] {T}"
        self._L = lag
        self._T = T
        self._M = int(T / lag)
        self._U = self._construct_space(view)
        self._U_prj_mtx = self._U @ self._U.T  # shape is (L, L)
        self._L, self._K = self._U.shape
        self._c = 0
        slide_view = test_view.safe_slice(0, self._L)
        test_scores = []
        for i in range(test_view.size() - self._L + 1):
            test_scores.append(self._get_distance_score(slide_view))
            slide_view.step(1)
        max_distance = np.max(test_scores)
        self._c = max_distance
        self._h = 5 * max_distance
        self._y_last = 0

    def get_lag(self) -> int:
        return self._L

    def _construct_space(self, view: TimeSeriesView) -> np.ndarray:
        Zs = []
        ndim = view.get_dim()
        obs = view.get_values().reshape((-1, ndim))
        assert obs.shape == (self._T, ndim)
        for dim in range(ndim):
            Zs.append(np.vstack([
                obs[i:self._T:self._L, dim] for i in range(self._L)
            ]))
        Zs = np.hstack(Zs)
        assert Zs.shape == (self._L, self._M * ndim)
        u, sigma, vt = np.linalg.svd(Zs)
        u = u[:, :len(sigma)]
        cs = np.square(sigma).cumsum()
        valid = cs >= cs.max() * 0.95
        k = 0
        while k < len(sigma):
            if valid[k]:
                break
            k += 1

        u = u[:, :k+1]
        return u

    def _get_distance_score(self, view: TimeSeriesView):
        ndim = view.get_dim()
        obs: np.ndarray = view.get_values().reshape((-1, ndim))
        assert obs.shape == (self._L, ndim), f"SSA test matrix shape should be ({self._L}, {ndim}) currently {obs.shape}"
        distance = np.sum(np.square(obs.T - obs.T @ self._U_prj_mtx)) - self._c
        return distance

    def test_whether_change(self, view: TimeSeriesView) -> bool:
        assert view.size() == self.get_lag()
        d_score = self._get_distance_score(view)
        self._y_last = max(0, self._y_last + d_score)
        return self._y_last > self._h


class MSSA(BaseOnlineMethod4CD):

    def __init__(self, hparam: dict = None):
        super().__init__(hparams=hparam)
        self._lag: int = hparam.get("lag", None)
        self._T0m: int = hparam.get("T0m", None)

    def online_initialize(self):
        self.online_state.initialize()
        self.online_state.custom_dict["report_point_indexes"] = list()
        self.online_state.custom_dict["need_retrain"] = True
        self.online_state.custom_dict["retrain_on"] = True
        self.online_state.custom_dict["test_ts_view"] = None

    def online_retrain(self, timeseries: TimeSeriesView):
        T0 = timeseries.size()
        N = timeseries.get_dim()
        if self._lag is None:
            self._lag: int = int(np.floor(np.sqrt(min(N, T0) * T0)))
            self._lag: int = int(min(np.sqrt(T0), self._lag))
        self._T0m: int = int((T0 // self._lag) * self._lag)
        c_timeseries = timeseries.head(self._T0m)
        if self._T0m == 0 or c_timeseries.size() < self._T0m:
            self.online_state.update({"need_retrain": False, "retrain_on": False})
            self.logger.info("[SSA] retrain data insufficient, stop retrain.")
            return
        self.online_state.custom_dict["mssa_state"] = MSSAState(self._lag, self._T0m, c_timeseries, timeseries)
        self.online_state.update({"need_retrain": False})

    def need_retrain(self) -> bool:

        return self.online_state.custom_dict.get("need_retrain", True) & \
            self.online_state.custom_dict.get("retrain_on", True)

    def online_step(self, timeseries: TimeSeriesView):
        self.online_state.record_index(index_arr=timeseries.get_indexes())
        test_ts_view: TimeSeriesView = self.online_state.custom_dict.get("test_ts_view", None)
        mssa_state: MSSAState = self.online_state.custom_dict.get("mssa_state", None)
        assert mssa_state is not None, "Please retrain to construct mssa state first."
        if test_ts_view is None:
            test_ts_view = timeseries.last(self._lag)
        else:
            test_ts_view = test_ts_view.expand_end_with(timeseries, size_upperbound=self._lag)
        self.online_state.custom_dict["test_ts_view"] = test_ts_view
        if test_ts_view.size() < self._lag:
            return
        elif test_ts_view.size() > self._lag:
            test_ts_view.crop_size(self._lag, align_right=True)
        if mssa_state.test_whether_change(view=test_ts_view):
            self.online_state.custom_dict["need_retrain"] = True
            self.online_state.custom_dict["report_point_indexes"].append(timeseries.end_index-1)

    def online_collect_results(self) -> MethodTestResults:
        indexes = np.array(sorted(list(self.online_state.test_indexes)), dtype=np.int32)
        report_point_indexes = np.array(self.online_state.custom_dict["report_point_indexes"], dtype=np.int32)
        report_point_indexes = report_point_indexes - np.min(indexes)

        report_point_mask = np.zeros_like(indexes)
        report_point_mask[report_point_indexes] = 1

        change_point_mask = np.roll(report_point_mask, -self._lag)
        change_point_mask[-self._lag:] = 0

        label_store = LabelStore()
        label_store.set_label(ReportPointLabel(report_point_mask, annotator="MSSA(RP)"))
        label_store.set_label(ChangePointLabel(change_point_mask, annotator="MSSA(CP)"))
        label_store.set_label(RunLengthLabel.from_change_point_indexes(report_point_indexes - self._lag, len(indexes),
                                                                       annotator="MSSA(RL)"))
        self.test_results = MethodTestResults(label_store)
        return self.test_results

    @classmethod
    def _method_file_path(cls) -> str:
        return __file__
