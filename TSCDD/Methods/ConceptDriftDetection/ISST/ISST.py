import numpy as np
from scipy.sparse.linalg import svds
from TSCDD.DataFactory.TimeSeries import TimeSeriesView
from ..BaseMethod4CD import BaseOnlineMethod4CD
from ... import MethodTestResults
from ....DataFactory.LabelStore import ChangePointLabel, LabelStore, RunLengthLabel, ReportPointLabel
from scipy.stats import genextreme


class SSAState:
    _epsilon = 1e-20

    def __init__(self, omega: int, timeseries: np.ndarray, eta: int, crop_delta: bool = True,
                 threshold_setter_callback=lambda scores: np.max(scores), thresh_backup: float = 0
                 ):
        assert timeseries.ndim == 1, "SSA timeseries should be 1d"
        N = len(timeseries)
        delta = N - 1 - omega
        assert delta >= eta, "SSA delta should be no less than eta"
        assert delta >= omega, "SSA delta should be no less than omega"
        self._eta: int = eta
        self._k: int = 2 * eta - 1 if eta % 2 == 1 else 2 * eta
        self._omega: int = omega
        self._delta: int = delta if not crop_delta else omega
        self._window_size = self._omega + self._delta - 1
        assert self._k < self._delta, f"SSA k[{self._k}] should be less than delta[{self._delta}]"
        assert self._k < self._omega, f"SSA k[{self._k}] should be less than omega[{self._omega}]"
        self._us, self._sigmas = self._construct_subspace(timeseries)
        self._A = None
        self._last_test_score = None
        self._train_change_scores = []
        tem_A = None
        for i in range(self._window_size, len(timeseries) - self._window_size):
            pre = timeseries[i - self._window_size:i]
            post = timeseries[i:i + self._window_size]
            if tem_A is None:
                tem_A = np.vstack([timeseries[i:i + self._delta] for i in range(self._omega)])
            else:
                tem_A[:, 0] = post[-self._omega:]
                tem_A = np.roll(tem_A, -1, axis=0)  # shape (omega, delta)
            score = self._slide_test_with_A(tem_A, pre, post)
            self._train_change_scores.append(score)
        if len(self._train_change_scores) > 1:
            self._threshold: float = threshold_setter_callback(self._train_change_scores)
        else:
            self._threshold: float = thresh_backup

    def get_threshold(self) -> float:
        return self._threshold

    def get_window_size(self) -> int:
        return self._window_size

    def _construct_subspace(self, timeseries: np.ndarray):
        B = np.vstack([timeseries[i:i + self._delta] for i in range(self._omega)])
        u, s, vt = svds(B, k=self._k)
        selected_us = u[:, -self._eta:]  # shape: (omega, eta)
        selected_sigmas = s[-self._eta:]
        return selected_us, selected_sigmas

    def _calc_stat(self, timeseries_1d: np.ndarray):
        med = np.median(timeseries_1d)
        madn = np.median(np.absolute(timeseries_1d - med)) / 1.4826
        return med, madn

    def slide_test(self, pre: np.ndarray, post: np.ndarray):
        assert pre.ndim == 1 and post.ndim == 1, f"Pre & Post timeseries should be 1d"
        assert len(pre) == self._window_size, f"Pre timeseries size should be {self._window_size}"
        assert len(post) == self._window_size, f"Post timeseries size should be {self._window_size}"
        if self._A is None:
            self._A = np.vstack([post[i:i + self._delta] for i in range(self._omega)])
        else:
            self._A[:, 0] = post[-self._omega:]
            self._A = np.roll(self._A, -1, axis=0)  # shape (omega, delta)
        return self._slide_test_with_A(self._A, pre, post) > self._threshold

    def _slide_test_with_A(self, A: np.ndarray, pre: np.ndarray, post: np.ndarray) -> float:
        AAT = A @ A.T
        u, lmd, vt = svds(AAT, k=self._k)
        beta = u[:, -self._eta:]
        phi = 1 - np.sum(np.dot(beta.T, self._us), axis=1)
        x_hat = np.average(phi, weights=lmd[-self._eta:])
        med_pre, madn_pre = self._calc_stat(pre)
        med_post, madn_post = self._calc_stat(post)
        self._last_test_score = x_hat * np.abs(med_pre - med_post) * np.abs(madn_pre - madn_post)
        return self._last_test_score


def get_extreme_threshold(data, pct=0.95):
    params = genextreme.fit(data)
    gev_dist = genextreme(*params)
    threshold = gev_dist.ppf(pct)
    return threshold


class ISSTMethod(BaseOnlineMethod4CD):
    def __init__(self, hparams: dict = None, threshold_setter_callback=lambda x: np.max(x)):
        super().__init__(hparams)
        self._window_size = None
        self._threshold_setter_callback = threshold_setter_callback

    def online_initialize(self):
        self.online_state.initialize()
        self.online_state.update({"ssa_states": None,
                                  "need_retrain": True,
                                  "retrain_on": True,
                                  "past_timeseries": None,
                                  "window_size": 1,
                                  "change_point_indexes": [],
                                  })
        self._window_size = None

    def online_retrain(self, timeseries: TimeSeriesView):
        omega = self.hparams.get("omega", 5)
        eta = self.hparams.get("eta", 3)
        crop_delta = self.hparams.get("crop_delta", True)
        ssa_states: list[SSAState] = []
        values = timeseries.get_values()
        old_states: list[SSAState] = self.online_state.custom_dict.get("ssa_states", None)
        L = len(values)
        if L < omega * 2 + 2:
            self.online_state.custom_dict["need_retrain"] = False
            self.online_state.custom_dict["retrain_on"] = False
            return
        for dim in range(timeseries.get_dim()):
            train_set = values[:, dim]
            thresh_backup = 0 if old_states is None else old_states[dim].get_threshold()
            ssa_states.append(SSAState(omega, train_set, eta, crop_delta,
                                       threshold_setter_callback=self._threshold_setter_callback,
                                       thresh_backup=thresh_backup))
        self._window_size = ssa_states[-1].get_window_size()
        self.online_state.custom_dict["ssa_states"] = ssa_states
        self.online_state.custom_dict["need_retrain"] = False

    def need_retrain(self) -> bool:
        return self.online_state.custom_dict.get("need_retrain", True) & \
            self.online_state.custom_dict.get("retrain_on", True)

    def online_step(self, timeseries: TimeSeriesView):
        self.online_state.record_index(index_arr=timeseries.get_indexes())
        ssa_states: list[SSAState] = self.online_state.custom_dict.get("ssa_states", None)
        assert ssa_states is not None, "Please train before online test"
        past_timeseries = self.online_state.custom_dict.get("past_timeseries", None)
        if past_timeseries is None:
            past_timeseries = timeseries.last(self._window_size * 2)
        if isinstance(past_timeseries, TimeSeriesView):
            past_timeseries.expand_end_with(timeseries, size_upperbound=self._window_size * 2)
            self.online_state.custom_dict["past_timeseries"] = past_timeseries
            if past_timeseries.size() > self._window_size:
                test_timeseries = past_timeseries.get_values()
                dims = past_timeseries.get_dim()
                change_point_mask = False
                for dim in range(dims):
                    pre = test_timeseries[:self._window_size, dim]
                    post = test_timeseries[-self._window_size:, dim]
                    if ssa_states[dim].slide_test(pre, post):
                        change_point_mask = True
                        break
                if change_point_mask:
                    self.online_state.custom_dict["need_retrain"] = True
                    self.online_state.custom_dict["change_point_indexes"].append(timeseries.end_index - 1)

    def online_collect_results(self) -> MethodTestResults:
        indexes = np.array(sorted(list(self.online_state.test_indexes)), dtype=np.int32)
        change_point_indexes = np.array(self.online_state.custom_dict["change_point_indexes"], dtype=np.int32)
        change_point_indexes -= int(np.min(indexes))

        run_lengths = np.arange(len(indexes))
        for idx in change_point_indexes:
            run_lengths[idx:] -= run_lengths[idx] - self._window_size
        run_length_label = RunLengthLabel(run_lengths, annotator="iSST(RL)")

        report_point_label = ReportPointLabel.from_point_list(list(change_point_indexes), sequence_length=len(indexes),
                                                              annotator="iSST(RP)")

        change_point_indexes -= self._window_size

        change_point_mask = np.zeros_like(indexes)
        change_point_mask[change_point_indexes] = 1
        change_point_label = ChangePointLabel.from_point_list(list(change_point_indexes), sequence_length=len(indexes),
                                                              annotator="iSST(CP)")

        label_store = LabelStore([run_length_label, change_point_label, report_point_label])
        self.test_results = MethodTestResults(label_store)
        return self.test_results

    @classmethod
    def _method_file_path(cls) -> str:
        return __file__


class ISSTMax(ISSTMethod):
    def __init__(self, hparams: dict = None):
        super().__init__(hparams, lambda x: np.max(x))


class ISSTEvt(ISSTMethod):
    def __init__(self, hparams: dict = None):
        evt_pct = hparams.get("evt_pct", 0.95)
        super().__init__(hparams, lambda scores: get_extreme_threshold(scores, pct=evt_pct))
