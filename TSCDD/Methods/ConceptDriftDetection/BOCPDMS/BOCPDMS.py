from ....DataFactory.TimeSeries import TimeSeriesView
from .bocpdms.cp_probability_model import CpModel
from .bocpdms.BVAR_NIG import BVARNIG
from .bocpdms.detector import Detector
import numpy as np

from ..BaseMethod4CD import BaseOnlineMethod4CD
from ... import MethodTestResults
from ....DataFactory.LabelStore.Label import ChangePointLabel, RunLengthLabel, ReportPointLabel
from ....DataFactory.LabelStore import LabelStore
from ....DataFactory.TimeSeries.Preprocess import Preprocessor
from ....Config.Options import PreprocessOptions


class BOCPDMS(BaseOnlineMethod4CD):
    def online_initialize(self):
        self.online_state.initialize()
        self.online_state.custom_dict["detector"] = None

    def online_retrain(self, timeseries: TimeSeriesView):
        S1 = timeseries.get_dim()
        S2 = 1
        nT = len(timeseries)

        intensity = self.hparams.get("intensity", 100)
        cp_model = CpModel(intensity)
        prior_a = self.hparams.get("prior_a", 1.)
        prior_b = self.hparams.get("prior_b", 1.)
        prior_mean_scale = self.hparams.get("prior_mean_scale", 0.)
        prior_var_scale = self.hparams.get("prior_var_scale", 0.075)
        alpha_param_opt_t = self.hparams.get("alpha_param_opt_t", 30)
        alpha_param = self.hparams.get("alpha_param", 0.01)
        threshold = self.hparams.get("threshold", 50)
        trim_type = self.hparams.get("trim_type", "KeepK")

        AR_lags = self.hparams.get("AR_lags", [1, 2, 3, 4, 5])
        AR_models = []

        for lag in AR_lags:
            """Generate next model object"""
            AR_models += [BVARNIG(
                prior_a=prior_a, prior_b=prior_b,
                S1=S1, S2=S2,
                prior_mean_scale=prior_mean_scale,
                prior_var_scale=prior_var_scale,
                intercept_grouping=None,
                nbh_sequence=[0] * lag,
                restriction_sequence=[0] * lag,
                hyperparameter_optimization="online")]

        model_universe = np.array(AR_models)
        model_prior = np.array([1 / len(model_universe)] * len(model_universe))
        self.online_state.custom_dict["detector"] = Detector(
            data=timeseries.get_values(),
            model_universe=model_universe,
            model_prior=model_prior,
            cp_model=cp_model,
            S1=S1, S2=S2, T=nT,
            store_rl=True, store_mrl=True,
            trim_type=trim_type,
            threshold=threshold,
            save_performance_indicators=True,
            generalized_bayes_rld="kullback_leibler",
            alpha_param_learning="individual",
            alpha_param=alpha_param,
            alpha_param_opt_t=alpha_param_opt_t,
            alpha_rld_learning=True,
            loss_der_rld_learning="squared_loss",
            loss_param_learning="squared_loss")

    def need_retrain(self) -> bool:
        return self.online_state.custom_dict["detector"] is None

    def online_step(self, timeseries: TimeSeriesView):
        indexes = timeseries.get_indexes()
        self.online_state.record_index(indexes)

        S1 = timeseries.get_dim()
        detector: Detector = self.online_state.custom_dict["detector"]
        data = timeseries.get_values().reshape((-1, S1))
        if len(self.online_state.test_indexes) > 10:
            exit(0)
        else:
            print(detector.all_retained_run_lengths)
        for row in range(data.shape[0]):
            value = data[row, :].reshape((1, S1))
            detector.next_run(y=value, t=int(row + min(indexes) - min(self.online_state.test_indexes)))

    def online_collect_results(self) -> MethodTestResults:
        indexes = np.array(sorted(list(self.online_state.test_indexes)), dtype=np.int32)
        detector: Detector = self.online_state.custom_dict["detector"]

        change_points = detector.CPs
        last_change_point_indexes = [cp[0] for cp in change_points[len(indexes)-2]]

        change_point_indexes = np.array(last_change_point_indexes, dtype=np.int32)
        change_point_indexes[0] = 0  # default is 3
        change_point_mask = np.zeros_like(indexes)
        change_point_mask[change_point_indexes - np.min(indexes)] = 1
        change_point_label = ChangePointLabel(change_point_mask, annotator="BOCPDMS(CP-final)")

        run_lengths = [i + 1 - r[-1][0] if len(r) > 0 else None for i, r in enumerate(detector.CPs)]
        # process the head and tail nan
        run_lengths = np.array(run_lengths, dtype=np.float64)
        nan_mask = np.isnan(run_lengths)
        valid_start, = np.where(~nan_mask & np.roll(nan_mask, 1))
        valid_end, = np.where(nan_mask & ~np.roll(nan_mask, 1))
        valid_start = valid_start[0]
        valid_end = valid_end[-1]
        run_lengths = run_lengths[valid_start: valid_end].reshape((-1, 1))
        run_lengths = Preprocessor.process(run_lengths, options=[PreprocessOptions.FillNAWithInterp]).flatten()

        report_run_lengths = np.zeros_like(indexes, dtype=np.float64)
        report_run_lengths[-len(run_lengths):] = run_lengths
        run_length_label = RunLengthLabel(report_run_lengths, annotator="BOCPDMS(RL)")

        report_indexes, = np.where((report_run_lengths[1:] - report_run_lengths[:-1]) < -1)
        report_point_label = ReportPointLabel.from_point_list(report_indexes, sequence_length=len(indexes),
                                                              annotator="BOCPDMS(RP)")

        label_store = LabelStore([change_point_label, run_length_label, report_point_label])
        self.test_results = MethodTestResults(label_store)
        return self.test_results

    @classmethod
    def _method_file_path(cls) -> str:
        return __file__
