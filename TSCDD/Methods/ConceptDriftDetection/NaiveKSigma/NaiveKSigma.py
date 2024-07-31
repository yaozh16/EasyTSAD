import numpy as np

from TSCDD.DataFactory.TimeSeries import TimeSeriesView
from TSCDD.Methods.ConceptDriftDetection.BaseMethod4CD import BaseOnlineMethod4CD
from ... import MethodTestResults
from ....DataFactory.LabelStore import LabelStore
from ....DataFactory.LabelStore.Label import ChangePointLabel, ReportPointLabel, RunLengthLabel


class NaiveKSigma(BaseOnlineMethod4CD):
    @classmethod
    def _method_file_path(cls) -> str:
        return __file__

    def __init__(self, hparam: dict = None):
        super().__init__(hparams=hparam)
        self._k = self.hparams.get("K", 3)

    def online_initialize(self):
        self.online_state.initialize()
        self.online_state.custom_dict["change_point_indexes"] = list()
        self.online_state.custom_dict["need_retrain"] = True

    def online_retrain(self, timeseries: TimeSeriesView):
        n_dim = timeseries.get_dim()
        indexes = timeseries.get_indexes()
        self.online_state.test_indexes |= set(indexes)
        observations = timeseries.get_values().reshape((-1, n_dim))
        mean = np.mean(observations, axis=0)
        std = np.std(observations, axis=0)
        self.online_state.update({"n_dim": n_dim,
                                  "mean": mean,
                                  "std": std,
                                  "ub": mean + std * self._k,
                                  "lb": mean - std * self._k,
                                  "last_train": timeseries.end_index,
                                  })
        self.online_state.update({"need_retrain": False})

    def need_retrain(self) -> bool:
        return self.online_state.custom_dict.get("need_retrain", True)

    def online_step(self, timeseries: TimeSeriesView):
        n_dim = timeseries.get_dim()
        indexes = timeseries.get_indexes()
        self.online_state.test_indexes |= set(indexes)
        observations = timeseries.get_values().reshape((-1, n_dim))
        ub = self.online_state.custom_dict["ub"]
        lb = self.online_state.custom_dict["lb"]
        change_mask = np.any(observations > ub, axis=1) | np.any(observations < lb, axis=1)
        change_indexes = indexes[np.where(change_mask)[0]]
        if len(change_indexes) > 0:
            self.online_state.custom_dict["need_retrain"] = True
            self.online_state.custom_dict["change_point_indexes"].append(change_indexes[0])
        return change_mask

    def online_collect_results(self) -> MethodTestResults:
        indexes = np.array(sorted(list(self.online_state.test_indexes)), dtype=np.int32)
        change_point_indexes = np.array(self.online_state.custom_dict["change_point_indexes"], dtype=np.int32)

        change_point_indexes -= np.min(indexes)
        change_point_mask = np.zeros_like(indexes)
        change_point_mask[change_point_indexes] = 1
        label_store = LabelStore([
            ChangePointLabel(change_point_mask, annotator="NaiveKSigma(CP)"),
            ReportPointLabel(change_point_mask, annotator="NaiveKSigma(RP)"),
            RunLengthLabel.from_change_point_indexes(change_point_indexes, len(indexes), annotator="NaiveKSigma(RL)"),
        ])
        self.test_results = MethodTestResults(label_store)
        return self.test_results


