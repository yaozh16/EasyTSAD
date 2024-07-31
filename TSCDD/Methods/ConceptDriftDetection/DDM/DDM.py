from ....DataFactory.TimeSeries import TimeSeriesView
from ....Methods.ConceptDriftDetection.BaseMethod4CD import BaseOnlineMethod4CD
from ... import MethodTestResults
from ....DataFactory.LabelStore.Label import ChangePointLabel, ReportPointLabel, RunLengthLabel
from ....DataFactory.LabelStore import LabelStore
from collections import OrderedDict
from sklearn.svm import OneClassSVM
import numpy as np
from river.drift.binary import DDM as river_ddm


class DDM(BaseOnlineMethod4CD):

    def __init__(self, hparam: dict = None):
        super().__init__(hparams=hparam)
        warm_start = self.hparams.get("warm_start", 3)
        drift_threshold = self.hparams.get("drift_threshold", 3)
        ocs_params = self.hparams.get("ocs_params", dict())
        self.classifier = OneClassSVM(**ocs_params)
        self.model = river_ddm(warm_start, drift_threshold=drift_threshold)

    def online_initialize(self):
        self.online_state.initialize()
        self.online_state.update({
            "report_points": [],
            "run_lengths": [],
        })

    def online_retrain(self, timeseries: TimeSeriesView):
        # train with timeseries...
        self.classifier.fit(timeseries.get_values())
        self.online_state.need_retrain = False

    def need_retrain(self) -> bool:
        return self.online_state.need_retrain

    def online_step(self, timeseries: TimeSeriesView):
        self.online_state.record_index(index_arr=timeseries.get_indexes())
        dim = timeseries.get_dim()
        binary_seq = self.classifier.predict(timeseries.get_values().reshape(-1, dim))
        binary_seq = (binary_seq + 1) / 2  # for {-1,1} to {0, 1}
        for e in binary_seq:
            self.model.update(e)
        if self.model.drift_detected:
            self.online_state.custom_dict["report_points"].append(timeseries.end_index - 1)
            self.online_state.custom_dict["run_lengths"].append(0)
            self.online_state.need_retrain = True

    def online_collect_results(self) -> MethodTestResults:
        indexes = np.array(sorted(list(self.online_state.test_indexes)), dtype=np.int32)

        label_store = LabelStore()

        report_point_indexes = np.array(self.online_state.custom_dict["report_points"]) - np.min(indexes)

        label_store.set_label(ReportPointLabel.from_point_list(report_point_indexes, len(indexes), "DDM(RP)"))

        run_length_values = np.arange(len(indexes))
        run_lengths = np.array(self.online_state.custom_dict["run_lengths"])
        for rpi, rl in zip(report_point_indexes, run_lengths):
            run_length_values[rpi:] -= run_length_values[rpi] - rl

        label_store.set_label(RunLengthLabel(run_length_values, "DDM(RL)"))

        change_point_indexes = np.array((report_point_indexes - run_lengths) // 1, dtype=np.int32)

        label_store.set_label(ChangePointLabel.from_point_list(change_point_indexes, len(indexes), "DDM(CP)"))

        self.test_results = MethodTestResults(label_store)
        return self.test_results

    @classmethod
    def _method_file_path(cls) -> str:
        return __file__
