import numpy as np

from ....DataFactory.TimeSeries import TimeSeriesView
from ..BaseMethod4CD import BaseOnlineMethod4CD
from ... import MethodTestResults
from ....DataFactory.LabelStore import LabelStore
from ....DataFactory.LabelStore.Label import ChangePointLabel, RunLengthLabel, ReportPointLabel
from TSCDD.Methods.ConceptDriftDetection.utils.BOCPDStream import BOCPDStream, StudentTProb1d


class BOCD(BaseOnlineMethod4CD):
    """
    code source: https://github.com/y-bar/bocd/
    """

    def online_initialize(self):
        self.online_state.initialize()
        self.online_state.custom_dict["detectors"] = None
        self.online_state.custom_dict["mle"] = None

    def online_retrain(self, timeseries: TimeSeriesView):
        D = timeseries.get_dim()
        detectors = []
        for i in range(D):
            detector = BOCPDStream(StudentTProb1d())
            detectors.append(detector)
        self.online_state.custom_dict["detectors"] = detectors
        self.online_state.custom_dict["rt_mle"] = [[np.ones(1)*np.inf] for i in range(D)]

    def need_retrain(self) -> bool:
        return self.online_state.custom_dict["detectors"] is None

    def online_step(self, timeseries: TimeSeriesView):
        indexes = timeseries.get_indexes()
        self.online_state.test_indexes |= set(indexes)

        D = timeseries.get_dim()
        detectors: list[BOCPDStream] = self.online_state.custom_dict["detectors"]
        rt_mle: list[list] = self.online_state.custom_dict["rt_mle"]
        assert len(detectors) == len(rt_mle) == D
        data = timeseries.get_values().reshape((-1, D))
        for row in range(data.shape[0]):
            value = data[row, :]
            for dim in range(D):
                detectors[dim].update(value[dim])
                rt_mle[dim].append(detectors[dim].get_rt())

    def online_collect_results(self) -> MethodTestResults:
        indexes = np.array(sorted(list(self.online_state.test_indexes)), dtype=np.int32)
        rt_mle: list[list] = self.online_state.custom_dict["rt_mle"]
        rt_mle: list[np.ndarray] = [np.array(e).flatten() for e in rt_mle]
        label_store = LabelStore()
        for dim, rt in enumerate(rt_mle):

            report_run_lengths = np.zeros_like(indexes, dtype=np.float64)
            report_run_lengths[-len(indexes):] = rt[-len(indexes):]
            run_length_label = RunLengthLabel(report_run_lengths, annotator="BOCD(RL)")

            index_diff_changes, = np.where(np.diff(rt) < -1)
            report_point_label = ReportPointLabel.from_point_list(index_diff_changes, len(indexes), annotator="BOCD(RP)")
            index_change_points = [int(e+1-rt[e+1]) for e in index_diff_changes]
            change_point_label = ChangePointLabel.from_point_list(index_change_points, len(indexes),
                                                                  annotator="BOCD(CP)")

            label_store.set_label(run_length_label)
            label_store.set_label(change_point_label)
            label_store.set_label(report_point_label)

        self.test_results = MethodTestResults(label_store)
        return self.test_results

    @classmethod
    def _method_file_path(cls) -> str:
        return __file__
