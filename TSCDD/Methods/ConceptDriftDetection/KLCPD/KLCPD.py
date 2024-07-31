import numpy as np
from ....DataFactory.LabelStore import ChangePointLabel, LabelStore, ReportPointLabel, RunLengthLabel
from ....DataFactory.TimeSeries import TimeSeriesView
from ....Methods import MethodTestResults
from .klcpd import KL_CPD
from ..BaseMethod4CD import BaseOfflineMethod4CD
from .ThresholdSelector import ThresholdSelector


class KLCPD(BaseOfflineMethod4CD):
    def offline_initialize(self):
        pass

    def offline_test(self, timeseries: TimeSeriesView) -> MethodTestResults:
        window_size = self.hparams.get("window_size", 5)
        label_store = LabelStore()
        ndim = timeseries.get_dim()
        timeseries_values: np.ndarray = timeseries.get_values().reshape((-1, ndim))

        model = KL_CPD(D=timeseries.get_dim(), p_wnd_dim=window_size, f_wnd_dim=window_size, sub_dim=window_size)
        model.fit(timeseries_values, epoches=100)
        change_scores = model.predict(timeseries_values)

        thresh = ThresholdSelector.k_sigma_value(change_scores, 3)
        report_point_indexes = np.where(change_scores > thresh)[0]

        label_store.set_label(ReportPointLabel.from_point_list(report_point_indexes, timeseries.size(), "KLCPD(RP)"))

        run_length_values = np.arange(timeseries.size())
        for r_id in report_point_indexes:
            r_id = int(r_id)
            run_length_values[r_id:] -= run_length_values[r_id] - window_size
        label_store.set_label(RunLengthLabel(run_length_values, "KLCPD(RL)"))

        change_point_indexes = report_point_indexes - run_length_values[report_point_indexes]
        change_point_indexes[change_point_indexes < 0] = 0
        label_store.set_label(ChangePointLabel.from_point_list(change_point_indexes, timeseries.size(), "KLCPD(CP)"))

        self.test_results = MethodTestResults(label_store)
        return self.test_results

    @classmethod
    def _method_file_path(cls) -> str:
        return __file__

    @classmethod
    def internal_online(cls) -> bool:
        return True

