import numpy as np
import ruptures as rpt

from ..BaseMethod4CD import BaseOfflineMethod4CD
from ....DataFactory.LabelStore import ChangePointLabel, ReportPointLabel, RunLengthLabel, LabelStore
from ....DataFactory.TimeSeries import TimeSeriesView
from ....Methods import MethodTestResults


class PELT(BaseOfflineMethod4CD):
    """
    implementation source: https://github.com/deepcharles/ruptures
    """

    def offline_initialize(self):
        pass

    def offline_test(self, timeseries: TimeSeriesView) -> MethodTestResults:
        model = self.hparams.get("model", "rbf")
        pen = self.hparams.get("pen", 10)
        ndim = timeseries.get_dim()
        timeseries_values: np.ndarray = timeseries.get_values().reshape((-1, ndim))
        algo = rpt.Pelt(model=model).fit(timeseries_values)
        change_point_indexes = algo.predict(pen=pen)[:-1]  # drop last point
        change_point_label = ChangePointLabel.from_point_list(change_point_indexes,
                                                              sequence_length=len(timeseries_values),
                                                              annotator="PELT(CP)")
        report_point_label = ReportPointLabel.from_point_list([timeseries.size() - 1] if len(change_point_indexes) > 0
                                                              else [],
                                                              sequence_length=len(timeseries_values),
                                                              annotator="PELT(RP)")
        run_length_label = RunLengthLabel.from_change_point_indexes(change_point_indexes,
                                                                    seq_length=len(timeseries_values),
                                                                    annotator="PELT(RL)")

        self.test_results = MethodTestResults(LabelStore([
            change_point_label,
            report_point_label,
            run_length_label
        ]))
        return self.test_results

    @classmethod
    def _method_file_path(cls) -> str:
        return __file__


