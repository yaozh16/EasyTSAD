import numpy as np

from ....DataFactory.TimeSeries import TimeSeriesView
from ....Methods.ConceptDriftDetection.BaseMethod4CD import BaseOfflineMethod4CD
from ... import MethodTestResults
from ....DataFactory.LabelStore.Label import ChangePointLabel, ReportPointLabel, RunLengthLabel
from ....DataFactory.LabelStore import LabelStore
from collections import OrderedDict


class RuLSIF(BaseOfflineMethod4CD):
    def offline_initialize(self):
        pass

    def offline_test(self, timeseries: TimeSeriesView) -> MethodTestResults:
        from roerich.change_point import ChangePointDetectionRuLSIF
        import roerich

        # generate time series
        X, cps_true = roerich.generate_dataset(period=200, N_tot=2000)

        # change points detection
        cpd = ChangePointDetectionRuLSIF(periods=1, window_size=100, step=5, n_runs=1)
        score, cps_pred = cpd.predict(X)
        return MethodTestResults(LabelStore([

        ]))

    @classmethod
    def _method_file_path(cls) -> str:
        return __file__


