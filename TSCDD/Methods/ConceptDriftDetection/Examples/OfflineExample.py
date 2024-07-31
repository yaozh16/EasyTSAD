import numpy as np

from ....DataFactory.TimeSeries import TimeSeriesView
from ....Methods.ConceptDriftDetection.BaseMethod4CD import BaseOfflineMethod4CD
from ... import MethodTestResults
from ....DataFactory.LabelStore.Label import ChangePointLabel, ReportPointLabel, RunLengthLabel
from ....DataFactory.LabelStore import LabelStore
from collections import OrderedDict


class OfflineExample(BaseOfflineMethod4CD):
    def offline_initialize(self):
        pass

    def offline_test(self, timeseries: TimeSeriesView) -> MethodTestResults:
        pass

    @classmethod
    def _method_file_path(cls) -> str:
        return __file__


