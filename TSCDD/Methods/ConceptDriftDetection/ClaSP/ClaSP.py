import numpy as np

from ....DataFactory.TimeSeries import TimeSeriesView
from ....Methods.ConceptDriftDetection.BaseMethod4CD import BaseOfflineMethod4CD
from ... import MethodTestResults
from ....DataFactory.LabelStore import LabelStore
from ....DataFactory.LabelStore.Label import ChangePointLabel
from .clasp_src.segmentation import segmentation


class ClaSP(BaseOfflineMethod4CD):
    """
    code source: https://sites.google.com/view/ts-parameter-free-clasp/
    """

    def offline_initialize(self):
        pass

    def offline_test(self, timeseries: TimeSeriesView) -> MethodTestResults:
        window_size = self.hparams.get("window_size", 5)
        ndim = timeseries.get_dim()
        timeseries_values: np.ndarray = timeseries.get_values().reshape((-1, ndim))
        labels = []
        L = timeseries.size()
        for dim in range(ndim):
            profile, window_size, found_cps, found_scores = segmentation(timeseries_values[:, dim],
                                                                         window_size=window_size)
            change_point_label = ChangePointLabel.from_point_list(found_cps.tolist(), L, f"ClaSP(dim-{dim})")
            labels.append(change_point_label)

        return MethodTestResults(LabelStore(labels))

    @classmethod
    def _method_file_path(cls) -> str:
        return __file__

