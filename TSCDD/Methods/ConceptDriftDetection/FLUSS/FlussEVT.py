import numpy as np
from ....DataFactory.LabelStore import ChangePointLabel, LabelStore, ReportPointLabel
from ....DataFactory.TimeSeries import TimeSeriesView
from ....Methods import MethodTestResults

from ..BaseMethod4CD import BaseOfflineMethod4CD
from ..utils.MatrixProfile import AC, IAC, MatrixProfile
from ..utils.ThresholdSelector import ThresholdSelector


def fluss_get_cac(timeseries: np.ndarray, window_size, margin_ignore=3):
    assert timeseries.ndim == 1
    mp, mpi, mp_left, mpi_left, mp_right, mpi_right = MatrixProfile.matrix_profile_1d(timeseries, window=window_size)
    ac = AC.calc_ac(mpi, margin_ignore=margin_ignore)
    iac = IAC.get_mpi_iac(len(mpi))
    cac = ac / iac
    cac[cac > 1] = 1.
    cac[-margin_ignore:] = 1.
    cac[:margin_ignore] = 1.
    return cac


def get_extreme_values(data: np.ndarray, pct=0.95):
    data = 1 - data

    threshold = ThresholdSelector.extreme_theory_value(data, pct)

    outliers, = np.where(data > threshold)
    return outliers


class FlussEVT(BaseOfflineMethod4CD):
    def offline_initialize(self):
        pass

    def offline_test(self, timeseries: TimeSeriesView) -> MethodTestResults:
        window_size = self.hparams.get("window_size", 5)
        margin_ignore = self.hparams.get("margin_ignore", 5)
        ndim = timeseries.get_dim()
        timeseries_values: np.ndarray = timeseries.get_values().reshape((-1, ndim))
        labels = []
        L = timeseries.size()
        has_changed = False
        for dim in range(ndim):
            cac = fluss_get_cac(timeseries_values[:, dim], window_size, margin_ignore)
            change_points = get_extreme_values(cac)
            change_point_label = ChangePointLabel.from_point_list(change_points.tolist(), L, f"FLUSS(dim-{dim})")
            has_changed |= len(change_points) > 0
            labels.append(change_point_label)
        if has_changed:
            labels.append(ReportPointLabel.from_point_list([L-1], L, "FLUSS(RP)"))
        else:
            labels.append(ReportPointLabel(np.zeros(L), "FLUSS(RP)"))
        self.test_results = MethodTestResults(LabelStore(labels))
        return self.test_results

    @classmethod
    def _method_file_path(cls) -> str:
        return __file__
