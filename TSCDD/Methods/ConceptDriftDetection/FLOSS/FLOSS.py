from abc import abstractmethod
from collections import OrderedDict

from ..utils.MatrixProfile import *
from ..utils.ThresholdSelector import ThresholdSelector
from ... import MethodTestResults
from ....DataFactory.LabelStore import LabelStore
from ....DataFactory.LabelStore.Label import ChangePointLabel, ReportPointLabel, RunLengthLabel
from ....DataFactory.TimeSeries import TimeSeriesView
from ....Methods.ConceptDriftDetection.BaseMethod4CD import BaseOnlineMethod4CD

class FlossBase(BaseOnlineMethod4CD):

    def __init__(self, hparam: dict = None):
        super().__init__(hparams=hparam)
        self._margin_ignore = self.hparams.get("margin_ignore", 3)
        self._window = self.hparams.get("window_size", 5)

    def online_initialize(self):
        self.online_state.initialize()
        self.online_state.update({
            "run_lengths": OrderedDict(),
            "floss_streams": [],
            "thresholds": [],
            "last_index": None,
        })

    def online_retrain(self, timeseries: TimeSeriesView):
        # train with timeseries...
        D = timeseries.get_dim()
        values = timeseries.get_values()
        floss_streams = []
        thresholds = []
        for d in range(D):
            stream = MatrixProfileRightStream(values[:, d], window=self._window, ignore_distance=1,
                                              egress_oldest=True, index_rolling=True)
            floss_streams.append(stream)
            cac = self._get_cac(stream)
            thresh = self._get_thresh(cac)
            thresholds.append(thresh)
        self.online_state.need_retrain = False
        self.online_state.update({
            "last_index": timeseries.end_index,
            "report_points": [],
            "run_lengths": [],
            "floss_streams": floss_streams,
            "thresholds": thresholds,
        })

    def _get_cac(self, stream: MatrixProfileRightStream) -> np.ndarray:
        mpi_right = stream.get_mpi_right()
        ac = AC.calc_ac(mpi_right, margin_ignore=self._margin_ignore)
        iac = IAC.get_mpi_right_iac(len(ac), count_head=True)
        cac = ac / iac
        cac[cac > 2.] = 2.
        cac[:self._margin_ignore] = 2.
        cac[-self._margin_ignore:] = 2.
        return cac

    @abstractmethod
    def _get_thresh(self, cac_values: np.ndarray) -> float:
        raise NotImplementedError("FlossBase get_thresh not implemented")

    def need_retrain(self) -> bool:
        return self.online_state.need_retrain

    def online_step(self, timeseries: TimeSeriesView):
        self.online_state.record_index(index_arr=timeseries.get_indexes())
        last_index = self.online_state.custom_dict["last_index"]
        if timeseries.end_index <= last_index:
            return
        floss_streams: list[MatrixProfileRightStream] = self.online_state.custom_dict["floss_streams"]
        thresholds: list[float] = self.online_state.custom_dict["thresholds"]
        D = timeseries.get_dim()
        assert len(floss_streams) == D

        update_length = timeseries.end_index - last_index
        update_values = timeseries.get_values()[-update_length:]

        rl = []
        for dim, stream in enumerate(floss_streams):
            for v in update_values[:, dim]:
                stream.update(v)
            cac = self._get_cac(stream)
            cps = np.where(cac < thresholds[dim])[0]
            if len(cps) > 0:
                rl.append(stream.get_L() - cps[-1])
        if len(rl) > D / 2:
            run_length = np.mean(rl)
            self.online_state.custom_dict["report_points"].append(timeseries.end_index - 1)
            self.online_state.custom_dict["run_lengths"].append(run_length)

    def online_collect_results(self) -> MethodTestResults:
        indexes = np.array(sorted(list(self.online_state.test_indexes)), dtype=np.int32)

        label_store = LabelStore()

        report_point_indexes = np.array(self.online_state.custom_dict["report_points"]) - np.min(indexes)

        label_store.set_label(
            ReportPointLabel.from_point_list(report_point_indexes, len(indexes), "OnlineExample(RP)"))

        run_length_values = np.arange(len(indexes), dtype=np.float64)
        run_lengths = np.array(self.online_state.custom_dict["run_lengths"], dtype=np.float64)
        for rpi, rl in zip(report_point_indexes, run_lengths):
            run_length_values[rpi:] -= run_length_values[rpi] - rl

        label_store.set_label(RunLengthLabel(run_length_values, "OnlineExample(RL)"))

        change_point_indexes = np.array((report_point_indexes - run_lengths) // 1, dtype=np.int32)

        label_store.set_label(
            ChangePointLabel.from_point_list(change_point_indexes, len(indexes), "OnlineExample(CP)"))

        self.test_results = MethodTestResults(label_store)
        return self.test_results

    @classmethod
    def _method_file_path(cls) -> str:
        return __file__


class FlossEVT(FlossBase):
    def _get_thresh(self, cac_values: np.ndarray) -> float:
        return 1 - ThresholdSelector.extreme_theory_value(1. - cac_values, 0.95)


class FlossMax(FlossBase):
    def _get_thresh(self, cac_values: np.ndarray) -> float:
        return 1 - ThresholdSelector.max_value(1. - cac_values)


class Floss3Sigma(FlossBase):
    def _get_thresh(self, cac_values: np.ndarray) -> float:
        return 1 - ThresholdSelector.k_sigma_value(1. - cac_values, 3.0)


class FlossFixThresh(FlossBase):
    def _get_thresh(self, cac_values: np.ndarray) -> float:
        return self.hparams.get("thresh_fix", 0.3)
