import numpy as np

from ....DataFactory.TimeSeries import TimeSeriesView
from ....Methods.ConceptDriftDetection.BaseMethod4CD import BaseOnlineMethod4CD
from ... import MethodTestResults
from ....DataFactory.LabelStore.Label import ChangePointLabel, ReportPointLabel, RunLengthLabel
from ....DataFactory.LabelStore import LabelStore
from collections import OrderedDict


class OnlineExample1(BaseOnlineMethod4CD):
    """
        An example online method that detects concept-drift at each timestamp
    """

    def __init__(self, hparam: dict = None):
        super().__init__(hparams=hparam)

    def online_initialize(self):
        self.online_state.initialize()
        self.online_state.update({
            "report_points": [],
            "run_lengths": [],
        })

    def online_retrain(self, timeseries: TimeSeriesView):
        # train with timeseries...
        perform_training(timeseries)

        self.online_state.need_retrain = False

    def need_retrain(self) -> bool:
        return self.online_state.need_retrain

    def online_step(self, timeseries: TimeSeriesView):
        self.online_state.record_index(index_arr=timeseries.get_indexes())

        change_exist, run_length = detect_change(timeseries.get_values())

        if change_exist:
            self.online_state.custom_dict["report_points"].append(timeseries.end_index - 1)
            self.online_state.custom_dict["run_lengths"].append(run_length)
            self.online_state.need_retrain = True

    def online_collect_results(self) -> MethodTestResults:
        indexes = np.array(sorted(list(self.online_state.test_indexes)), dtype=np.int32)

        label_store = LabelStore()

        report_point_indexes = np.array(self.online_state.custom_dict["report_points"]) - np.min(indexes)

        label_store.set_label(ReportPointLabel.from_point_list(report_point_indexes, len(indexes), "OnlineExample(RP)"))

        run_length_values = np.arange(len(indexes))
        run_lengths = np.array(self.online_state.custom_dict["run_lengths"])
        for rpi, rl in zip(report_point_indexes, run_lengths):
            run_length_values[rpi:] -= run_length_values[rpi] - rl

        label_store.set_label(RunLengthLabel(run_length_values, "OnlineExample(RL)"))

        change_point_indexes = np.array((report_point_indexes - run_lengths) // 1, dtype=np.int32)

        label_store.set_label(ChangePointLabel.from_point_list(change_point_indexes, len(indexes), "OnlineExample(CP)"))

        self.test_results = MethodTestResults(label_store)
        return self.test_results

    @classmethod
    def _method_file_path(cls) -> str:
        return __file__


class OnlineExample2(BaseOnlineMethod4CD):
    """
        An example online method that report run lengths at each timestamp
    """

    def __init__(self, hparam: dict = None):
        super().__init__(hparams=hparam)

    def online_initialize(self):
        self.online_state.initialize()
        self.online_state.update({
            "run_lengths": OrderedDict(),
        })

    def online_retrain(self, timeseries: TimeSeriesView):
        # train with timeseries...
        perform_training(timeseries)

        self.online_state.need_retrain = False

    def need_retrain(self) -> bool:
        return self.online_state.need_retrain

    def online_step(self, timeseries: TimeSeriesView):
        self.online_state.record_index(index_arr=timeseries.get_indexes())

        run_length = detect_change(timeseries.get_values())
        self.online_state.custom_dict["run_lengths"][timeseries.end_index - 1] = run_length

    def online_collect_results(self) -> MethodTestResults:
        indexes = np.array(sorted(list(self.online_state.test_indexes)), dtype=np.int32)

        label_store = LabelStore()

        run_length_values = np.arange(len(indexes))
        run_lengths: OrderedDict = self.online_state.custom_dict["run_lengths"]
        for rpi, rl in run_lengths.items():
            run_length_values[rpi:] -= run_length_values[rpi] - rl
        label_store.set_label(RunLengthLabel(run_length_values, "OnlineExample(RL)"))

        report_point_indexes = np.where(run_length_values[1:] - run_length_values[:-1] < -1)[0] + 1
        label_store.set_label(ReportPointLabel.from_point_list(report_point_indexes, len(indexes), "OnlineExample(RP)"))

        change_point_indexes = np.array((report_point_indexes - run_length_values[report_point_indexes]) // 1,
                                        dtype=np.int32)

        label_store.set_label(ChangePointLabel.from_point_list(change_point_indexes, len(indexes), "OnlineExample(CP)"))

        self.test_results = MethodTestResults(label_store)
        return self.test_results

    @classmethod
    def _method_file_path(cls) -> str:
        return __file__
