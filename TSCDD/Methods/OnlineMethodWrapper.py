from . import MethodTestResults
from .BaseMethods import BaseOnlineMethod, BaseOfflineMethod
from ..DataFactory.DataStore import DataStoreViewIndex
from ..DataFactory.LabelStore import LabelStore
from ..DataFactory.LabelStore.Label import Label, ReportPointLabel
from ..DataFactory.TimeSeries import TimeSeriesView


class OnlineMethodWrapper(BaseOnlineMethod):

    _offline_method: BaseOfflineMethod = None

    def __init__(self, offline_method: BaseOfflineMethod, test_period: int = 30, cache_size: int = 300):
        self.__class__._offline_method = offline_method
        super().__init__({})
        self._test_period = test_period
        self._retest_size = cache_size
        self._test_overlap = 1

    def online_initialize(self):
        self.online_state.initialize()
        self.online_state.update({
            "accumulated": None,
            "accumulated_index": None,
            "accumulated_results": [],
        })
        self.online_state.need_retrain = False

    def online_retrain(self, timeseries: TimeSeriesView):
        pass

    def need_retrain(self) -> bool:
        return False

    def online_step(self, timeseries: TimeSeriesView):
        self.online_state.record_index(timeseries.get_indexes())
        accumulated: TimeSeriesView = self.online_state.custom_dict.get("accumulated", None)
        accumulated_results: list = self.online_state.custom_dict.get("accumulated_results", [])
        if accumulated is None:
            accumulated = timeseries.safe_slice(0, 0)

        last_test_index = accumulated.end_index
        while accumulated.end_index + self._test_period <= timeseries.end_index:
            accumulated.expand_end(self._test_period, size_upperbound=self._retest_size, inplace=True)
            test_results = self._test(accumulated)
            accumulated_results.append([
                lbl.last(accumulated.end_index - last_test_index + self._test_overlap)
                for lbl in test_results.label_stores.labels
            ])
            last_test_index = accumulated.end_index
        self.online_state.custom_dict["accumulated_index"] = timeseries.end_index
        self.online_state.custom_dict["accumulated"] = accumulated
        self.online_state.custom_dict["accumulated_results"] = accumulated_results

    def online_collect_results(self) -> MethodTestResults:
        accumulated: TimeSeriesView = self.online_state.custom_dict.get("accumulated", None)
        accumulated_index: int = self.online_state.custom_dict.get("accumulated_index", None)
        ac_rs: list[list[Label]] = self.online_state.custom_dict.get("accumulated_results", [])
        last_test_index = accumulated.end_index
        if accumulated_index > last_test_index:
            accumulated.expand_end(accumulated_index - last_test_index, size_upperbound=self._retest_size, inplace=True)
            test_results = self._test(accumulated)
            ac_rs.append([
                lbl.last(accumulated.end_index - last_test_index) for lbl in test_results.label_stores.labels
            ])

        label_count = 0 if len(ac_rs) < 1 else len(ac_rs[0])
        connected_labels: list[list[Label]] = [[] for i in range(label_count)]
        for i, seg_labels in enumerate(ac_rs):
            for d, lb in enumerate(seg_labels):
                connected_labels[d].append(lb)
        connected_labels: list[Label] = [e[0].connect_from(e, overlap=self._test_overlap) for e in connected_labels]
        return MethodTestResults(LabelStore(connected_labels))

    def _test(self, accumulated: TimeSeriesView) -> MethodTestResults:
        self._offline_method.offline_initialize()
        return self._offline_method.offline_test(accumulated)

    @classmethod
    def _method_file_path(cls) -> str:
        return cls._offline_method._method_file_path()

    def train_valid(self, data_store_view_index: DataStoreViewIndex):
        pass
