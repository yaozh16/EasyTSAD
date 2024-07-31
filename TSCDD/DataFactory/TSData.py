import pandas as pd
import json
from .LabelStore import LabelStore, LabelStoreView
from .TimeSeries import TimeSeries, TimeSeriesView
from .LabelStore.LabelType import LabelType
from .ObjectView import ObjectView

from TSCDD.Config.PathManager import PathManager
from typing import Dict


class TSData:
    '''
    TSData contains all information used for training, validation and test, including the dataset values and dataset information. Some typical preprocessing method are provided in class methods.

    Attributes:
        timeseries (TimeSeries):
            The observation information in DataFrame format. Each column represents a dimension.
        labels (Labels4CD)
        info (dict):
            Some information about the dataset, which might be useful.
    '''

    def __init__(self, timeseries: TimeSeries, labels: LabelStore, info: Dict) -> None:
        self.timeseries = timeseries
        self.labels = labels
        self.info = info

    @classmethod
    def buildfrom(cls, dataset_type, dataset, curve_name, label_types: list[LabelType], preprocess: list):
        '''
        Build customized TSDataSet instance from numpy file.
`
        Args:
            dataset_type (str):
                The dataset type, e.g. "TS4CD".
            dataset (str):
                The dataset name where the curve comes from, e.g. "AlanTuringTCPD";
            curve_name (str):
                The curve's name;
            label_types (list[LabelType]):
                The types of the label.
            preprocess (list):
                preprocess options.
        Returns:
         A TSData4CD instance.
        '''
        pm = PathManager.get_instance()

        timestamp_path = pm.get_dataset_timestamps(dataset_type, dataset, curve_name)
        timestamps: pd.DataFrame = pd.read_pickle(timestamp_path)
        value_path = pm.get_dataset_values(dataset_type, dataset, curve_name)
        values: pd.DataFrame = pd.read_pickle(value_path)
        assert isinstance(timestamps, pd.DataFrame), f"Loaded timestamp file {timestamp_path} is not a DataFrame"
        assert isinstance(values, pd.DataFrame), f"Loaded observation file {value_path} is not a DataFrame"

        assert len(timestamps.index) == len(values.index), f"The size ({len(timestamps.index)}) of timestamps in " \
                                                           f"{timestamp_path} is not the same as the size " \
                                                           f"({len(values.index)}) of observations in {value_path}"
        timeseries = TimeSeries(timestamps, values, preprocess=preprocess)
        labels = LabelStore()
        for label_type in label_types:
            label_path = pm.get_dataset_labels(dataset_type, dataset, curve_name, label_type)
            labels.load(label_path, label_type)

        info_path = pm.get_dataset_info(dataset_type, dataset, curve_name)
        with open(info_path, 'r') as f:
            info = json.load(f)

        return cls(timeseries, labels, info)

    def get_dim(self):
        return self.timeseries.get_dim()

    def __len__(self):
        return len(self.timeseries)

    def size(self):
        return len(self.timeseries)

    def get_view(self, start_index: int = None, end_index: int = None, dims: list[int] = None):
        return TSDataView(self, start_index, end_index, dims)


class TSDataView(ObjectView):

    def _inherit(self, parent: "TSDataView"):
        self._dims = parent._dims

    def __init__(self, tsdata: TSData, start_index: int, end_index: int, dims: list[int] = None):
        super().__init__(tsdata, start_index, end_index)
        self._tsdata = tsdata
        if isinstance(dims, list):
            dims = [int(e) for e in dims if 0 <= int(e) < tsdata.get_dim()]
        else:
            dims = list(range(tsdata.get_dim()))
        self._dims = dims
        self._timeseries_view = TimeSeriesView(tsdata.timeseries, self.start_index, self.end_index, dims=dims)
        self._labels_view = LabelStoreView(tsdata.labels, self.start_index, self.end_index)

    def get_timeseries(self) -> TimeSeriesView:
        return self._timeseries_view

    def get_labels(self) -> LabelStoreView:
        return self._labels_view
