import pandas as pd
import numpy as np
from itertools import accumulate
from typing import Union
from ..ObjectView import ObjectView
from .Preprocess import Preprocessor


class TimeSeries:
    """
    `
    Attributes:
        timestamps_df (pd.DataFrame):
            The time information in DataFrame format.
        index (np.ndarray):
            The index of timestamps
        timestamps (np.ndarray):
            The time information in np.ndarray format.
        values (np.ndarray):
            The observation information in DataFrame format. Each column represents a dimension.
    """

    def __init__(self, timestamps_df: pd.DataFrame, values_df: pd.DataFrame, preprocess: list):
        """
        Build TimeSeries
`
        Args:
            timestamps_df (pd.DataFrame):
                The time information in DataFrame format.
            values_df (DataFrame):
                The observation information in DataFrame format. Each column represents a dimension.
            preprocess (list):
                Preprocesses
        """
        self.timestamps_df = timestamps_df
        self.values_df = values_df
        self._len = len(timestamps_df)
        if len(timestamps_df) != len(values_df):
            raise ValueError("The timeseries lengths is not aligned")

        self.index = timestamps_df.index.values
        if "timestamps" in timestamps_df:
            self.timestamps = timestamps_df["timestamps"].values
        else:
            self.timestamps = self.index
        self.values = Preprocessor.process(values_df.values, options=preprocess)
        assert not np.any(np.isnan(self.values)), "NA value found in timeseries, please specify a preprocessing option"
        self.columns = np.array(values_df.columns)
        self._n_dim = len(values_df.columns)

    def get_dim(self):
        return self._n_dim

    def __len__(self):
        return self._len

    def size(self):
        return self._len

    def get_view(self, start_index: int, end_index: int):
        return TimeSeriesView(self, start_index, end_index)

    def split_views(self, fractions: list[float]) -> "list[TimeSeriesView]":
        return self.get_view(0, self.size()).split_by_fractions(fractions)

    def get_indexes(self, start_index: int, end_index: int) -> np.ndarray:
        return self.index[start_index: end_index]

    def get_timestamps(self, start_index: int, end_index: int) -> np.ndarray:
        return self.timestamps[start_index: end_index]

    def get_values(self, start_index: int, end_index: int, dims: list[int]) -> np.ndarray:
        if self.values.ndim > 0:
            return self.values[start_index: end_index, dims]
        else:
            return self.values[start_index: end_index]

    def get_columns(self, dims: list[int]):
        return self.columns[dims]


class TimeSeriesView(ObjectView):
    def _inherit(self, parent: "TimeSeriesView"):
        self._dims: list[int] = [e for e in parent._dims]

    def __init__(self, timeseries: TimeSeries, start_index: int, end_index: int, dims: list[int] = None):

        super().__init__(timeseries, start_index, end_index, safety_check=True)
        self._timeseries = timeseries
        if isinstance(dims, list):
            self._dims: list[int] = [int(e) for e in dims if 0 <= int(e) < timeseries.get_dim()]
        else:
            self._dims: list[int] = list(range(timeseries.get_dim()))

    def get_indexes(self) -> np.ndarray:
        return self._timeseries.get_indexes(self.start_index, self.end_index)

    def get_timestamps(self) -> np.ndarray:
        return self._timeseries.get_timestamps(self.start_index, self.end_index)

    def get_values(self) -> np.ndarray:
        return self._timeseries.get_values(self.start_index, self.end_index, self._dims)

    def get_columns(self) -> list:
        return self._timeseries.get_columns(self._dims)

    def step(self, step_num: int = 1, inplace: bool = True, upperbound: int = None) -> Union[None, "TimeSeriesView"]:
        next_start = self.start_index + step_num
        next_end = self.end_index + step_num
        if not isinstance(upperbound, int):
            upperbound = self._timeseries.size()
        next_end = int(min(next_end, upperbound))
        next_start = int(min(next_start, next_end))
        if inplace:
            self.start_index = next_start
            self.end_index = next_end
        else:
            return TimeSeriesView(self._timeseries, next_start, next_end)

    def get_dim(self):
        return len(self._dims)

    def split_by_dimensions(self) -> list["TimeSeriesView"]:
        views = []
        for dim_index in self._dims:
            views.append(TimeSeriesView(self._timeseries, self.start_index, self.end_index, [dim_index]))
        return views

    def __add__(self, other: "TimeSeriesView") -> "TimeSeriesView":
        assert isinstance(other, TimeSeriesView), "Add operation is only allowed between TimeSeriesViews"
        assert self._timeseries == other._timeseries, "Add operation is only allowed between TimeSeriesViews " \
                                                      "managing the same Timeseries."
        if set(self._dims) == set(other._dims):
            assert self.start_index <= other.end_index and other.start_index <= self.end_index, \
                "Add operation is only allowed between connected or intersected TimeSeriesViews"
            new_start_index = int(min(self.start_index, other.start_index))
            new_end_index = int(max(self.end_index, other.end_index))
            return TimeSeriesView(self._timeseries, new_start_index, new_end_index, self._dims)
        elif self.start_index == other.start_index and self.end_index == other.end_index:
            new_dims = sorted(list(set(self._dims + other._dims)))
            return TimeSeriesView(self._timeseries, self.start_index, self.end_index, new_dims)

    def align_end_with(self, other: "TimeSeriesView"):
        assert isinstance(other, TimeSeriesView), "Align operation is only allowed between TimeSeriesViews " \
                                                  "managing the same Timeseries."
        move_steps = other.end_index - self.end_index
        self.end_index += move_steps
        self.start_index += move_steps
        if self.start_index < 0:
            self.start_index = 0
        if self.end_index > self._timeseries.size():
            self.end_index = self._timeseries.size()

    def expand_end_with(self, other: "TimeSeriesView", size_upperbound: int, inplace: bool = True) -> "TimeSeriesView":
        assert isinstance(size_upperbound, int) and size_upperbound >= 0, "Size upperbound in expand operation must " \
                                                                          "be non-negative"
        assert isinstance(other, TimeSeriesView), "Expand operation is only allowed between TimeSeriesViews " \
                                                  "managing the same Timeseries."

        return self.expand_end(expand_num=other.end_index - self.end_index, size_upperbound=size_upperbound,
                               inplace=inplace)

    def expand_end(self, expand_num: int, size_upperbound: int, inplace: bool = True) -> "TimeSeriesView":

        assert isinstance(size_upperbound, int) and size_upperbound >= 0, "Size upperbound in expand operation must " \
                                                                          "be non-negative"
        if inplace:
            view = self
        else:
            view = TimeSeriesView(self._timeseries, self.start_index, self.end_index, self._dims)
        if expand_num > 0:
            view.end_index = int(expand_num + self.end_index)
        if view.size() > size_upperbound:
            view.start_index += view.size() - size_upperbound
        if view.start_index < 0:
            view.start_index = 0
        if view.start_index >= view.end_index:
            view.start_index = view.end_index
        return view
