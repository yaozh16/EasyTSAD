from logging import getLogger
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..Config.Options.VisualizationOption import VisOption
from ..Config.PathManager import PathManager
from ..Config.VisConfig import VisConfig
from ..DataFactory.DataStore import DataStore
from ..DataFactory.LabelStore import LabelView, Label, LabelTools
from ..Schema.TestResultLoader import TestResultLoader
from ..DataFactory.LabelStore import LabelType
from ..Config.Options import EvalLoadOption

class RawValuePlotter:
    @classmethod
    def name(cls):
        return "Raw"

    @classmethod
    def plot(cls, label_view: LabelView, label_msg:str):
        plt.plot(label_view.get(), label=label_msg)


class RunLengthPlotter(RawValuePlotter):

    @classmethod
    def name(cls):
        return "RunLength"

    @classmethod
    def plot(cls, label_view: LabelView, label_msg: str):
        if label_view.label_type() is LabelType.RunLength:
            run_length = label_view.get()
        else:
            run_length, _ = LabelTools.convert_binary_to_run_length(label_view.get())
        plt.plot(run_length, label=label_msg)


class Visualizer:
    def __init__(self, data_store: DataStore, vis_config: VisConfig):
        self._ds: DataStore = data_store
        self._logger = getLogger("logger")
        self._cfg = vis_config

    def _plot_per_curve(self, method_names: list[str], schema_names: list[str], method_label_types: list[LabelType],
                        plot: type[RawValuePlotter]):
        pm = PathManager.get_instance()
        for method_name, schema_name in zip(method_names, schema_names):
            iter_msg = f"Visualization Method[{method_name}] Schema[{schema_name}] Plot[{plot.name()}]"
            label_loader = TestResultLoader(method_name, schema_name, self._ds, EvalLoadOption.Last)
            desc = tqdm(label_loader)
            desc.set_description(iter_msg)
            for timestamp, curve_key, ground_truth_labels, method_output_labels in desc:
                if method_label_types is not None:
                    method_output_labels = [e for e in method_output_labels
                                            if e.label_type() in method_label_types]
                dataset_type, dataset_name, curve_name = curve_key

                subplot_count = len(method_output_labels) + len(ground_truth_labels)
                if self._cfg.plot_timeseries:
                    timeseries = self._ds.get_ts_data(curve_key).get_view().get_timeseries()
                    subplot_count += timeseries.get_dim()
                # configure grid setting
                cols = self._cfg.plot_columns
                rows = int(np.ceil(subplot_count / cols))
                if self._cfg.total_size:
                    plt.figure(figsize=self._cfg.fig_size, dpi=self._cfg.dpi)
                else:
                    spw, sph = self._cfg.fig_size
                    plt.figure(figsize=(spw * cols, sph * rows), dpi=self._cfg.dpi)

                subplot_index = 0
                if self._cfg.plot_timeseries:
                    timeseries = self._ds.get_ts_data(curve_key).get_view().get_timeseries()
                    timeseries_values = timeseries.get_values()
                    timeseries_columns = timeseries.get_columns()
                    for dim, column in enumerate(timeseries_columns):
                        subplot_index += 1
                        ax = plt.subplot(rows, cols, subplot_index)
                        plt.plot(timeseries_values[:, dim], label=f"raw_ts[{column}]")
                        plt.legend()

                for gti, gt in enumerate(ground_truth_labels):
                    gt: LabelView = gt
                    subplot_index += 1
                    ax = plt.subplot(rows, cols, subplot_index)
                    plot.plot(gt, label_msg=f"GT[{gt.annotator()}]")
                    plt.legend()

                for moi, mo in enumerate(method_output_labels):
                    subplot_index += 1
                    ax = plt.subplot(rows, cols, subplot_index)
                    plot.plot(mo, label_msg=f"MO[{mo.annotator()}]")
                    plt.legend()

                vis_file_path = pm.get_vis_by_curve_path(method_name, schema_name, dataset_type, dataset_name,
                                                         curve_name, f"{plot.name()}.PerCurve.png",
                                                         safe_dir=True)
                plt.savefig(vis_file_path)
                plt.close()

    def plot_all(self, options: set[VisOption], method_names: list[str], schema_names: list[str],
                 method_label_types: list[LabelType] = None):
        for option in options:
            if option is VisOption.RawPlotPerCurve:
                self._plot_per_curve(method_names, schema_names, method_label_types, RawValuePlotter)
            elif option is VisOption.RunLengthPlotPerCurve:
                self._plot_per_curve(method_names, schema_names, method_label_types, RunLengthPlotter)
            else:
                self._logger.warning(f"Visualizer meets unknown option {option}")

