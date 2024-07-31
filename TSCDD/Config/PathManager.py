import logging
import os

from . import GlobalConfig
from ..DataFactory.LabelStore.LabelType import LabelType

logger = logging.getLogger("logger")


def build_dir(path1, path2):
    path = os.path.join(path1, path2)
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def check_and_build(path):
    if not os.path.exists(path):
        os.makedirs(path)


class PathManager:
    '''
    PathManager manages the paths related to this project. It will automatically build directory for newly introduced methods. Also, you can easily get access to the name of any file you want using the given methods.
    
    NOTE: This class obeys Singleton Pattern. Only one instance exists for global access.
    '''
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PathManager, cls).__new__(cls)
        else:
            logger.error("Multiple PathManager instances. Violate Singleton Pattern.")
        return cls._instance

    @staticmethod
    def get_instance() -> "PathManager":
        return PathManager._instance

    @staticmethod
    def del_instance():
        PathManager._instance = None

    def __init__(self, glo_cfg: GlobalConfig) -> None:
        self.glo_cfg: GlobalConfig = glo_cfg

    def load_dataset_path(self, data_dir):
        self.glo_cfg.data_config.dataset_dir = data_dir
        self.glo_cfg.sync_cfg()

    def get_dataset_path(self, types, dataset):
        return os.path.join(self.glo_cfg.data_config.dataset_dir, types, dataset)

    def get_dataset_curves(self, types, dataset):
        return os.listdir(os.path.join(self.glo_cfg.data_config.dataset_dir, types, dataset))

    def get_dataset_one_curve(self, types, dataset, curve):
        return os.path.join(self.glo_cfg.data_config.dataset_dir, types, dataset, curve)

    def get_dataset_timestamps(self, types, dataset, curve):
        return os.path.join(self.glo_cfg.data_config.dataset_dir, types, dataset, curve, "timestamps.pkl")

    def get_dataset_values(self, types, dataset, curve):
        return os.path.join(self.glo_cfg.data_config.dataset_dir, types, dataset, curve, "values.pkl")

    def get_dataset_labels(self, types, dataset, curve, label_type: LabelType):
        if label_type is LabelType.ChangePoint:
            return os.path.join(self.glo_cfg.data_config.dataset_dir, types, dataset, curve, "change_point_labels.pkl")
        else:
            raise NotImplementedError(f"Path format of label_type {label_type} is not specified yet.")

    def get_dataset_info(self, types, dataset, curve):
        return os.path.join(self.glo_cfg.data_config.dataset_dir, types, dataset, curve, "info.json")

    def get_test_output_dir(self, method_name: str, schema_name: str, dataset_type: str, dataset_name: str,
                            curve_name: str, timestamp: str, safe_dir=True):
        test_output_dir = os.path.join(self.glo_cfg.path_config.output_path, "Results", "Test", schema_name,
                                       method_name, dataset_type, dataset_name, curve_name, timestamp)
        if safe_dir:
            if not os.path.exists(test_output_dir):
                os.makedirs(test_output_dir)
            elif not os.path.isdir(test_output_dir):
                raise OSError(f"Creating directory failed: {test_output_dir}")
        return test_output_dir

    def get_test_output_result_path(self, method_name: str, schema_name: str, dataset_type: str, dataset_name: str,
                                    curve_name: str, timestamp: str, safe_dir=True):
        return os.path.join(self.get_test_output_dir(method_name, schema_name, dataset_type, dataset_name, curve_name,
                                                     timestamp, safe_dir),
                            "result.json")

    def get_test_output_reference_view_path(self, method_name: str, schema_name: str, dataset_type: str,
                                            dataset_name: str, curve_name: str, timestamp: str, safe_dir=True):
        return os.path.join(self.get_test_output_dir(method_name, schema_name, dataset_type, dataset_name, curve_name,
                                                     timestamp, safe_dir),
                            "ground_truth_view.json")

    @classmethod
    def split_test_output_result_path(cls, path):
        path, _ = os.path.split(path)
        path, timestamp = os.path.split(path)
        path, curve_name = os.path.split(path)
        path, dataset_name = os.path.split(path)
        path, dataset_type = os.path.split(path)
        path, method_name = os.path.split(path)
        path, schema_name = os.path.split(path)
        return schema_name, method_name, dataset_type, dataset_name, curve_name, timestamp

    def get_eval_curve_dir(self, method_name: str, schema_name: str, dataset_type: str, dataset_name: str,
                           curve_name: str, cur_timestamp: str, safe_dir=True):
        eval_path = os.path.join(self.glo_cfg.path_config.output_path, "Results", "Eval", schema_name, method_name,
                                 dataset_type, dataset_name, curve_name, str(cur_timestamp))
        if safe_dir:
            if not os.path.exists(eval_path):
                os.makedirs(eval_path)
            elif not os.path.isdir(eval_path):
                raise OSError(f"Creating directory failed: {eval_path}")
        return eval_path

    def get_eval_detail_path(self, method_name: str, schema_name: str, dataset_type: str, dataset_name: str,
                             curve_name: str, label_name: str, cur_timestamp: str, safe_dir=True):
        eval_path = self.get_eval_curve_dir(method_name, schema_name, dataset_type, dataset_name, curve_name,
                                            cur_timestamp, safe_dir)
        return os.path.join(eval_path, f"{label_name}.detail.json")

    def get_eval_agg_by_curve_path(self, method_name: str, schema_name: str, dataset_type: str, dataset_name: str,
                                   curve_name: str, label_name: str, cur_timestamp: str, safe_dir=True):
        eval_path = self.get_eval_curve_dir(method_name, schema_name, dataset_type, dataset_name, curve_name,
                                            cur_timestamp, safe_dir)
        return os.path.join(eval_path, f"{label_name}.aggregated.json")

    @classmethod
    def split_eval_agg_by_curve_path(cls, path):
        path, agg_json = os.path.split(path)
        label_name = agg_json.replace(".aggregated.json", "")
        path, timestamp = os.path.split(path)
        path, curve_name = os.path.split(path)
        path, dataset_name = os.path.split(path)
        path, dataset_type = os.path.split(path)
        path, method_name = os.path.split(path)
        path, schema_name = os.path.split(path)
        return schema_name, method_name, dataset_type, dataset_name, curve_name, timestamp, label_name

    def get_eval_by_dataset_path(self, method_name: str, schema_name: str, dataset_type: str, dataset_name: str,
                                 safe_dir=True):
        eval_path = os.path.join(self.glo_cfg.path_config.output_path, "Results", "Eval", schema_name, method_name,
                                 dataset_type, dataset_name)
        if safe_dir:
            if not os.path.exists(eval_path):
                os.makedirs(eval_path)
            elif not os.path.isdir(eval_path):
                raise OSError(f"Creating directory failed: {eval_path}")
        return os.path.join(eval_path, "aggregated.json")

    def get_vis_by_curve_path(self, method_name: str, schema_name: str, dataset_type: str, dataset_name: str,
                              curve_name: str, file_name: str, safe_dir=True):
        vis_path = os.path.join(self.glo_cfg.path_config.output_path, "Results", "Vis", schema_name, method_name,
                                dataset_type, dataset_name, curve_name)
        if safe_dir:
            if not os.path.exists(vis_path):
                os.makedirs(vis_path)
            elif not os.path.isdir(vis_path):
                raise OSError(f"Creating directory failed: {vis_path}")
        return os.path.join(vis_path, file_name)

    def check_valid(self, path, msg):
        if not os.path.exists(path):
            raise FileNotFoundError(msg)
