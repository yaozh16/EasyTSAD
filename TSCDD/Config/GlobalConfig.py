from .Config import Config
from .PathConfig import PathConfig
from .ExpSchemaConfig import ExpSchemaConfig
from .EvalConfig import EvalConfig
from .DataConfig import DataConfig
from .VisConfig import VisConfig


class GlobalConfig(Config):

    def _parse(self):
        self.path_config = PathConfig(self.cfg.get("PathConfig", None))
        self.data_config = DataConfig(self.cfg.get("DataConfig", None))
        self.exp_schema_config = ExpSchemaConfig(self.cfg.get("ExpSchemaConfig", None))
        if isinstance(self.path_config.exp_schema_config_path, str):
            self.exp_schema_config.update(self.path_config.exp_schema_config_path)
        self.eval_config = EvalConfig(self.cfg.get("EvalConfig", None))
        if isinstance(self.path_config.eval_config_path, str):
            self.exp_schema_config.update(self.path_config.eval_config_path)
        self.vis_config = VisConfig(self.cfg.get("VisConfig", None))

    def items(self):
        return {
            "PathConfig": self.path_config.items(),
            "DataConfig": self.data_config.items(),
            "ExpSchemaConfig": self.exp_schema_config.items(),
            "EvalConfig": self.eval_config.items(),
            "VisConfig": self.vis_config.items(),
        }



