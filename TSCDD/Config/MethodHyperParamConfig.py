from .Config import Config
from ..utils import update_nested_dict
from typing import Tuple
"""
{
    "ExampleMethod":{
        "Default":{...},
        (DatasetType1,):{...},
        (DatasetType1, Dataset1):{...},
        (DatasetType1, Dataset1, Curve1):{...},
    }
}
"""


class MethodHyperParamConfig(Config):

    def _parse(self):
        pass

    def items(self) -> dict:
        return self.cfg

    def get_hparams(self, method_name: str, custom_config: dict, curve_key: Tuple = (), cascading: bool = True):
        assert len(curve_key) <= 3, "Curve key should be no more than 3"
        if method_name not in self.cfg:
            self.cfg[method_name] = {"Default": {}}
            return {}
        method_configs = self.cfg.get(method_name, {"Default": {}})
        if not cascading:
            curve_key = "Default" if len(curve_key) < 1 else curve_key
            config_dict = method_configs.get(curve_key, {})
        else:
            config_dict = method_configs.get("Default", {})
            for i in range(1, len(curve_key)):
                config_dict = update_nested_dict(config_dict, method_configs.get(curve_key[:i], {}))
        if isinstance(custom_config, dict):
            config_dict.update(custom_config)
        return config_dict

