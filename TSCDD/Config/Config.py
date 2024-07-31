import copy
import json
from abc import ABC, abstractmethod
from typing import Union, Dict
from TSCDD.utils import update_nested_dict
import toml


class Config(ABC):
    def __init__(self, cfg: Union[dict, str, "Config"]):
        if isinstance(cfg, dict):
            self.cfg = cfg
        elif isinstance(cfg, str):
            if cfg.endswith(".toml"):
                self.cfg = toml.load(cfg)
            elif cfg.endswith(".json"):
                with open(cfg, "r") as f:
                    self.cfg = json.load(f)
            else:
                raise NotImplementedError(f"Unknown configuration file type :{cfg}")
        elif isinstance(cfg, self.__class__):
            self.cfg = copy.deepcopy(cfg.cfg)
        else:
            self.cfg = {}
        self._parse()
        self.sync_cfg()

    @abstractmethod
    def _parse(self):
        raise NotImplementedError(f"{self.__class__} _parse is not implemented.")

    def update(self, cfg: Union[dict, str, "Config"]):
        if isinstance(cfg, dict):
            update_config: dict = cfg
        elif isinstance(cfg, str):
            if cfg.endswith(".toml"):
                update_config: dict = toml.load(cfg)
            elif cfg.endswith(".json"):
                with open(cfg, "r") as f:
                    update_config: dict = json.load(f)
            else:
                raise NotImplementedError(f"Unknown configuration file type :{cfg}")
        elif isinstance(cfg, Config):
            update_config: dict = copy.deepcopy(cfg.cfg)
        else:
            update_config: dict = {}

        self.cfg = update_nested_dict(self.cfg, update_config)
        self._parse()
        self.sync_cfg()
        return self

    def sync_cfg(self):
        self.cfg = self.items()

    def copy(self, update_dict: dict = None):
        copy_cfg = copy.deepcopy(self.cfg)
        if isinstance(update_dict, dict):
            copy_cfg.update(update_dict)
        return self.__class__(copy_cfg)

    @abstractmethod
    def items(self) -> dict:
        raise NotImplementedError(f"{self.__class__} items is not implemented.")


