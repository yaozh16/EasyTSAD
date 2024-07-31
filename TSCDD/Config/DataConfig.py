from .Config import Config
from typing import Union
from ..DataFactory.LabelStore.LabelType import LabelType
from .Options import PreprocessOptions


class DataConfig(Config):

    def _parse(self):
        self.preprocess: list = [PreprocessOptions.get(e) for e in self.cfg.get("preprocess", [])]
        self.dataset_dir: str = self.cfg.get("dataset_dir", "../datasets")
        self.dataset_type: str = self.cfg.get("dataset_type", "TS4CD")
        self.datasets: Union[None, str, list[Union[str, list[str, list[str]]]]] = self.cfg.get("datasets", None)
        self.label_types: list[LabelType] = LabelType.get_types(self.cfg.get("label_types", ["Default"]))

    def items(self) -> dict:
        return {
            "preprocess": [e.name for e in self.preprocess],
            "dataset_dir": self.dataset_dir,
            "dataset_type": self.dataset_type,
            "datasets": self.datasets,
            "label_types": [e.name for e in self.label_types],
        }


