from .Config import Config
from .Options import ExpSchemaOptions
from .TestSchemaConfig import TestSchemaConfig
from .MethodHyperParamConfig import MethodHyperParamConfig


class ExpSchemaConfig(Config):

    _exp_schema_options = {member.name: member for member in ExpSchemaOptions}

    def _parse(self):
        self.test_schema_config = TestSchemaConfig(self.cfg.get("TestSchemaConfig", {}))
        self.option: ExpSchemaOptions = self._exp_schema_options.get(self.cfg.get("exp_schema_option", "Naive"))
        self.train_ratio = self.cfg.get("train_ratio", 0.5)
        self.test_ratio = self.cfg.get("test_ratio", 1.0)
        self.save_test_results: bool = self.cfg.get("save_test_results", True)
        self.method_hparams: MethodHyperParamConfig = MethodHyperParamConfig(self.cfg.get("Method Hparams", {}))

    def items(self) -> dict:
        return {
            "TestSchemaConfig": self.test_schema_config.items(),
            "exp_schema_option": self.option.name,
            "train_ratio": self.train_ratio,
            "test_ratio": self.test_ratio,
            "save_test_results": self.save_test_results,
            "Method Hparams": self.method_hparams.items(),
        }

