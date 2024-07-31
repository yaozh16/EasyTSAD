from .Config import Config
from typing import Dict


class PathConfig(Config):
    def _parse(self):
        self.method_config_paths: Dict[str, str] = self.cfg.get("Method", {})
        self.exp_schema_config_path: str = self.cfg.get("ExpSchema", None)
        self.test_schema_config_path: str = self.cfg.get("TestSchema", None)
        self.eval_config_path: str = self.cfg.get("EvalConfig", None)
        self.output_path: str = self.cfg.get("Output", ".")

    def items(self) -> dict:
        return {
            "Method": self.method_config_paths,
            "ExpSchema": self.exp_schema_config_path,
            "TestSchema": self.test_schema_config_path,
            "EvalConfig": self.eval_config_path,
            "Output": self.output_path,
        }


