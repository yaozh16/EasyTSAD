from .Config import Config
from .Options import TestSchemaOptions


class TestSchemaConfig(Config):

    _test_schema_options = {member.name: member for member in TestSchemaOptions}

    def _parse(self):
        self.option: TestSchemaOptions = TestSchemaOptions.get(self.cfg.get("test_schema_mode", None))
        self.online_step_size: int = self.cfg.get("online_step_size", 1)
        self.online_retrain_window_size: int = self.cfg.get("online_retrain_window_size", 10)
        self.online_sliding_window_size: int = self.cfg.get("online_sliding_window_size", 1)
        self.quiet: bool = self.cfg.get("quiet", False)
        self.offline_test_period: int = self.cfg.get("offline_test_period", 30)
        self.offline_test_size: int = self.cfg.get("offline_test_size", 300)

    def items(self) -> dict:
        return {
            "test_schema_mode": self.option.name,
            "quiet": self.quiet,
            "online_step_size": self.online_step_size,
            "online_retrain_window_size": self.online_retrain_window_size,
            "online_sliding_window_size": self.online_sliding_window_size,
            "offline_test_size": self.offline_test_size,
            "offline_test_period": self.offline_test_period,
        }


