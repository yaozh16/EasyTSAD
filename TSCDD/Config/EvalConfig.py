from .Config import Config
from .Options import EvalLoadOption


class EvalConfig(Config):

    def items(self) -> dict:
        return {
            "margins": self.margins,
            "eval_load_option": self.eval_load_option.name
        }

    def _parse(self):
        self.margins = self.cfg.get("margins", [0, 0])
        self.eval_load_option = EvalLoadOption.get(self.cfg.get("eval_load_option", "Last"))
