from enum import Enum


class EvalLoadOption(Enum):
    Last = "Last"
    All = "All"

    @classmethod
    def get(cls, name: str):
        try:
            return cls(name)
        except ValueError:
            return EvalLoadOption.Last


