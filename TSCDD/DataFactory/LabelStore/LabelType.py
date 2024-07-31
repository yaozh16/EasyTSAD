from enum import Enum
from typing import Union


class LabelType(Enum):
    AnomalyPoint = "AnomalyPoint"
    ChangePoint = "ChangePoint"
    ReportPoint = "ReportPoint"
    SegmentIndex = "SegmentIndex"
    RunLength = "RunLength"
    ChangeScore = "ChangeScore"
    Default = "Default"

    @classmethod
    def get_type(cls, name: Union[str, "LabelType"]):
        if isinstance(name, str):
            try:
                return cls(name)
            except ValueError:
                return cls("Default")
        else:
            return name

    @classmethod
    def get_types(cls, names: list):
        return [cls.get_type(e) for e in names]

