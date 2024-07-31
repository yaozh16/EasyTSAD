from enum import Enum


class TestSchemaOptions(Enum):
    Online = "Online"
    Offline = "Offline"
    Default = "Default"

    @classmethod
    def get(cls, name: str):
        try:
            return cls(name)
        except ValueError:
            return TestSchemaOptions.Default


