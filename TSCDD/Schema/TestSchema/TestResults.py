from typing import Tuple, Union
from TSCDD.Methods import BaseMethod, MethodTestResults


class TestResult:
    def __init__(self, method_output: MethodTestResults = None, other_result: dict = None):
        self._result = method_output
        self.other_result = other_result if isinstance(other_result, dict) else {}

    def set_result(self, result: MethodTestResults):
        self._result = result

    def get_result(self):
        return self._result

    def to_json(self):
        json_obj = {"method_output": self._result.to_json()}
        if len(self.other_result) > 0:
            json_obj["other"] = self.other_result
        return json_obj

    @classmethod
    def from_json(cls, json_obj):
        test_result = cls(method_output=MethodTestResults.from_json(json_obj["method_output"]),
                          other_result=json_obj.get("method_output", None))
        return test_result


class OnlineTestResult(TestResult):
    def __init__(self, method_output: MethodTestResults = None):
        super().__init__(method_output)
        self.other_result = {"retrain_count": 0}

    def record_retrain(self):
        self.other_result["retrain_count"] += 1

    def get_retrain_count(self):
        return self.other_result["retrain_count"]


class TestResults:
    def __init__(self, result_dict: dict[Tuple[str, str, str], TestResult] = None):
        if isinstance(result_dict, dict):
            self._result_dict: dict[Tuple[str, str, str], TestResult] = result_dict
        else:
            self._result_dict: dict[Tuple[str, str, str], TestResult] = {}

    def update(self, results: Union[dict, "TestResults"] = None):
        if isinstance(results, dict):
            self._result_dict.update(results)
        elif isinstance(results, TestResults):
            self._result_dict.update(results._result_dict)
        else:
            raise ValueError(f"TestResults update type error. Found {type(results)}")

    def get(self, keys, default=None):
        return self._result_dict.get(keys, default)

    def keys(self):
        return self._result_dict.keys()

    def items(self):
        return self._result_dict.items()

    def to_json(self):
        return {k: v.to_json() for k, v in self.items()}

    @staticmethod
    def from_json(json_obj: dict) -> "TestResults":
        return TestResults({k: TestResult.from_json(v) for k, v in json_obj.items()})

