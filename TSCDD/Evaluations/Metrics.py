from dataclasses import dataclass
from . import MetricInterface


class F1class(MetricInterface):
    '''
    The F1class is a concrete implementation of the MetricInterface abstract class. It represents an F1 metric and
    provides methods for adding metric values, calculating averages, and converting the metric into a dictionary.

    Attributes:
    - name: str: The name of the F1 metric.
    - p: float = 0: The precision value of the metric (default 0).
    - r: float = 0: The recall value of the metric (default 0).
    - f1: float = 0: The F1 score value of the metric (default 0).
    - thres: float = -1: The threshold value of the metric (default -1).
    - num: int = 1: The number of metric instances (default 1).
    
    Methods:
    - add(self, other): Adds the values of another F1class instance to the current metric by summing their respective
        attributes.
    - avg(self): Calculates the average values of the precision, recall, and F1 score by dividing them by the num
        attribute.
    - to_dict(self): Converts the metric object into a dictionary representation. If num is 1, it includes the
        threshold value in the dictionary, otherwise, it excludes it.'''

    def __init__(self, name: str, p: float = 0, r: float = 0, f1: float = 0, thres: float = -1,
                 aggregated: bool = False):
        super(F1class, self).__init__(name, aggregated)
        self.p = p
        self.r = r
        self.f1 = f1
        self.thres = thres

    def to_dict(self):
        if self.aggregated:
            return {
                self.name: {
                    'f1': self.f1,
                    'precision': self.p,
                    'recall': self.r,
                }
            }
        else:
            return {
                self.name: {
                    'f1': self.f1,
                    'precision': self.p,
                    'recall': self.r,
                    'threshold': self.thres
                }
            }

    @classmethod
    def aggregate(cls, metrics: list["F1class"], strict_name_check: bool = False) -> "F1class":
        L = len(metrics)
        if L == 0:
            raise ValueError("Aggregation error: empty list.")
        name = metrics[0].name
        p = 0
        r = 0
        f1 = 0
        for metric in metrics:
            if strict_name_check:
                if name != metric.name:
                    raise TypeError(f"Aggregation error: name mismatch between {name} and {metric.name}")
            if not isinstance(metric, cls):
                raise TypeError(f"Aggregation error: class mismatch between {metric.__class__} and {cls}")
            p += metric.p
            r += metric.r
            f1 += metric.f1
        p /= L
        r /= L
        f1 /= L
        return cls(name, p, r, f1, aggregated=True)


class MultipleFloatValueMetric(MetricInterface):
    def to_dict(self):
        return {self.name: self.attr_dict}

    @classmethod
    def aggregate(cls, metrics: list["MultipleFloatValueMetric"], strict_name_check: bool = False) \
            -> "MultipleFloatValueMetric":
        L = len(metrics)
        if L == 0:
            raise ValueError("Aggregation error: empty list.")
        if not isinstance(metrics[0], MultipleFloatValueMetric):
            raise TypeError("Aggregation error: non-MultipleFloatValueMetric founded")
        name = metrics[0].name
        attr_dict = {k: 0 for k in metrics[0].attr_dict}
        for metric in metrics:
            if strict_name_check:
                if name != metric.name:
                    raise TypeError(f"Aggregation error: name mismatch between {name} and {metric.name}")
            if not isinstance(metric, cls):
                raise TypeError(f"Aggregation error: class mismatch between {metric.__class__} and {cls}")
            if not isinstance(metric, MultipleFloatValueMetric):
                raise TypeError("Aggregation error: non MultipleFloatValueMetric founded")
            for k, v in metric.attr_dict.items():
                if k not in attr_dict:
                    raise ValueError(f"Aggregation error: key mismatch while aggregation")
                attr_dict[k] += v

        attr_dict = {k: v / L for k, v in attr_dict.items()}
        return cls(name, **attr_dict, aggregated=True)

    def __init__(self, name: str, aggregated: bool = False, **kwargs):
        super(MultipleFloatValueMetric, self).__init__(name, aggregated)
        self.attr_dict = {k: float(v) for k, v in kwargs.items()}


class SingleValueMetric(MetricInterface):

    def __init__(self, name: str, value: float, aggregated: bool = False):
        value = float(value)
        super(SingleValueMetric, self).__init__(name, aggregated)
        self.value = value

    def to_dict(self):
        return {
            self.name: self.value
        }

    @classmethod
    def aggregate(cls, metrics: list["SingleValueMetric"], strict_name_check: bool = False) -> "SingleValueMetric":
        L = len(metrics)
        if L == 0:
            raise ValueError("Aggregation error: empty list.")
        name = metrics[0].name
        value = 0
        for metric in metrics:
            if strict_name_check:
                if name != metric.name:
                    raise TypeError(f"Aggregation error: name mismatch between {name} and {metric.name}")
            if not isinstance(metric, cls):
                raise TypeError(f"Aggregation error: class mismatch between {metric.__class__} and {cls}")
            value += metric.value
        value /= L
        return cls(name, value, aggregated=True)








