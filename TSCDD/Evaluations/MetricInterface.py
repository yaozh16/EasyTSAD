from abc import ABCMeta, abstractmethod


class MetricInterface(metaclass=ABCMeta):
    '''
    The MetricInterface class is an abstract base class that defines the interface for metrics. It serves as a blueprint for creating subclasses that represent specific metrics.

    The class includes three abstract methods: add(), avg(), and to_dict(). Subclasses inheriting from this class must implement these methods according to their specific metric calculations and requirements.

    You should implement the following methods:

        - add(self, other_metric): This abstract method represents the operation of combining two metrics. It takes another metric object (other_metric) as a parameter and is responsible for adding its values to the current metric.

        - avg(self): This abstract method calculates the average value of the metric. It should be implemented by subclasses to compute the average based on the accumulated values.

        - to_dict(self): This abstract method converts the metric object into a dictionary representation. It should return a dictionary containing the metric's values and any additional information needed for representation or storage.

    '''

    def __init__(self, name: str, aggregated: bool = False):
        self.name = name
        self.aggregated = aggregated

    @abstractmethod
    def to_dict(self):
        """
        This abstract method converts the metric object into a dictionary representation. It should return a dictionary containing the metric's values and any additional information needed for representation or storage.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def aggregate(cls, metrics: list["MetricInterface"], strict_name_check: bool = False) -> "MetricInterface":
        raise NotImplementedError()

    @staticmethod
    def list_to_dict(metrics: list["MetricInterface"]) -> dict:
        result_dict = {}
        for metric in metrics:
            result_dict.update(metric.to_dict())
        return result_dict

