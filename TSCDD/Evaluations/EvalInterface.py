from abc import ABCMeta, abstractmethod
from typing import Dict, Type
from .MetricInterface import MetricInterface
from .EvalPreprocess import EvalPreprocessOption
from ..DataFactory.LabelStore import LabelType


class EvalInterface(object):
    '''
    The EvalInterface is an abstract base class that defines the interface for evaluation metrics in a generic evaluation system. It serves as a blueprint for concrete evaluation metric classes that implement specific evaluation logic.

    Methods:
        - calc(self, scores, labels, margins) -> Type[MetricInterface]: Abstract method that calculates the evaluation metric based on the provided scores, labels, and margins parameters. It returns an instance of a class that inherits from MetricInterface.

        - get_name(self): Abstract method that returns the name of the evaluation metric. Concrete classes implementing this interface should provide their own implementation of this method.
    '''
    __metaclass__ = ABCMeta

    def __init__(self, name, preprocess_option: EvalPreprocessOption = EvalPreprocessOption.Default):
        self.name = name
        self.preprocess_option = preprocess_option

    @abstractmethod
    def calc(self, scores, labels, margins) -> MetricInterface:
        """
        Calculates the evaluation metric based on the provided scores, labels, and margins.

        Args:
            scores (numpy.ndarray): An array of anomaly scores.
            labels (numpy.ndarray): An array of ground truth labels.
            margins ([int(margin-before-anomaly), int(margin-after-anomaly)]): You can use margin to
            prune your evaluations if needed.

        Returns:
            MetricInterface: An instance of the event detection metric.

        """
        raise NotImplementedError()




