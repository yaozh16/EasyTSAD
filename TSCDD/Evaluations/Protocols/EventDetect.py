from ..Metrics import SingleValueMetric
from typing import Type

import numpy as np
from .. import MetricInterface, EvalInterface


        
class EventDetect(EvalInterface):
    """
    Using the UCR detection protocol to evaluate the models. As there is only one anomaly segment in one time series, if and only if the highest score is in the anomaly segment, this time series is considered to be detected.
    """
    def __init__(self) -> None:
        super().__init__("Event Detected")
        
    def calc(self, scores, labels, margins) -> MetricInterface:
        '''
        Returns:
            MetricInterface: An instance of Precision representing if the anomaly is detected.
        '''

        idx = np.argmax(scores)
        detected = 0
        if labels[idx] == 1:
            detected = 1
        return SingleValueMetric(
            self.name,
            value=float(detected)
        )
    