import numpy as np
from ...EvalPreprocess import EvalPreprocessOption
from ...EvalInterface import EvalInterface
from ...Metrics import MultipleFloatValueMetric


class DetectionDelay(EvalInterface):
    """
    Using DetectionDelay to evaluate the models.
    """

    def __init__(self) -> None:
        super().__init__("Detection Delay", EvalPreprocessOption.Default)

    def calc(self, scores: np.ndarray, labels: np.ndarray, margins) -> MultipleFloatValueMetric:
        assert scores.ndim == 1, f"Score dimension should be 1 (current score dimension: {scores.ndim}."
        assert labels.ndim == 1, f"Label dimension should be 1 (current label dimension: {labels.ndim}."

        worst = len(scores)
        distance = [worst] * worst

        for i in range(len(scores) - 1, -1, -1):
            if scores[i]:
                worst = i
            distance[i] = worst - i
        distance = np.array(distance)
        real_change_points_indexes, = np.where(labels)
        detection_delays = distance[real_change_points_indexes]
        if len(detection_delays) == 0:
            detection_delays = np.array([0])
        attr_dict = {
            "delay_mean": np.mean(detection_delays),
            "delay_std": np.std(detection_delays),
            "delay_pct(90)": np.percentile(detection_delays, 90),
            "delay_pct(70)": np.percentile(detection_delays, 70),
            "delay_pct(50)": np.percentile(detection_delays, 50),
            "delay_pct(30)": np.percentile(detection_delays, 30),
            "delay_pct(10)": np.percentile(detection_delays, 10),
        }

        return MultipleFloatValueMetric(
            name=self.name,
            ** attr_dict
        )
