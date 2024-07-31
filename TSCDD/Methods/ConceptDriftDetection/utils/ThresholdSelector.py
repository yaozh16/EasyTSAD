import numpy as np
from scipy.stats import genextreme


class ThresholdSelector:
    @classmethod
    def extreme_theory_value(cls, arr: np.ndarray, pct: float = 0.95):
        params = genextreme.fit(arr)
        gev_dist = genextreme(*params)
        threshold = gev_dist.ppf(pct)
        return threshold

    @classmethod
    def max_value(cls, arr: np.ndarray, times: float = 1):
        mean_value = np.mean(arr)
        return np.max(arr - mean_value) * times + mean_value

    @classmethod
    def k_sigma_value(cls, arr: np.ndarray, k: float = 1):
        return np.mean(arr) + np.std(arr) * k
