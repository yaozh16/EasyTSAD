import numpy as np
from abc import ABC, abstractmethod
from scipy import stats


class ProbabilityRecord(ABC):
    """
    记录所有从last n的distribution的结果, n=N,N-1,....,0
    """

    @abstractmethod
    def update(self, obs: float | np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def get_probability(self, obs: float | np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def egress_point(self, num: int):
        raise NotImplementedError()


class BOCPDStream:
    def __init__(self, distribution: ProbabilityRecord, ignore_prop_lb=1e-4, h_lambda: int = 500,
                 egress_distance: int = 10, record_probs: bool = False, enable_crop_acceleration: bool = True):
        # P(r_0=0)=1
        self._P_r_t = np.ones(0, dtype=np.float64)
        self._dist = distribution
        self._h_lambda = h_lambda
        self._ignore_prop_lb = ignore_prop_lb
        self._egress_distance = egress_distance
        self._probs = np.ones(0, dtype=np.float64)
        self._record_probs = record_probs
        self._enable_crop_acceleration = enable_crop_acceleration

    def update(self, obs: float | np.ndarray):
        R = len(self._P_r_t)
        if R == 0:
            self._P_r_t = np.ones(1, dtype=np.float64)
            if self._record_probs:
                self._probs = np.ones((1, 1), dtype=np.float64)
            self._dist.update(obs)
            return
        # P(r_t=i), i=R-1,R-2,...,0
        prob_all_r = self._dist.get_probability(obs)
        assert len(prob_all_r) == R
        if self._record_probs:
            OBS_N, MAX_R = self._probs.shape
            if R > MAX_R:
                self._probs = np.hstack([self._probs, np.zeros((OBS_N, R - MAX_R))])
                MAX_R = R
            self._probs = np.vstack([self._probs, np.zeros((1, MAX_R))])
            self._probs[-1, :R] = np.flip(prob_all_r)

        H = 1 / self._h_lambda

        # P(r_t=r_{t-1}+1), r_t=R,R-1,...,1
        grow_prop = self._P_r_t * prob_all_r * (1 - H)

        # P(r_t=0)
        change_prop = np.sum(self._P_r_t * prob_all_r * H)

        # probability of the r_t as i, i=R,R-1,...,1,0
        self._P_r_t = np.append(grow_prop, change_prop)
        self._P_r_t /= np.sum(self._P_r_t)

        self._dist.update(obs)
        if self._enable_crop_acceleration:
            cdf_rt = np.cumsum(self._P_r_t)
            remove_n: int = np.count_nonzero(cdf_rt < self._ignore_prop_lb) - self._egress_distance
            if remove_n > 0:
                self._dist.egress_point(remove_n)
                self._P_r_t = self._P_r_t[remove_n:]

    def get_rt(self):
        return len(self._P_r_t) - 1 - np.argmax(self._P_r_t)

    def get_P_rt(self):
        return self._P_r_t

    def get_probs(self):
        return self._probs


class Probability1d(ProbabilityRecord):
    """
    记录所有从last n的distribution的结果, n=N,N-1,....,0
    """

    @abstractmethod
    def update(self, obs: float):
        raise NotImplementedError()

    @abstractmethod
    def get_probability(self, obs: float) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def egress_point(self, num: int):
        raise NotImplementedError()


class GaussianProb1d(Probability1d):

    def __init__(self, init_sigma2=1.0):
        self._init_sigma2 = init_sigma2
        self._sumX = np.array([])
        self._sumX_sqr = np.array([])
        self._N = np.array([])

    def update(self, obs: float):
        """
        mu = sum(X)/len(X) = E(X)
        sigma^2 = sum((X-mu)**2) = E(X^2) - (E(X))^2 = sum(X^2)/N - (sumX / N)^2
        """
        self._sumX = np.append(self._sumX + obs, obs)
        self._sumX_sqr = np.append(self._sumX_sqr + obs * obs, obs * obs)
        self._N = np.append(self._N + 1, 1)

    def get_probability(self, obs: float) -> np.ndarray:
        mu = self._sumX / self._N
        sigma = np.sqrt((self._sumX_sqr / self._N) - np.square(self._sumX / self._N) + self._init_sigma2)
        prob = stats.norm.pdf(obs, mu, sigma)
        return prob

    def egress_point(self, num: int):
        self._sumX = self._sumX[num:]
        self._sumX_sqr = self._sumX_sqr[num:]
        self._N = self._N[num:]


class StudentTProb1d(Probability1d):
    def __init__(self):
        self._alphaAll = np.array([])
        self._betaAll = np.array([])
        self._countAll = np.array([])
        self._muAll = np.array([])

    def update(self, obs: float):
        self._betaAll = np.append(
            self._countAll + self._countAll / (self._countAll + 1) * np.square(obs - self._muAll) / 2, 1)
        self._muAll = np.append((self._muAll * self._countAll + obs) / (self._countAll + 1), obs)
        self._countAll = np.append(self._countAll + 1, 1)
        self._alphaAll = np.append(self._alphaAll + 0.5, 0.5)

    def get_probability(self, obs: float) -> np.ndarray:
        mean = self._muAll
        freedom = 2 * self._alphaAll
        precision = self._alphaAll * self._countAll / (self._countAll + 1) / self._betaAll
        prob = stats.t.pdf(obs, loc=mean, df=freedom, scale=precision)
        return prob

    def egress_point(self, num: int):
        self._alphaAll = self._alphaAll[num:]
        self._countAll = self._countAll[num:]
        self._muAll = self._muAll[num:]
        self._betaAll = self._betaAll[num:]


class LinearProb1d(Probability1d):
    """
    X - ai - b ~ N(0, sigma)

    斜率 a = (n * Σ(I * X) - ΣI * ΣX) / (n * Σ(I^2) - (ΣI)^2)
    截距 b = (sum X - a * sumI) / n
    sigma^2 = sum((X-a*i-b)^2) = sum(X^2) -2b sumX + b^2 * n +  sum(i^2 * a^2) + 2ab sumI - 2a sumIX

    """

    def __init__(self, init_sigma2=1.0):
        self._init_sigma2 = init_sigma2
        self._sumIX = np.array([])
        self._sumI = np.array([])
        self._sumI_sqr = np.array([])
        self._sumX = np.array([])
        self._sumX_sqr = np.array([])
        self._N = np.array([])

    def get_a(self):
        return (self._N * self._sumIX - self._sumI * self._sumX) / \
            (self._N * self._sumI_sqr - np.square(self._sumI) + 1e-30)

    def get_b(self, a: np.ndarray = None):
        if a is None:
            a = self.get_a()
        return (self._sumX - a * self._sumI) / self._N

    def get_sigma(self, a: np.ndarray = None, b: np.ndarray = None):
        if a is None:
            a = self.get_a()
        if b is None:
            b = self.get_b(a)
        sigma_sqr = self._sumX_sqr + self._N * np.square(b) + np.square(a) * self._sumI_sqr + 2 * a * b * self._sumI - \
                    2 * a * self._sumIX - 2 * b * self._sumX + self._init_sigma2
        return np.sqrt(sigma_sqr)

    def update(self, obs: float):
        # I start from 0
        self._sumIX = np.append(self._sumIX + self._N * obs, 0)
        self._sumI = np.append(self._sumI + self._N, 0)
        self._sumI_sqr = np.append(self._sumI_sqr + np.square(self._N), 0)
        self._sumX = np.append(self._sumX + obs, obs)
        obs_sqr = obs * obs
        self._sumX_sqr = np.append(self._sumX_sqr + obs_sqr, obs_sqr)
        self._N = np.append(self._N + 1, 1)

    def get_probability(self, obs: float) -> np.ndarray:
        a = self.get_a()
        b = self.get_b(a)
        sigma = self.get_sigma(a, b)
        mu = obs - a * (self._N - 1) - b
        probs = stats.norm.pdf(0, mu, sigma) + 1e-40
        """print(f"get_prob: {obs}")
        print(f"a: {a}")
        print(f"b: {b}")
        print(f"sigma: {sigma}")
        print(f"mu: {mu}")
        print(f"probs: {probs}")
        print(f"")"""
        return probs

    def egress_point(self, num: int):
        self._sumIX = self._sumIX[num:]
        self._sumI = self._sumI[num:]
        self._sumI_sqr = self._sumI_sqr[num:]
        self._sumX = self._sumX[num:]
        self._sumX_sqr = self._sumX_sqr[num:]
        self._N = self._N[num:]


class BVARProb1d(Probability1d):
    """
    X_t = \sum_{i=1}^{L} A_i X_{t-i} + B
    """

    def __init__(self, max_lag: int = 3):
        self._max_lag = max_lag

    def update(self, obs: float):
        pass

    def get_probability(self, obs: float) -> np.ndarray:
        pass

    def egress_point(self, num: int):
        pass


class MTSProbIID(ProbabilityRecord):
    def __init__(self, distributions: list[Probability1d]):
        self._dists: list[Probability1d] = distributions

    @classmethod
    def define_iid_mts_distribution(cls, distribution_1d_type: type[Probability1d], n_dims: int):
        return cls([distribution_1d_type() for i in range(n_dims)])

    def update(self, obs: np.ndarray):
        for _x, _d in zip(obs, self._dists):
            _d.update(_x)

    def get_probability(self, obs: np.ndarray) -> np.ndarray:
        prob = None
        for _x, _d in zip(obs, self._dists):
            if prob is None:
                prob = _d.get_probability(_x)
            else:
                prob *= _d.get_probability(_x)
        return prob

    def egress_point(self, num: int):
        for _d in self._dists:
            _d.egress_point(num)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(10)
    data = np.random.random((200, 1))
    # data[:70, 0] += 5
    # data[:30, 1] += np.arange(30) * 2
    # data[:, 1] += np.arange(100) * 2
    data[80:81, 0] += 5
    data = data[:]
    probability_model = MTSProbIID([
        StudentTProb1d()
    ])
    rls = []
    ubs = []
    rl_probs = np.zeros((len(data), len(data)))
    stream = BOCPDStream(probability_model, h_lambda=500, enable_crop_acceleration=False)
    for idx, d in enumerate(data):
        stream.update(d)
        rt = stream.get_rt()
        rls.append(rt)
        rl_probs[idx, :len(stream.get_P_rt())] = np.flip(stream.get_P_rt() / np.max(stream.get_P_rt()))
        ubs.append(len(stream.get_P_rt()))
    plt.subplot(311)
    plt.plot(data[:, 0], label="timeseries")
    plt.legend()
    plt.subplot(312)
    plt.plot(rls, label="run length")
    plt.legend()
    plt.subplot(313)
    # plt.imshow(np.rot90(255-rl_probs)[50:], cmap="grey")
    for i in range(len(rl_probs)):
        rl_probs[i] = np.roll(rl_probs[i], -i)
    cp_prob = np.rot90(255-255*rl_probs)

    plt.imshow(cp_prob, cmap="grey")

    #plt.plot(ubs)
    plt.show()
