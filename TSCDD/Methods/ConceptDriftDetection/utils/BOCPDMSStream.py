import numpy as np
from abc import ABC, abstractmethod
from scipy import stats
from BOCPDStream import ProbabilityRecord, StudentTProb1d, GaussianProb1d, LinearProb1d


class BOCPDMSStream:
    def __init__(self, distributions: list[ProbabilityRecord], ignore_prop_lb=1e-4, h_lambda: int = 500,
                 egress_distance: int = 10):
        self._M = len(distributions)
        self._egress_distance = egress_distance
        # P(r_0=0)=1
        self._P_r_t = np.ones((self._M, 1), dtype=np.float64) / self._M
        self._dists = distributions
        self._h_lambda = h_lambda
        self._ignore_prop_lb = ignore_prop_lb

    def update(self, obs: float):

        R = len(self._P_r_t)
        if R == 0:
            self._P_r_t = np.ones(1, dtype=np.float64)
            for dist in self._dists:
                dist.update(obs)
            return
        # P(r_t=i), i=R-1,...,1,0
        # (M, R+1)
        prop_all_r = np.vstack([dist.get_probability(obs) for dist in self._dists]).reshape(self._M, -1)
        H = 1 / self._h_lambda

        # P(r_t=r_{t-1}+1), r_{t-1}=R,R-1,...,1,0, shape is (M, R+1)
        grow_prop = self._P_r_t * prop_all_r * (1 - H)

        # P(r_t=0) : shape is (M, 1)
        change_prop = np.ones((self._M, 1)) * np.sum(self._P_r_t * prop_all_r * H) / self._M
        # print(f"prop_all_r : {prop_all_r}")
        # probability of the r_t as i, i=R+1,R,R-1,...,1,0
        self._P_r_t = np.hstack([grow_prop, change_prop]) + 1e-80
        self._P_r_t /= np.sum(self._P_r_t)

        for dist in self._dists:
            dist.update(obs)

        cdf_rt = np.cumsum(np.sum(self._P_r_t, axis=0))
        remove_n: int = np.count_nonzero(cdf_rt < self._ignore_prop_lb) - self._egress_distance
        if remove_n > 0:
            for dist in self._dists:
                dist.egress_point(remove_n)
            self._P_r_t = self._P_r_t[:, remove_n:].reshape(self._M, -1)

    def get_p_m(self):
        P_m = np.sum(self._P_r_t, axis=1)
        return P_m / np.sum(P_m)

    def get_best_rt(self):
        M, L = self._P_r_t.shape
        best_index = np.argmax(self._P_r_t)
        best_m = best_index // L
        best_rt = L - 1 - (best_index % L)
        return best_m, best_rt

    def get_P_rt(self):
        return self._P_r_t


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 2000
    np.random.seed(10)
    data = np.random.random(N)
    data = np.sin(np.linspace(0, np.pi * 20, N)) + data
    data = data[:]
    rls = []
    bms = []
    ubs = []
    mps = []
    probs = np.array([]).reshape((-1, 3))
    models = [LinearProb1d(), StudentTProb1d(), GaussianProb1d()]
    stream = BOCPDMSStream(models, h_lambda=500)
    for i, d in enumerate(data):
        stream.update(d)
        RLP = np.sum(stream.get_P_rt(), axis=0)
        RLP /= np.max(RLP)
        md, rt = stream.get_best_rt()
        rls.append(rt)
        bms.append(md)
        mps.append(stream.get_p_m())
        ubs.append(stream.get_P_rt().shape[1])
        probs = np.vstack([
            probs,
            np.vstack([
                i * np.ones_like(RLP),
                np.flip(np.arange(len(RLP))),
                RLP
            ]).T
        ])
    mps = np.array(mps)
    plt.figure(figsize=(20, 20))
    plt.subplot(411)
    plt.plot(data, label="value")
    plt.legend()
    plt.subplot(412)
    plt.plot(rls, label="run length")
    plt.plot(ubs, label="upper bound")
    plt.legend()
    plt.subplot(413)
    plt.plot(mps[:, 0], label="LinearProb")
    plt.plot(mps[:, 1], label="StudentTProb")
    plt.plot(mps[:, 2], label="GaussianProb")
    plt.legend()
    plt.subplot(414)
    plt.scatter(probs[:, 0], probs[:, 1], c=1-probs[:, 2], cmap='gray')
    plt.show()
