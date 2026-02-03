# eval/advantage/base.py
import numpy as np
from abc import ABC, abstractmethod

class AdvantageEstimator(ABC):
    @abstractmethod
    def compute(self, trajectory):
        pass

class GAE(AdvantageEstimator):
    def __init__(self, gamma, lam):
        self.gamma = gamma
        self.lam = lam

    def compute(self, traj):
        rews, vals = traj["rews"], traj["vals"]
        T = len(rews)
        adv = np.zeros(T)
        last = 0.0

        for t in reversed(range(T)):
            next_v = vals[t+1] if t+1 < T else 0.0
            delta = rews[t] + self.gamma * next_v - vals[t]
            last = delta + self.gamma * self.lam * last
            adv[t] = last
        return adv

class DAE(AdvantageEstimator):
    def __init__(self, **kwargs):
        self.params = kwargs

    def compute(self, traj):
        # 你的 DAE 逻辑
        raise NotImplementedError