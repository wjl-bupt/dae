# -*- encoding: utf-8 -*-
'''
@File :evaluator.py
@Created-Time :2026-02-01 15:19:21
@Author  :june
@Description   :Description of this file
@Modified-Time : 2026-02-01 15:19:21
'''

# eval/evaluator/base.py
from abc import ABC, abstractmethod
from evaluator.collector import RolloutCollector


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, ckpt):
        pass

class Evaluator(Evaluator):
    def __init__(self, collector, adv_estimators, metric, n_episodes, policy):
        self.collector = collector
        self.adv_estimators = adv_estimators
        self.metric = metric
        self.n_episodes = n_episodes
        self.policy = policy
    
    
    def evaluate(self):
        trajs = self.collector.collect(
            n_episodes=self.n_episodes,
            random_policy=True
        )

        results = {}
        for name, adv_est in self.adv_estimators.items():
            corrs = []
            for traj in trajs:
                adv = adv_est.compute(traj)
                corrs.append(
                    self.metric.compute(adv, traj["returns"], traj["vals"])
                )
            results[name] = sum(corrs) / len(corrs)
        return results