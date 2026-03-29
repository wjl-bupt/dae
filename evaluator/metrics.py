# -*- encoding: utf-8 -*-
'''
@File :metrics.py
@Created-Time :2026-02-01 15:16:50
@Author  :june
@Description   : metric for evaluator
@Modified-Time : 2026-02-01 15:16:50
'''

# eval/metrics/correlation.py
import numpy as np

class CorrelationMetric:
    @staticmethod
    def compute(adv, returns, vals):
        target = returns - vals
        return float(np.corrcoef(adv, target)[0, 1])


