# -*- encoding: utf-8 -*-
'''
@File :mp_run.py
@Created-Time :2026-02-01 15:17:51
@Author  :june
@Description   :Description of this file
@Modified-Time : 2026-02-01 15:17:51
'''

# eval/run.py
import torch as th
import multiprocessing as mp
from evaluator.config import EvalConfig
from evaluator.checkpoint.manager import CheckpointManager
from evaluator.rollout.collector import RolloutCollector
from evaluator.advantage.gae import GAE
from evaluator.advantage.dae import DAE
from evaluator.metrics.correlation import CorrelationMetric
from evaluator.evaluator.random_policy import RandomPolicyEvaluator
from evaluator.evaluator.correspond_policy import CorrespondPolicyEvaluator



def eval_one_ckpt(ckpt_path):
    cfg = EvalConfig()
    ckpt = th.load(ckpt_path, map_location=cfg.device)

    collector = RolloutCollector(
        cfg.env_name,
        ckpt["policy"],
        ckpt["value"],
        cfg.gamma,
        cfg.device
    )

    advs = {
        "gae": GAE(cfg.gamma, cfg.gae_lambda),
        "dae": DAE()
    }

    metric = CorrelationMetric()

    random_eval = RandomPolicyEvaluator(collector, advs, metric)
    corr_eval = CorrespondPolicyEvaluator(collector, advs, metric)

    return {
        "ckpt": str(ckpt_path),
        "random": random_eval.evaluate(),
        "correspond": corr_eval.evaluate()
    }

if __name__ == "__main__":
    manager = CheckpointManager("models/dae/")
    ckpts = manager.list_checkpoints()

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(eval_one_ckpt, ckpts)

    import json
    json.dump(results, open("adv_corr.json", "w"), indent=2)


