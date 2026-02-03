# eval/config.py
from dataclasses import dataclass

@dataclass
class EvalConfig:
    env_name: str = "HalfCheetah-v4"
    gamma: float = 0.99

    n_eval_episodes: int = 20
    num_envs: int = 8

    device: str = "cpu"

    adv_types = ("gae", "dae")
    eval_modes = ("random", "correspond")

    gae_lambda: float = 0.95
