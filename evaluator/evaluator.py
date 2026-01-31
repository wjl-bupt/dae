# -*- encoding: utf-8 -*-
'''
@File :evaluator.py
@Created-Time :2026-01-31 11:27:59
@Author  :june
@Description   : evaluation for dae/gae
@Modified-Time : 2026-01-31 11:27:59
'''

import os
import pickle
import time
import gymnasium as gym
import numpy as np
import torch as th
from con_algo.dae.policy import CustomActorCriticPolicy as daefunc
from con_algo.vanilla_ppo.policy import VanillaPPOAC as gaefunc
from evaluator.rollout_collector import create_vector_env,  RolloutCollector


class Evaluator:
    def __init__(self, env: gym.Env, policy: th.nn.Module, eval_policy : str = "random",eval_episodes: int = 100):
        self.env = env
        self.eval_policy = eval_policy
        self.advantage_funcs = {
            "dae": daefunc,
            "gae": gaefunc,
        }
        self.eval_episodes = eval_episodes

    def load_traj(self, traj_path: str):
        if not os.path.exists(traj_path):
            self.collect_traj(traj_path, self.eval_episodes)
        

    def collect_traj(self, num_episodes: int = 100):
        collector = RolloutCollector(self.env, self.policy, self.eval_policy)
        return collector.collect_rollouts(num_episodes)
    
    def dae(self, obs, actions, rewards, dones, values, next_values, gamma=0.99, lam=0.95):
        deltas = rewards + gamma * next_values * (1 - dones) - values
        advantages = np.zeros_like(rewards)
        adv = 0.0
        for t in reversed(range(len(rewards))):
            adv = deltas[t] + gamma * lam * (1 - dones[t]) * adv
            advantages[t] = adv
        return advantages

    def compute_advantage(self, traj_data: dict, method: str = "dae"):
        if method not in self.advantage_funcs:
            raise ValueError(f"Unknown advantage function: {method}")
        return self.advantage_funcs[method](traj_data)

    def evaluate(self) -> float:
        total_rewards = []
        for episode in range(self.eval_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                obs_tensor = th.tensor(obs, dtype=th.float32).unsqueeze(0)
                with th.no_grad():
                    action, _, _, _ = self.policy.forward(obs_tensor)
                action = action.squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated
                episode_reward += reward
            total_rewards.append(episode_reward)
        avg_reward = np.mean(total_rewards)
        return avg_reward

