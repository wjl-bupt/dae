# -*- encoding: utf-8 -*-
'''
@File :collector.py
@Created-Time :2026-02-01 15:14:43
@Author  :june
@Description   :Description of this file
@Modified-Time : 2026-02-01 15:14:43
'''

# eval/rollout/collector.py
import gymnasium as gym
import numpy as np
import torch

class RolloutCollector:
    def __init__(self, env_name, policy, value_fn, gamma, device):
        self.env = gym.make(env_name)
        self.policy = policy
        self.value_fn = value_fn
        self.gamma = gamma
        self.device = device

    def collect(self, n_episodes, random_policy=False):
        trajectories = []

        for _ in range(n_episodes):
            obs = self.env.reset()
            done = False

            obs_buf, act_buf, rew_buf, val_buf = [], [], [], []

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)

                if random_policy:
                    act = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        act = self.policy(obs_t.unsqueeze(0)).cpu().numpy()[0]

                val = self.value_fn(obs_t.unsqueeze(0)).item()
                next_obs, rew, done, _ = self.env.step(act)

                obs_buf.append(obs)
                act_buf.append(act)
                rew_buf.append(rew)
                val_buf.append(val)

                obs = next_obs

            returns = self._mc_return(rew_buf)
            trajectories.append({
                "obs": np.array(obs_buf),
                "acts": np.array(act_buf),
                "rews": np.array(rew_buf),
                "vals": np.array(val_buf),
                "returns": returns
            })

        return trajectories

    def _mc_return(self, rewards):
        G, out = 0.0, []
        for r in reversed(rewards):
            G = r + self.gamma * G
            out.insert(0, G)
        return np.array(out)


