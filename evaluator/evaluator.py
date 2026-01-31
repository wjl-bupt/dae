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
        save_path = f"./trajectories/{self.eval_policy}_episodes{num_episodes}_traj.pkl"

        buffers = []  # 存最终完整轨迹
        num_collected = 0

        obs, _ = self.env.reset()
        num_envs = obs.shape[0]

        # 每个并行环境一个 trajectory buffer
        traj_buffers = [SingleTrajctoryBuffer() for _ in range(num_envs)]

        while num_collected < num_episodes:
            obs_tensor = th.tensor(obs, dtype=th.float32)

            with th.no_grad():
                actions, _, _, _ = self.policy.forward(obs_tensor)

            actions = actions.cpu().numpy()

            next_obs, rewards, terminated, truncated, _ = self.env.step(actions)
            dones = terminated | truncated

            for env_id in range(num_envs):
                traj_buffers[env_id].add(
                    obs[env_id],
                    actions[env_id],
                    rewards[env_id],
                    dones[env_id],
                )

                if dones[env_id]:
                    # 一条完整轨迹结束
                    single_traj = traj_buffers[env_id].get_trajectory()
                    single_traj["returns"] = traj_buffers[env_id].compute_returns(
                        gamma=0.99
                    )

                    buffers.append(single_traj)
                    num_collected += 1

                    # 重置该环境对应的 buffer
                    traj_buffers[env_id] = SingleTrajctoryBuffer()

                    if num_collected >= num_episodes:
                        break

            obs = next_obs

        with open(save_path, "wb") as f:
            pickle.dump(buffers, f)

        logger.success(f"Collected {num_collected} trajectories")
        logger.success(f"Trajectories saved to {save_path}")


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
                done = terminated or truncated
                episode_reward += reward
            total_rewards.append(episode_reward)
        avg_reward = np.mean(total_rewards)
        return avg_reward

