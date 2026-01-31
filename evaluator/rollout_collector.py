# -*- encoding: utf-8 -*-
'''
@File :rollout_collector.py
@Created-Time :2026-01-31 12:00:53
@Author  :june
@Description   :Description of this file
@Modified-Time : 2026-01-31 12:00:53
'''

import pickle
import torch as th
from loguru import logger
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from loguru import logger

def make_ant_env(seed: int | None = None):
    def _init():
        env = gym.make("Ant-v5")
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init
def create_vector_env(num_envs: int, seed: int = 0):
    env_fns = [
        make_ant_env(seed + i) for i in range(num_envs)
    ]
    return SyncVectorEnv(env_fns)

class SingleTrajctoryBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def add(self, obs, action, reward, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def get_trajectory(self):
        return {
            "observations": np.array(self.observations),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "dones": np.array(self.dones),
        }
    
    def compute_returns(self, gamma=0.99):
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        return returns

    def save_pkl(self, filepath, gamma=0.99):
        traj = self.get_trajectory()
        traj["returns"] = self.compute_returns(gamma=gamma)
        with open(filepath, "wb") as f:
            pickle.dump(traj, f)
        logger.success(f"Trajectory saved to {filepath}")


class RolloutCollector:
    def __init__(
        self,
        env,
        policy,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.device = device

    def collect(self, num_episodes: int, save_path: str):
        logger.info(f"Start collecting {num_episodes} trajectories")

        obs, _ = self.env.reset()
        num_envs = obs.shape[0]

        # 每个 env 一条“正在进行”的轨迹
        active_trajs = [
            SingleTrajctoryBuffer() for _ in range(num_envs)
        ]

        finished_trajs = []
        num_finished = 0

        while num_finished < num_episodes:
            obs_tensor = th.tensor(
                obs,
                dtype=th.float32,
                device=self.device,
            )

            with th.no_grad():
                actions, _, _, _ = self.policy.forward(obs_tensor)

            actions = actions.cpu().numpy()

            next_obs, rewards, terminated, truncated, _ = self.env.step(actions)
            dones = terminated | truncated

            for env_id in range(num_envs):
                active_trajs[env_id].add(
                    obs=obs[env_id],
                    action=actions[env_id],
                    reward=rewards[env_id],
                    done=dones[env_id],
                )

                if dones[env_id]:
                    traj = active_trajs[env_id].get_trajectory()
                    traj["returns"] = active_trajs[env_id].compute_returns(
                        gamma=self.gamma
                    )

                    finished_trajs.append(traj)
                    num_finished += 1

                    active_trajs[env_id] = SingleTrajctoryBuffer()

                    if num_finished >= num_episodes:
                        break

            obs = next_obs

        with open(save_path, "wb") as f:
            pickle.dump(finished_trajs, f)

        logger.success(
            f"Saved {num_finished} trajectories to {save_path}"
        )
