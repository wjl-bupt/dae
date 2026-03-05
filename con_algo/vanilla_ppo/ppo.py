# -*- encoding: utf-8 -*-
'''
@File :ppo.py
@Created-Time :2025-11-29 16:10:47
@Author  :june
@Description   :
@Modified-Time : 2025-11-29 16:10:47
'''

# Only use 

import torch as th
import numpy as np
from stable_baselines3 import PPO
from gymnasium import spaces
from stable_baselines3.common.buffers import  RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

class VanillaPPO(PPO):
    def train(self) -> None:
        # add some metrics for advantage statistics
        advantages = self.rollout_buffer.advantages
        if not isinstance(advantages, th.Tensor):
            advantages = th.as_tensor(advantages)

        # ===== 2. 统计信息 =====
        adv_mean = advantages.mean().item()
        adv_std  = advantages.std().item()
        adv_min  = advantages.min().item()
        adv_max  = advantages.max().item()

        # ===== 3. 记录到 logger =====
        self.logger.record("adv/adv_mean", adv_mean, self.num_timesteps)
        self.logger.record("adv/adv_std", adv_std, self.num_timesteps)
        self.logger.record("adv/adv_min", adv_min, self.num_timesteps)
        self.logger.record("adv/adv_max", adv_max, self.num_timesteps)


        self.logger.record("adv/abs_adv_mean", advantages.abs().mean().item(), self.num_timesteps)
        self.logger.record("adv/abs_adv_std", advantages.abs().std().item(), self.num_timesteps)
        self.logger.record("adv/abs_adv_min", advantages.abs().min().item(), self.num_timesteps)
        self.logger.record("adv/abs_adv_max", advantages.abs().max().item(), self.num_timesteps)
        
        # add some metrics for value function statistics
        values = self.rollout_buffer.values
        if not isinstance(values, th.Tensor):
            values = th.as_tensor(values)
        vf_mean = values.mean().item()
        vf_std  = values.std(unbiased=False).item()
        vf_min  = values.min().item()       
        
        vf_max  = values.max().item()
        
        self.logger.record("values/V_mean", vf_mean, self.num_timesteps)
        self.logger.record("values/V_std", vf_std, self.num_timesteps)
        self.logger.record("values/V_min", vf_min, self.num_timesteps)
        self.logger.record("values/V_max", vf_max, self.num_timesteps)
        
        # add some metrics for actions statistics
        actions = self.rollout_buffer.actions
        if not isinstance(actions, th.Tensor):
            actions = th.as_tensor(actions)
        act_mean = actions.mean().item()
        act_std  = actions.std(unbiased=False).item()
        act_min  = actions.min().item()       
        act_max  = actions.max().item()
        
        self.logger.record("actions/actions_mean", act_mean, self.num_timesteps)
        self.logger.record("actions/actions_std", act_std, self.num_timesteps)
        self.logger.record("actions/actions_min", act_min, self.num_timesteps)  
        self.logger.record("actions/actions_max", act_max, self.num_timesteps)

        # ===== 4. 正常走 PPO 的 train =====
        super().train()


    # def collect_rollouts(
    #     self,
    #     env: VecEnv,
    #     callback: BaseCallback,
    #     rollout_buffer: RolloutBuffer,
    #     n_rollout_steps: int,
    # ) -> bool:
    #     """
    #     Collect experiences using the current policy and fill a ``RolloutBuffer``.
    #     The term rollout here refers to the model-free notion and should not
    #     be used with the concept of rollout used in model-based RL or planning.

    #     :param env: The training environment
    #     :param callback: Callback that will be called at each step
    #         (and at the beginning and end of the rollout)
    #     :param rollout_buffer: Buffer to fill with rollouts
    #     :param n_rollout_steps: Number of experiences to collect per environment
    #     :return: True if function returned with at least `n_rollout_steps`
    #         collected, False if callback terminated rollout prematurely.
    #     """
    #     assert self._last_obs is not None, "No previous observation was provided"
    #     # Switch to eval mode (this affects batch norm / dropout)
    #     self.policy.set_training_mode(False)

    #     n_steps = 0
    #     rollout_buffer.reset()
    #     # Sample new weights for the state dependent exploration
    #     if self.use_sde:
    #         self.policy.reset_noise(env.num_envs)

    #     callback.on_rollout_start()

    #     while n_steps < n_rollout_steps:
    #         if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
    #             # Sample a new noise matrix
    #             self.policy.reset_noise(env.num_envs)

    #         with th.no_grad():
    #             # Convert to pytorch tensor or to TensorDict
    #             obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]
    #             actions, values, log_probs = self.policy(obs_tensor)
    #         actions = actions.cpu().numpy()

    #         # Rescale and perform action
    #         clipped_actions = actions

    #         if isinstance(self.action_space, spaces.Box):
    #             if self.policy.squash_output:
    #                 # Unscale the actions to match env bounds
    #                 # if they were previously squashed (scaled in [-1, 1])
    #                 clipped_actions = self.policy.unscale_action(clipped_actions)
    #             else:
    #                 pass
    #                 # Otherwise, clip the actions to avoid out of bound error
    #                 # as we are sampling from an unbounded Gaussian distribution
    #                 # clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

    #         new_obs, rewards, dones, infos = env.step(clipped_actions)

    #         self.num_timesteps += env.num_envs

    #         # Give access to local variables
    #         callback.update_locals(locals())
    #         if not callback.on_step():
    #             return False

    #         self._update_info_buffer(infos, dones)
    #         n_steps += 1

    #         if isinstance(self.action_space, spaces.Discrete):
    #             # Reshape in case of discrete action
    #             actions = actions.reshape(-1, 1)

    #         # Handle timeout by bootstrapping with value function
    #         # see GitHub issue #633
    #         for idx, done in enumerate(dones):
    #             if (
    #                 done
    #                 and infos[idx].get("terminal_observation") is not None
    #                 and infos[idx].get("TimeLimit.truncated", False)
    #             ):
    #                 terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
    #                 with th.no_grad():
    #                     terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
    #                 rewards[idx] += self.gamma * terminal_value

    #         rollout_buffer.add(
    #             self._last_obs,  # type: ignore[arg-type]
    #             actions,
    #             rewards,
    #             self._last_episode_starts,  # type: ignore[arg-type]
    #             values,
    #             log_probs,
    #         )
    #         self._last_obs = new_obs  # type: ignore[assignment]
    #         self._last_episode_starts = dones

    #     with th.no_grad():
    #         # Compute value for the last timestep
    #         values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

    #     rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

    #     callback.update_locals(locals())

    #     callback.on_rollout_end()

    #     return True