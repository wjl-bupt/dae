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
from stable_baselines3 import PPO

class VanillaPPO(PPO):
    def train(self) -> None:
        # add some metrics for advantage statistics
        advantages = self.rollout_buffer.advantages
        if not isinstance(advantages, th.Tensor):
            advantages = th.as_tensor(advantages)

        # ===== 2. 统计信息 =====
        adv_mean = advantages.mean().item()
        adv_std  = advantages.std(unbiased=False).item()
        adv_min  = advantages.min().item()
        adv_max  = advantages.max().item()

        # ===== 3. 记录到 logger =====
        self.logger.record("advantage/advantage_mean", adv_mean, self.num_timesteps)
        self.logger.record("advantage/advantage_std", adv_std, self.num_timesteps)
        self.logger.record("advantage/advantage_min", adv_min, self.num_timesteps)
        self.logger.record("advantage/advantage_max", adv_max, self.num_timesteps)
        
        # add some metrics for value function statistics
        values = self.rollout_buffer.values
        if not isinstance(values, th.Tensor):
            values = th.as_tensor(values)
        vf_mean = values.mean().item()
        vf_std  = values.std(unbiased=False).item()
        vf_min  = values.min().item()       
        
        vf_max  = values.max().item()
        
        self.logger.record("value/V_mean", vf_mean, self.num_timesteps)
        self.logger.record("value/V_std", vf_std, self.num_timesteps)
        self.logger.record("value/V_min", vf_min, self.num_timesteps)
        self.logger.record("value/V_max", vf_max, self.num_timesteps)
        
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


