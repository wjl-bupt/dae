# Only use 
import time
import sys
import torch as th
import numpy as np
from stable_baselines3 import PPO
from gymnasium import spaces
from typing import Dict
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, explained_variance
from typing import Any, Dict, Optional, Type, Union
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

class SPO(PPO):
    """
    Implement SPO: (Advantage) Alpha-Modulation with Proximal Policy Optimization. 
    Paper is from  https://arxiv.org/abs/2505.15514. 
    Code is from https://github.com/Soham4001A/CleanRLfork/blob/main/cleanrl/AMPPO.py
    
    """
    def __init__(
        self,
        policy,
        env,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 128,
        batch_size: Optional[int] = 64,
        n_epochs: int = 4,
        gamma: float = 0.99,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf:Union[float, Schedule] = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        policy_kwargs: Optional[Dict[str, Any]] = dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])], activation_fn=th.nn.Tanh,),
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        tensorboard_log: Optional[str] = None,
        advantage_normalization: bool = True,
    ):

        super(SPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            clip_range=clip_range,
            n_epochs=n_epochs,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            # NOTE(junweiluo)： 新版本的sb需要注释掉这个参数
            # create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=_init_setup_model,
            clip_range_vf=clip_range_vf,
            normalize_advantage = advantage_normalization,
        )
    
    def train(self) -> None:
        
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

        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                mb_advantages = rollout_data.advantages
                # amppo core part
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(mb_advantages) > 1:
                    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # simple policy loss
                # policy_loss_1 = advantages * ratio
                # policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                # policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                
                policy_loss = - (advantages * ratio - (advantages.abs() * (ratio - 1).square() / (2 * max(clip_range, 1e-3)))).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = th.nn.functional.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        if not isinstance(advantages, th.Tensor):
            advantages = th.as_tensor(advantages)
        # ===== 2. 统计信息 =====
        adv_mean = mb_advantages.mean().item()
        adv_std  = mb_advantages.std(unbiased=False).item()
        adv_min  = mb_advantages.min().item()
        adv_max  = mb_advantages.max().item()
        self.logger.record("advantage/advantage_mean", adv_mean, self.num_timesteps)
        self.logger.record("advantage/advantage_std", adv_std, self.num_timesteps)
        self.logger.record("advantage/advantage_min", adv_min, self.num_timesteps)
        self.logger.record("advantage/advantage_max", adv_max, self.num_timesteps)


        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


    def dump_logs(self, iteration: int = 0) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        if iteration > 0:
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)