# -*- encoding: utf-8 -*-
'''
@File :ppo.py
@Created-Time :2025-11-29 09:59:14
@Author  :june
@Description   :
@Modified-Time : 2025-11-29 09:59:14
'''

# 
from typing import Any, Dict, Optional, Type, Union
from functools import partial

import sys
import time
import numpy as np
import torch as th
import warnings
import wandb
from torch.optim  import Adam

from gymnasium import spaces
from con_algo.dae.policy import CustomActorCriticPolicy
from con_algo.dae.buffer import CustomBuffer
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_schedule_fn, update_learning_rate
from stable_baselines3.common.vec_env import VecEnv


class CustomPPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) with direct advantage estimation

    Paper: https://arxiv.org/abs/1707.06347
    Code: Code borrowed from Stable Baselines 3 (https://github.com/DLR-RM/stable-baselines3)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param learning_rate_vf: The learning rate for the value function, it can be a function
        of the current progress remaining (from 1 to 0) (only used when actor/critic are separeted)
    :param n_steps: The number of steps to run for each environment per update
    :param batch_size: Minibatch size
    :param batch_size_vf: Minibatch size for critic training (only used when actor/critic are separeted)
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param n_epochs_vf: Number of epoch when optimizing the surrogate loss for critic training (only used when actor/critic are separeted)
    :param gamma: Discount factor
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param ent_coef: Entropy coefficient for the loss calculation
    :param kl_coef: KL penalty coefficient for actor training
    :param vf_coef: Value function coefficient for the loss calculation
    :param shared: Use shared network for actor/critic (seperate training is)
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param advantage_normalization: normalize the estimated advantage before computing PPO loss
    :param full_action: update policy using all actions instead of just the sampled ones
    :param dae_correction: compute critic loss with DAE-style (multistep advantage) or DuelingDQN-style (first step advantage)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Type[CustomActorCriticPolicy],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        learning_rate_vf: Union[float, Schedule] = 3e-4,
        n_steps: int = 128,
        batch_size: Optional[int] = 64,
        batch_size_vf: Optional[int] = 8,
        n_epochs: int = 4,
        n_epochs_vf: int = 4,
        gamma: float = 0.99,
        clip_range: Union[float, Schedule] = 0.2,
        ent_coef: float = 0.01,
        kl_coef: float = 0.0,
        vf_coef: float = 0.5,
        shared: bool = False,
        max_grad_norm: float = 0.5,
        tensorboard_log: Optional[str] = None,
        advantage_normalization: bool = False,
        full_action: bool = True,
        dae_correction: bool = True,
        dae_discouple_correction: bool = True,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])], activation_fn=th.nn.Tanh,),
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        nheads: int = 2,
        gae_like_lambda: float = 0.0,
        use_sub_action_ratio: bool = True,
        corr_coef: float = 0.2,
        use_huber_loss: bool = True,
        
    ):  
        super(CustomPPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=0,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=False,
            sde_sample_freq=-1,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            # NOTE(junweiluo)： 新版本的sb需要注释掉这个参数
            # create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            # NOTE(junweiluo): 
            supported_action_spaces=(spaces.Box, ),
            
        )

        self.batch_size = batch_size
        self.batch_size_vf = batch_size_vf
        self.learning_rate_vf = learning_rate_vf
        self.n_epochs = n_epochs
        self.n_epochs_vf = n_epochs_vf
        self.advantage_normalization = advantage_normalization
        self.full_action = full_action
        self.clip_range = clip_range
        self.kl_coef = kl_coef
        self.shared = shared
        self.nheads = nheads
        self.use_sub_action_ratio = use_sub_action_ratio
        self.learning_rate_vf = learning_rate_vf
        self.corr_coef = corr_coef
        self.use_huber_loss = use_huber_loss

        if not shared:
            warnings.warn(
                "Training with seperate actor/critic is deprecated, use at your own risk"
            )
        # self.dae_correction = dae_correction
        self.dae_discouple_correction = dae_discouple_correction
        self.gae_like_lambda = gae_like_lambda
        self.gl = self.gamma * self.gae_like_lambda
        self.discount_matrix = th.tensor(
            [
                [0 if j < i else (self.gamma) ** (j - i) for j in range(n_steps)]
                for i in range(n_steps)
            ],
            dtype=th.float32,
            device=self.device,
        )
        
        self.gae_like_lambdadiscount_matrix = th.tensor(
            [
                [0 if j < i else (self.gamma * self.gae_like_lambda) ** (j - i) for j in range(n_steps)]
                for i in range(n_steps)
            ],
            dtype=th.float32,
            device=self.device,
        )
        
        self.discount_vector = self.gamma ** th.arange(
            n_steps, 0, -1, dtype=th.float32, device=self.device
        )

        if _init_setup_model:
            self._setup_model()

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: CustomBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"

        self.policy.eval()

        rollout_buffer.reset()

        callback.on_rollout_start()
        for _ in range(n_rollout_steps):
            # self._last_obs = self._last_obs / 10.0 
            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs, device=self.device)
                actions, mu, log_policies, values, noise = self.policy.forward(
                    obs_tensor
                )
            # actions = th.clamp(actions, self.policy.action_space.low, self.policy.action_space.high).cpu().numpy()
            actions = actions.cpu().numpy()
            mu = mu.cpu().numpy()
            # np.clip(actions, self.policy.action_space.low, self.policy.action_space.high)
            new_obs, rewards, dones, infos = env.step(
                # actions
                np.clip(actions, self.action_space.low, self.action_space.high)
            )
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)

            # NOTE(junweiluo) reshape actions for discrete action space. But here action space is continuous.
            # actions = actions.reshape(-1, 1)
            rollout_buffer.add(
                self._last_obs, actions, mu, rewards, dones, log_policies, values, noise,
            )

            self._last_obs = new_obs

        with th.no_grad():
            obs_tensor = th.as_tensor(new_obs, device=self.device)
            values, _ = self.policy.predict_value(obs_tensor)
        rollout_buffer.finalize(last_values=values)

        callback.on_rollout_end()

        return True

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.rollout_buffer = CustomBuffer(
            buffer_size=self.n_steps,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            n_envs=self.n_envs,
        )

        self.policy = self.policy_class(
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=self.lr_schedule,
            lr_schedule_vf=self.lr_schedule,
            shared_features_extractor=self.shared,
            nheads = self.nheads,
            learning_rate_vf = self.learning_rate_vf,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)

    def _setup_lr_schedule(self) -> None:
        self.lr_schedule = get_schedule_fn(self.learning_rate)
        self.lr_schedule_vf = get_schedule_fn(self.learning_rate_vf)

    def _update_learning_rate(self, optimizer, schedule, suffix=""):
        
        if suffix == "_pi":
            new_lr = schedule(self._current_progress_remaining)
            update_learning_rate(optimizer, new_lr)
        elif suffix == "_vf":
            new_lr = self.learning_rate_vf * self._current_progress_remaining
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
        else:
            new_lr = schedule(self._current_progress_remaining)
            update_learning_rate(optimizer, new_lr)
        self.logger.record(f"train/learning_rate{suffix}", new_lr)

    def _normalize_advantage(self, advantages, policies = None, eps=1e-8):

        return advantages / (advantages.std() + eps)
        return (advantages - advantages.mean() ) / (advantages.std() + eps)

    def _value_loss(self, rewards, advantages, values, lasts):
        # use dae original mse loss
        if self.use_huber_loss == False:
            loss = th.cat(
                [
                    (
                        self.discount_matrix[: len(r), : len(r)].matmul(r)
                        - self.discount_matrix[: len(a), : len(a)].matmul(a)
                        + l * self.discount_vector[-len(r) :]
                        - v
                    ).square()
                    for r, a, v, l in zip(rewards, advantages, values, lasts)
                ]
            ).mean()
            # Alignment function return.
            return loss, 0.0
        # use Adaptive Scale Huber Loss
        # \beta = Std(G_t)
        # Loss = th.nn.functional.smooth_l1_loss
        else:
            preds = []
            targets = []
            for r, a, v, l in zip(rewards, advantages, values, lasts):
                target = (
                    self.discount_matrix[: len(r), : len(r)].matmul(r)
                    - self.discount_matrix[: len(a), : len(a)].matmul(a)
                    + l * self.discount_vector[-len(r):]
                )

                preds.append(v)
                targets.append(target)

            preds = th.cat(preds)
            targets = th.cat(targets)
            beta = targets.std().detach().item()
            loss = th.nn.functional.smooth_l1_loss(preds, targets, beta=beta)

            return loss, beta

    def _policy_loss(
        self, advantages, log_policy, old_log_policy, actions, clip_range=None
    ):

        if self.full_action:
            # TODO(junweiluo): implement full-action ratio in mujoco
            pass
        else:
            adv = advantages
            logp = log_policy
            old_logp = old_log_policy
            if self.use_sub_action_ratio:
                ratio = th.exp(logp - old_logp)
                #  / self.action_space.shape[0]
                adv = adv.unsqueeze(-1)
                policy_loss_1 = adv * ratio
                policy_loss_2 = adv * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                loss = -th.min(policy_loss_1, policy_loss_2).mean()
            else:
                log_ratio = (logp - old_logp).sum(dim = 1)
                ratio = log_ratio.exp()
                policy_loss_1 = adv * ratio
                policy_loss_2 = adv * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                loss = -th.min(policy_loss_1, policy_loss_2).mean()
            

        return loss, ratio

    def _compute_gae_like_advantages_(self, raw_advantages, lengths): 
        # give a time coef 
        # gae_like_coef = self.gamma * (self.gae_like_lambda * (1 - self._current_progress_remaining)) 
        # gae_like_coef = self.gamma * self.gae_like_lambda 
        # dones = th.zeros_like(raw_advantages) 
        # lengths = th.tensor(lengths) 
        # index = th.cumsum(lengths, dim=0) - 1 
        # dones[index.to(raw_advantages.device)] = 1 
        # gae_like_advantages = th.zeros_like(raw_advantages) 
        # gaelikelam = 0 
        # for t in reversed(range(lengths.sum())): 
        #     gaelikelam = raw_advantages[t] + (1 - dones[t]) * gae_like_coef * gaelikelam 
        #     gae_like_advantages[t] = gaelikelam 

        device = raw_advantages.device
        coef = self.gamma * self.gae_like_lambda
        B = raw_advantages.shape[0]
        lengths = th.tensor(lengths, device=device)
        # dones
        dones = th.zeros(B, device=device)
        dones[th.cumsum(lengths, dim=0) - 1] = 1.0
        # reverse
        adv = th.flip(raw_advantages, dims=[0])
        done = th.flip(dones, dims=[0])

        out = th.zeros_like(adv)

        acc = th.zeros(1, device=device)

        # ⚠️ 这里仍然是 loop，但在 GPU 上很快
        for t in range(B):
            acc = adv[t] + coef * (1 - done[t]) * acc
            out[t] = acc   
        gae_like_advantages = th.flip(out, dims=[0])     

        return gae_like_advantages

    def _compute_td_error(self, rewards, values, target_values, last_values, lengths, gamma=0.99):
        """
        rewards:        (T,)
        values:         (T,)
        target_values:  (T,)
        last_values:    (num_episodes,)
        lengths:        list[num_episodes]
        """
        if len(values.shape) > 1:
            values = values.squeeze(-1)
        if len(target_values.shape) > 1:
            target_values = target_values.squeeze(-1) 

        device = rewards.device
        lengths = th.tensor(lengths, device=device)
        last_values = th.tensor(last_values, dtype=th.float32, device=device)
        T = rewards.shape[0]
        # next state values
        next_values = th.zeros_like(values)
        # episode end index
        end_idx = th.cumsum(lengths, dim=0) - 1
        # 默认 next value = target_values[t+1]
        next_values[:-1] = target_values[1:]
        # episode 结束位置使用 last_values
        next_values[end_idx] = last_values
        # TD error
        deltas = rewards + gamma * next_values - values

        return deltas


    def _train_shared(self) -> None:

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer, self.lr_schedule)

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)

        entropy_losses, clip_fractions, gnorms = [], [], []
        losses, value_losses, pg_losses, kl_divs = [], [], [], []
        gnorm_max, gnorm_min = 0, float("inf")
        q_values = []
        log_std = self.policy.log_std.detach()
        # log_std = th.clamp(log_std, -4, 2)

        # with th.no_grad():
        #     self.rollout_buffer.update_advantage(self.policy, log_std = log_std, batch_size =self.batch_size, gamma = self.gamma, gae_like_lambda = self.gae_like_lambda)        


        for epoch in range(self.n_epochs):
            with th.no_grad():
                self.rollout_buffer.update_advantage(self.policy, log_std = log_std, batch_size =self.batch_size)
            #     self.rollout_buffer.update_value(self.policy, batch_size =self.batch_size)
            ex_advs = []
            # if self.advantage_normalization:
            #     self.rollout_buffer.advantages = (self.rollout_buffer.advantages - self.rollout_buffer.advantages.mean()) / (self.rollout_buffer.advantages.std() + 1e-8)

            for data in self.rollout_buffer.get_trajs(batch_size=self.batch_size):
                # old_policies = data.old_policies
                old_log_policies = data.old_log_policies
                actions = data.actions
                mu = data.mu
                rewards = data.rewards
                last_values = data.last_values
                lengths = data.lengths
                old_advantages = data.advantages
                target_values = data.values
                # log_std = data.noises

                (
                    values,
                    advantages,
                    log_policies,
                    entropy,
                    scores,
                    divs,
                    fs,
                ) = self.policy.evaluate_state(data.observations, actions, mu, log_std, log_std)

                # value loss
                values = values.flatten().split(lengths)
                
                
                # origin loss
                if self.dae_discouple_correction == False:
                    # - advantages
                    deltas = (
                        rewards - advantages
                    ).split(lengths)
                    # pred_values = target_values + th.clamp(th.cat(values) - target_values, - 0.2, 0.2)
                    main_value_loss, beta = self._value_loss(
                        rewards.split(lengths),
                        advantages.split(lengths),
                        values,
                        last_values
                    )
                    advantages_ = advantages.detach().clone()
                    td_error = self._compute_td_error(rewards , target_values, target_values, last_values, lengths, gamma = 0.99)
                    td_loss = 0.5 * (advantages - td_error).square().mean()
                    corr = ((advantages - advantages.mean()) * (td_error - td_error.mean())).mean() / (td_error.std(unbiased = False) * advantages.std(unbiased = False) + 1e-10)
                    td_direct_corr = ((advantages * td_error) > 0).sum() / advantages.shape[0]
                    value_loss = main_value_loss + self.corr_coef * (1 - corr)
                    # advantages_ = self._compute_gae_like_advantages_(advantages_, lengths)
                # discouple loss
                else:
                    # discouple dae loss
                    # 1. we train value network with dae-loss (advantage detach, value keep grad)
                    #  \sum_{k=0}^{T-t-1} \gamma^{k}(r_{t+k} - \hat{A}_{t+k}.detach()) + \gamma^{T-t} V_{T-t+1} - V_{t}
                    # 2. we add a regularization for td error: 
                    #  (r_{t} + \gamma V_{t+1}^{target} - V_{t} - \hat{A}_{t})^2
                    
                    # calculate original dae loss with detach A(s,a)
                    advantages_ = advantages.detach().clone()
                    deltas = (rewards - old_advantages).split(lengths)
                    pred_values = target_values + th.clamp(th.cat(values) - target_values, - 0.2, 0.2)
                    main_value_loss, beta = self._value_loss(
                        rewards.split(lengths),
                        advantages.split(lengths),
                        # values,
                        pred_values.flatten().split(lengths), 
                        last_values
                    )
                    main_value_loss = main_value_loss.mean()
                    next_advantages = self.gamma * self.gae_like_lambda * th.roll(old_advantages, -1)
                    next_advantages[th.cumsum(th.tensor(lengths), 0) - 1] = 0
                    # main_value_loss = (main_value_loss / (div.pow(2) + 1e-10) + 2  * div.log()).mean()
                    # calculate a auxlimary regularization for td error
                    # don't optimizer combine loss
                    td_error = self._compute_td_error(rewards , target_values, target_values, last_values, lengths, gamma = self.gamma)
                    td_loss = (0.5 * (advantages - next_advantages - td_error).square()).mean()
                    # td_loss = th.nn.functional.huber_loss(advanges_norm_, td_error_norm, delta = 1.0).mean()
                    td_direct_corr = ((advantages * td_error) > 0).sum() / advantages.shape[0]
                    corr = ((advantages - advantages.mean()) * (td_error - td_error.mean())).mean() / (td_error.std(unbiased = False) * advantages.std(unbiased = False) + 1e-10)
                    # 0.2 * td_error.var()  - 0.1 * advantages.var()
                    # add a coef for td loss
                    value_loss = main_value_loss + 0.1 * td_loss
                    # advantages_ = self._compute_gae_like_advantages_(advantages_, lengths)

                
                # kl divergence
                # kl_loss = (
                #     (old_policies * (old_log_policies - log_policies)).sum(dim=1).mean()
                # )

                # normalize adv
                advantages_ = old_advantages.clone()
                if self.advantage_normalization:
                    advantages_norm = self._normalize_advantage(advantages_, policies = None)

                # policy loss
                policy_loss, ratio = self._policy_loss(
                    advantages_norm, log_policies, old_log_policies, actions, clip_range
                )

                # entropy loss
                # entropy_loss = -th.mean(entropy)
                entropy_loss = - th.mean(entropy)
                # entropy_loss = th.mean(self.policy.log_alpha.exp() * (entropy - self.policy.target_entropy)**2)
                # entropy_loss = th.mean((- log_policies.detach() + self.policy.target_entropy) * self.policy.log_alpha.exp())
                # full loss
                kl_loss = ((ratio - 1) - ratio.log()).mean()
                
                # avoid collapse
                # k = th.arange(1, self.action_space.shape[0] + 1, 1).view(1, 1, self.action_space.shape[0]).to(ws.device)
                # ac_loss = (ws * k).pow(2).mean()                
                # log_alpha
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.kl_coef * kl_loss
                    + self.vf_coef * value_loss
                    # + self.vf_coef * ex_f.pow(2).mean()
                )

                losses.append(loss.item())
                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = th.norm(
                    th.stack(
                        [
                            th.norm(p.grad)
                            for p in self.policy.parameters()
                            if p.grad is not None
                        ]
                    )
                ).item()
                gnorm_max = max(gnorm_max, gnorm)
                gnorm_min = min(gnorm_min, gnorm)

                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                self.policy.optimizer.step()
                # Logging
                clip_fractions.append(
                    th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                )
                pg_losses.append(policy_loss.item())
                value_losses.append(main_value_loss.item())
                entropy_losses.append(-entropy.mean().item())
                kl_divs.append(kl_loss.item())
                gnorms.append(gnorm)

                # ex_advs.extend(ex_adv.detach().cpu().numpy().flatten().tolist())
                self._n_updates += 1
            
            # if kl_loss.mean().item() >= 0.05:
            #     break
            
            # self.policy.ema_ex_adv = self.policy.ema_ex_adv * self.policy.ema_coef + np.mean(ex_advs).item() * (1 - self.policy.ema_coef)

        # Logs
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        # self.logger.record("train/entropy_losses", np.mean(entropy_losses))
        self.logger.record("log_policies/policy_min", log_policies.sum(dim=1).detach().cpu().min().item())
        self.logger.record("log_policies/policy_max", log_policies.sum(dim=1).detach().cpu().max().item())
        self.logger.record("log_policies/policy_mean", log_policies.sum(dim=1).detach().cpu().mean().item())
        
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/gnorm", np.mean(gnorms))
        self.logger.record("train/gnorm_max", np.mean(gnorm_max))
        self.logger.record("train/gnorm_min", np.mean(gnorm_min))
        self.logger.record("train/approx_kl", np.mean(kl_divs))
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ratio_mean", ratio.detach().cpu().mean().item())
        # self.logger.record("train/trace", trace.cpu().mean().item())
        self.logger.record("train/log_std", log_std.mean().item())  
        self.logger.record("train/std", log_std.exp().mean().item()) 
        self.logger.record("train/td_loss", td_loss.detach().cpu().mean().item())
        self.logger.record("train/td_direct_corr", td_direct_corr.detach().cpu().mean().item())
        self.logger.record("train/rew_adv_delta", th.cat(deltas).detach().cpu().mean().item())
        self.logger.record("train/cur_corr_coef", self.corr_coef)
        self.logger.record("train/huber_loss_beta", beta)
        # advantage & td error correction
        # corr = th.corrcoef(th.stack([advantages, td_error]))[0,1]

        self.logger.record("train/corr", corr.detach().cpu().mean().item())
        self.logger.record("td/td_error_mean", td_error.detach().cpu().mean().item())
        self.logger.record("td/td_error_max", td_error.detach().cpu().max().item())
        self.logger.record("td/td_error_min", td_error.detach().cpu().min().item())
        self.logger.record("td/td_error_std", td_error.detach().cpu().std().item())
        
        # self.logger.record("train/alpha", self.policy.log_alpha.exp().item()) 

        # add some metric to log.
        concat_values = th.concat(values, dim = -1)
        q_values = (concat_values + advantages).detach()
        self.logger.record("advantage/advantage_mean", advantages_.cpu().mean().item())
        self.logger.record("advantage/advantage_std", advantages_.cpu().std().item())
        self.logger.record("advantage/advantage_max", advantages_.cpu().max().item())
        self.logger.record("advantage/advantage_min", advantages_.cpu().min().item())
        
        self.logger.record("advantage/abs_advantage_mean", advantages_.abs().cpu().mean().item())
        self.logger.record("advantage/abs_advantage_std", advantages_.abs().cpu().std().item())
        self.logger.record("advantage/abs_advantage_max", advantages_.abs().cpu().max().item())
        self.logger.record("advantage/abs_advantage_min", advantages_.abs().cpu().min().item())

        self.logger.record("advantage/dae_advantage_mean", advantages.detach().cpu().mean().item())
        self.logger.record("advantage/dae_advantage_std", advantages.detach().cpu().std().item())
        self.logger.record("advantage/dae_advantage_max", advantages.detach().cpu().max().item())
        self.logger.record("advantage/dae_advantage_min", advantages.detach().cpu().min().item())

        self.logger.record("values/V_mean", concat_values.detach().cpu().mean().item())
        self.logger.record("values/V_std", concat_values.detach().cpu().std().item())
        self.logger.record("values/V_max", concat_values.detach().cpu().max().item())
        self.logger.record("values/V_min", concat_values.detach().cpu().min().item())

        self.logger.record("Q_values/Q_values_mean", q_values.detach().cpu().mean().item())
        self.logger.record("Q_values/Q_values_std", q_values.detach().cpu().std().item())
        self.logger.record("Q_values/Q_values_max", q_values.detach().cpu().max().item())
        self.logger.record("Q_values/Q_values_min", q_values.detach().cpu().min().item())

        self.logger.record("actions/actions_mean", self.rollout_buffer.actions.mean().item())
        self.logger.record("actions/actions_max", self.rollout_buffer.actions.max().item())
        self.logger.record("actions/actions_min", self.rollout_buffer.actions.min().item())
        self.logger.record("actions/actions_std", self.rollout_buffer.actions.std().item())

        
        self.logger.record("rewards/reward_mean", self.rollout_buffer.rewards.mean().item())
        self.logger.record("rewards/reward_max", self.rollout_buffer.rewards.max().item())
        self.logger.record("rewards/reward_min", self.rollout_buffer.rewards.min().item())
        
        self.logger.record("weights/weights_max", fs.detach().cpu().max().item())
        self.logger.record("weights/weights_min", fs.detach().cpu().min().item())
        self.logger.record("weights/weights_mean", fs.detach().cpu().mean().item())
        self.logger.record("weights/weights_std", fs.detach().cpu().std().item())
        
        self.logger.record("div/div_max", divs.detach().cpu().max().item())
        self.logger.record("div/div_mean", divs.detach().cpu().mean().item())
        self.logger.record("div/div_min", divs.detach().cpu().min().item())
        self.logger.record("div/div_std", divs.detach().cpu().std().item())

        self.logger.record("scores/scores_max", scores.detach().cpu().max().item())
        self.logger.record("scores/scores_mean", scores.detach().cpu().mean().item())
        self.logger.record("scores/scores_min", scores.detach().cpu().min().item())
        self.logger.record("scores/scores_std", scores.detach().cpu().std().item())
        # 计算一下value network的评估是否准确
        targets = []
        preds = []

        for d, v, l in zip(deltas, values, last_values):

            target = (
                self.discount_matrix[:len(d), :len(d)].matmul(d)
                + l * self.discount_vector[-len(d):]
            )

            targets.append(target)
            preds.append(v)

        targets = th.cat(targets)
        preds = th.cat(preds)

        var_y = th.var(targets)
        if var_y < 1e-8:
            var_y =  th.tensor(0.0)

        explain_var =  1 - th.var(targets - preds) / var_y  
        self.logger.record("train/explained_variance", explain_var.cpu().mean().item())

    def _train_separate(self) -> None:
        cur_corr_coef = self.corr_coef
        # Update optimizer learning rate
        self._update_learning_rate(
            self.policy.optimizer, self.lr_schedule, suffix="_pi"
        )
        self._update_learning_rate(
            self.policy.optimizer_vf, self.lr_schedule_vf, suffix="_vf"
        )

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)

        entropy_losses, clip_fractions, kl_divs = [], [], []
        value_losses, pg_losses = [], []
        gnorm_pi, gnorm_vf = [], []
        gnorm_pi_max, gnorm_vf_max = 0.0, 0.0

        # train for n_epochs epochs
        old_log_std = self.policy.log_std.detach()
        
        self.policy.zero_grad(set_to_none=True)
        self.policy.optimizer.zero_grad(set_to_none=True)
        self.policy.optimizer_vf.zero_grad(set_to_none=True)
        for epoch in range(self.n_epochs_vf):
            for data in self.rollout_buffer.get_trajs(batch_size=self.batch_size_vf):
                old_log_policies = data.old_log_policies
                actions = data.actions
                mu = data.mu
                rewards = data.rewards
                last_values = data.last_values
                lengths = data.lengths
                target_values = data.values

                (
                    values, 
                    advantages,
                    scores,
                    divs,
                    fs,
                ) = self.policy.predict_value(
                    data.observations, actions, mu, old_log_std, noise = data.noises, return_all = True
                )
                # value loss
                values = values.flatten().split(lengths)
                
                # pred_values = target_values + th.clamp(th.cat(values) - target_values, - 0.2, 0.2)
                main_value_loss, beta = self._value_loss(
                    rewards.split(lengths), 
                    advantages.split(lengths), 
                    # pred_values.split(lengths), 
                    values,
                    last_values,
                )
                td_error = self._compute_td_error(rewards , target_values, target_values, last_values, lengths, gamma = 0.99)
                td_loss = 0.5 * (advantages - td_error).square().mean()
                corr = ((advantages - advantages.mean()) * (td_error - td_error.mean())).mean() \
                    / (td_error.std(unbiased = False) * advantages.std(unbiased = False) + 1e-10)
                td_direct_corr = ((advantages * td_error) > 0).sum() / advantages.shape[0]
                #  + 0.1 * (1 - corr) 
                # + cur_corr_coef * (1 - corr)
                value_loss = main_value_loss + cur_corr_coef * (1 - corr) 
                # value_loss = self.vf_coef * value_loss + 0.1 * (1.0 / (advantages.std() + 1.0)).mean() 
                # value_loss += (ex_adv**2).mean()
                # value_loss = value_loss + 0.2 * (ex_adv**2).mean()
                # add a new loss penalty
                self.policy.optimizer_vf.zero_grad(set_to_none=True)
                value_loss.backward()
                gnorm = th.norm(
                    th.stack(
                        [
                            th.norm(p.grad)
                            for p in self.policy.parameters()
                            if p.grad is not None
                        ]
                    )
                )
                th.nn.utils.clip_grad_norm_(self.policy.modules_vf.parameters(), self.max_grad_norm)
                self.policy.optimizer_vf.step()

                # logging
                value_losses.append(main_value_loss.item())
                gnorm_vf.append(gnorm.item())
                gnorm_vf_max = max(gnorm_vf_max, gnorm.item())


        self.logger.record("train/td_loss", td_loss.detach().cpu().mean().item())
        self.logger.record("train/td_direct_corr", td_direct_corr.detach().cpu().mean().item())
        self.logger.record("train/rew_adv_delta", (rewards - advantages).detach().cpu().mean().item())
        self.logger.record("train/corr", corr.detach().cpu().mean().item())
        self.logger.record("td/td_error_mean", td_error.detach().cpu().mean().item())
        self.logger.record("td/td_error_max", td_error.detach().cpu().max().item())
        self.logger.record("td/td_error_min", td_error.detach().cpu().min().item())
        self.logger.record("td/td_error_std", td_error.detach().cpu().std().item())

        concat_values = th.concat(values, dim = -1)
        q_values = (concat_values + advantages).detach()
        self.logger.record("advantage/dae_advantage_mean", advantages.detach().cpu().mean().item())
        self.logger.record("advantage/dae_advantage_std", advantages.detach().cpu().std().item())
        self.logger.record("advantage/dae_advantage_max", advantages.detach().cpu().max().item())
        self.logger.record("advantage/dae_advantage_min", advantages.detach().cpu().min().item())

        self.logger.record("values/V_mean", concat_values.detach().cpu().mean().item())
        self.logger.record("values/V_std", concat_values.detach().cpu().std().item())
        self.logger.record("values/V_max", concat_values.detach().cpu().max().item())
        self.logger.record("values/V_min", concat_values.detach().cpu().min().item())

        self.logger.record("Q_values/Q_values_mean", q_values.detach().cpu().mean().item())
        self.logger.record("Q_values/Q_values_std", q_values.detach().cpu().std().item())
        self.logger.record("Q_values/Q_values_max", q_values.detach().cpu().max().item())
        self.logger.record("Q_values/Q_values_min", q_values.detach().cpu().min().item())
        # 
        self.logger.record("actions/actions_mean", actions.mean().item())
        self.logger.record("actions/actions_max", actions.max().item())
        self.logger.record("actions/actions_min", actions.min().item())
        self.logger.record("actions/actions_std", actions.std().item())

        self.logger.record("weights/weights_max", fs.detach().cpu().max().item())
        self.logger.record("weights/weights_min", fs.detach().cpu().min().item())
        self.logger.record("weights/weights_mean", fs.detach().cpu().mean().item())
        self.logger.record("weights/weights_std", fs.detach().cpu().std().item())

        self.logger.record("div/div_max", divs.detach().cpu().max().item())
        self.logger.record("div/div_mean", divs.detach().cpu().mean().item())
        self.logger.record("div/div_min", divs.detach().cpu().min().item())
        self.logger.record("div/div_std", divs.detach().cpu().std().item())

        self.logger.record("scores/scores_max", scores.detach().cpu().max().item())
        self.logger.record("scores/scores_mean", scores.detach().cpu().mean().item())
        self.logger.record("scores/scores_min", scores.detach().cpu().min().item())
        self.logger.record("scores/scores_std", scores.detach().cpu().std().item())
        
        self.logger.record("train/cur_corr_coef", cur_corr_coef)
        self.logger.record("train/huber_loss_beta", beta)
        
        # 计算一下value network的评估是否准确
        targets = []
        preds = []
        deltas = (rewards - advantages).split(lengths)
        for d, v, l in zip(deltas, values, last_values):

            target = (
                self.discount_matrix[: len(d), : len(d)].matmul(d)
                + l * self.discount_vector[-len(d):]
            )

            targets.append(target)
            preds.append(v)

        targets = th.cat(targets)
        preds = th.cat(preds)

        var_y = th.var(targets)
        if var_y < 1e-8:
            var_y =  th.tensor(0.0)
        explain_var =  1 - th.var(targets - preds) / (var_y + 1e-12)  
        self.logger.record("train/explained_variance", explain_var.detach().cpu().mean().item())

        # self.logger.record("tanh_sigma")

        self.rollout_buffer.update_advantage(
            self.policy, 
            log_std = old_log_std, 
            batch_size =self.batch_size, 
            gamma = self.gamma, 
            gae_like_lambda = self.gae_like_lambda
        )
        
        # if self.advantage_normalization:
        #     self.rollout_buffer.advantages = (self.rollout_buffer.advantages - self.rollout_buffer.advantages.mean()) / (self.rollout_buffer.advantages.std() + 1e-8)
        self.policy.zero_grad(set_to_none=True)
        self.policy.optimizer.zero_grad(set_to_none=True)
        self.policy.optimizer_vf.zero_grad(set_to_none=True)
        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                old_log_policies = rollout_data.old_log_policies
                actions = rollout_data.actions
                tanh_w_actions = rollout_data.actions
                mu = rollout_data.mu
                old_advantages = rollout_data.advantages


                log_policies, entropy = self.policy.predict_policy(
                    rollout_data.observations, actions, tanh_w_actions
                )

                # kl divergence
                # kl_div = (
                #     (old_log_policies * (old_log_policies - log_policies)).sum(dim=1).mean()
                # )

                # Normalize advantage
                if self.advantage_normalization:
                    advantages_norm = self._normalize_advantage(old_advantages, old_log_policies.exp())
                

                # policy loss
                policy_loss, ratio = self._policy_loss(
                    advantages_norm, log_policies, old_log_policies, actions, clip_range
                )
                
                kl_div = ((ratio - 1) - ratio.log()).mean()

                # entropy loss
                entropy_loss = -th.mean(entropy)

                # full loss
                loss = (
                    policy_loss + self.ent_coef * entropy_loss + self.kl_coef * kl_div
                )

                # Optimization step
                self.policy.optimizer.zero_grad(set_to_none=True)

                loss.backward()
                gnorm = th.norm(
                    th.stack(
                        [
                            th.norm(p.grad)
                            for p in self.policy.parameters()
                            if p.grad is not None
                        ]
                    )
                ).item()

                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.modules_pi, self.max_grad_norm)
                self.policy.optimizer.step()

                # Logging
                gnorm_pi.append(gnorm)
                gnorm_pi_max = max(gnorm_pi_max, gnorm)
                pg_losses.append(policy_loss.item())
                clip_fractions.append(
                    th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                )
                entropy_losses.append(entropy_loss.item())
                kl_divs.append(kl_div.item())

            # if kl_div >= 0.05:
            #     break

        self.logger.record("advantage/advantage_mean", old_advantages.cpu().mean().item())
        self.logger.record("advantage/advantage_std", old_advantages.cpu().std().item())
        self.logger.record("advantage/advantage_max", old_advantages.cpu().max().item())
        self.logger.record("advantage/advantage_min", old_advantages.cpu().min().item())
        
        self.logger.record("advantage/abs_advantage_mean", old_advantages.abs().cpu().mean().item())
        self.logger.record("advantage/abs_advantage_std", old_advantages.abs().cpu().std().item())
        self.logger.record("advantage/abs_advantage_max", old_advantages.abs().cpu().max().item())
        self.logger.record("advantage/abs_advantage_min", old_advantages.abs().cpu().min().item())

        self._n_updates += self.n_epochs
        # Logs
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/gnorm_vf", np.mean(gnorm_vf))
        self.logger.record("train/gnorm_vf_max", gnorm_vf_max)
        self.logger.record("train/gnorm_pi", np.mean(gnorm_pi))
        self.logger.record("train/gnorm_pi_max", gnorm_pi_max)
        self.logger.record("train/approx_kl", np.mean(kl_divs))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        
        self.logger.record("train/log_std", old_log_std.cpu().mean().item())  
        # self.logger.record("train/tanh_std", (1 - tanh_mu.pow(2)) / (1 + (2 / th.sqrt(1 + (th.pi * sigma.pow(2) / 4)))))
        self.logger.record("train/std", old_log_std.cpu().exp().mean().item())
        self.logger.record("train/ratio", ratio.cpu().mean().item()) 
        # self.logger.record("losses/lr_vf_", self.policy.optimizer_vf.param_groups[0]["lr"])

        self.logger.record("log_policies/policy_min", log_policies.sum(-1).detach().cpu().min().item())
        self.logger.record("log_policies/policy_max", log_policies.sum(-1).detach().cpu().max().item())
        self.logger.record("log_policies/policy_mean", log_policies.sum(-1).detach().cpu().mean().item())

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """

        self.policy.train()

        if self.shared:
            return self._train_shared()
        else:
            return self._train_separate()

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "CustomPPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "CustomPPO":

        return super(CustomPPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            # NOTE(junweiluo): 注释掉这个参数
            # eval_env=eval_env,
            # eval_freq=eval_freq,
            # n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            # eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )


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