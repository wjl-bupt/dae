# -*- encoding: utf-8 -*-
'''
Policy with analytic pi-centered RBF advantage.
'''

import torch as th
import torch.nn as nn
import numpy as np
import gymnasium as gym

from typing import Optional, Tuple, Type, List, Union, Dict
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from con_algo.util import DiagGaussianDistribution, layer_init


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        lr_schedule_vf: Optional[Schedule] = None,
        ortho_init: bool = True,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        squash_output: bool = False,
        shared_features_extractor: bool = True,
        net_arch: List[Union[int, Dict[str, List[int]]]] = [dict(pi=[64, 64], vf=[64, 64])],
        activation_fn: Optional[nn.Module] = nn.Tanh,
    ):
        self.shared_features_extractor = shared_features_extractor
        self.lr_vf = lr_schedule_vf(1) if lr_schedule_vf else None

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            activation_fn=activation_fn,
            net_arch=net_arch,
            ortho_init=ortho_init,
            use_sde=False,
            log_std_init=0.0,
            full_std=True,
            use_expln=False,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=None,
            normalize_images=True,
            optimizer_class=th.optim.AdamW,
            optimizer_kwargs=None,
        )

        self.action_dist = DiagGaussianDistribution(self.action_space.shape[0])

    # ============================================================
    # build
    # ============================================================

    def _build(self, lr_schedule: Schedule) -> None:

        hidden_dim = 256
        self.actor_act = nn.Tanh()
        self.critic_act = nn.Tanh()

        # actor feature
        self.observation_feature_extractor = nn.Sequential(
            layer_init(nn.Linear(self.observation_space.shape[0], hidden_dim)),
            self.actor_act,
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            self.actor_act,
        )

        # critic feature
        self.critic_observation_feature_extractor = nn.Sequential(
            layer_init(nn.Linear(self.observation_space.shape[0], hidden_dim), std=np.sqrt(2)),
            self.critic_act,
            layer_init(nn.Linear(hidden_dim, hidden_dim), std=np.sqrt(2)),
            self.critic_act,
        )

        # actor head
        self.action_net = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, self.action_space.shape[0]), std=0.01),
        )
        self.log_std = nn.Parameter(th.zeros(self.action_space.shape[0]))

        # value
        self.value_net = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, 1), std=0.1),
        )

        # w(s)
        self.advantage_net = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            self.critic_act,
            layer_init(nn.Linear(hidden_dim, 1), std=0.1),
        )

        # ===== RBF parameters =====
        action_dim = self.action_space.shape[0]

        # center
        self.rbf_c = nn.Parameter(th.zeros(action_dim))

        # precision (diag)
        self.rbf_log_lambda = nn.Parameter(th.zeros(action_dim))
        self.rbf_log_lambda.data.fill_(-2.0)

        # optimizer
        if not self.shared_features_extractor:
            # self.modules_pi = nn.ModuleList([self.actor_feature_extractor, self.action_net, self.log_std])
            self.modules_pi = list(self.observation_feature_extractor.parameters()) \
                + list(self.action_net.parameters()) + [self.log_std]

            self.optimizer = self.optimizer_class(
                self.modules_pi, lr=lr_schedule(1), **self.optimizer_kwargs
            )
            self.modules_vf = list(self.critic_observation_feature_extractor.parameters()) + list(self.advantage_net.parameters()) + \
                list(self.value_net.parameters()) + [self.rbf_log_lambda] + [self.rbf_c]
            # self.lr_vf
            self.optimizer_vf = self.optimizer_class(
                self.modules_vf, lr = 1.5e-4,
            )
        else:
            self.optimizer = self.optimizer_class(
                self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
            )

    # ============================================================
    # utilities
    # ============================================================

    def _extract_latent(self, obs: th.Tensor):
        latent_pi = self.observation_feature_extractor(obs)
        latent_vf = self.critic_observation_feature_extractor(obs)
        return latent_pi, latent_vf

    def calc_meam_std(self, latent_obs):
        mu = self.action_net(latent_obs)
        log_std = self.log_std
        return mu, log_std

    # ============================================================
    # RBF
    # ============================================================

    def rbf_phi(self, a):
        lam = th.exp(self.rbf_log_lambda)
        diff = a - self.rbf_c
        return th.exp(-0.5 * th.sum(lam * diff * diff, dim=-1, keepdim=True))

    def rbf_expectation(self, mu, log_std):
        sigma2 = th.exp(log_std).pow(2)
        lam = th.exp(self.rbf_log_lambda)
        diff2 = (mu - self.rbf_c) ** 2

        logZ = -0.5 * th.sum(
            th.log(1 + sigma2 * lam)
            + lam * diff2 / (1 + sigma2 * lam),
            dim=-1,
            keepdim=True,
        )
        return th.exp(logZ)

    # ============================================================
    # forward
    # ============================================================

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
    ):
        obs = obs.float()
        latent_pi, latent_vf = self._extract_latent(obs)

        mean_actions, log_std = self.calc_meam_std(latent_pi)

        dist = self.action_dist.proba_distribution(mean_actions, log_std)
        actions = mean_actions if deterministic else dist.sample()

        log_policies = dist.log_prob(actions)
        values = self.value_net(latent_vf)

        return actions, mean_actions, log_policies, values, actions

    # ============================================================
    # evaluate
    # ============================================================

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        mu: th.Tensor,
        log_std: th.Tensor,
        noise: th.Tensor,
    ):

        obs = obs.float()
        latent_pi, latent_vf = self._extract_latent(obs)

        mean_actions = self.action_net(latent_pi)
        new_log_std = self.log_std

        dist = self.action_dist.proba_distribution(mean_actions, new_log_std)
        log_policies = dist.log_prob(actions)
        entropy = dist.entropy()

        # ===== analytic centered advantage =====
        w = self.advantage_net(latent_vf)  # (B,1)

        phi_a = self.rbf_phi(actions)
        Z = self.rbf_expectation(mu, log_std)

        advantages = (w * (phi_a - Z)).squeeze(-1)

        values = self.value_net(latent_vf)

        return values, advantages, log_policies, entropy, Z.squeeze(-1), w.squeeze(-1)

    def evaluate_state(self, obs, actions, mu, log_std=None, noise=None):
        return self.evaluate_actions(obs, actions, mu, log_std, noise)

    # ============================================================
    # predict helpers
    # ============================================================

    def predict_policy(self, obs: th.Tensor, actions=None, tanh_w_actions=None):
        latent_pi, _ = self._extract_latent(obs)
        mean_actions = self.action_net(latent_pi)
        new_log_std = self.log_std

        dist = self.action_dist.proba_distribution(mean_actions, new_log_std)
        log_policies = dist.log_prob(actions)
        return log_policies, dist.entropy()

    def predict_value(self, obs: th.Tensor, actions=None, mu=None, log_std=None, noise=None):
        obs = obs.float()

        if actions is None:
            _, latent_vf = self._extract_latent(obs)
            values = self.value_net(latent_vf)
            return values, None
        else:
            values, advantages, log_probs, entropy, ex_adv, w = \
                self.evaluate_actions(obs, actions, mu, log_std, noise)
            return values, advantages, ex_adv, w
