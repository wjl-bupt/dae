# -*- encoding: utf-8 -*-
'''
@File :policy.py
@Created-Time :2025-11-29 10:10:04
@Author  :june
@Description   : Policy for Mujoco.
@Modified-Time : 2025-11-29 10:10:04
'''

import torch as th
import torch.nn as nn
from functools import partial
from typing import Optional, Tuple, Type, List, Union, Dict
import numpy as np 
import gymnasium as gym
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor, FlattenExtractor, NatureCNN


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        lr_schedule_vf: Optional[Schedule] = None,
        ortho_init: bool = True,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        squash_output : bool = False,
        shared_features_extractor: bool = True,
        net_arch : List[Union[int, Dict[str, List[int]]]] = [dict(pi=[64, 64], vf=[64, 64])],
        activation_fn : Optional[nn.Module] = nn.Tanh,
    ):
        self.shared_features_extractor = shared_features_extractor
        self.lr_vf = lr_schedule_vf(1) if lr_schedule_vf else None

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            activation_fn=activation_fn,
            net_arch=net_arch,
            ortho_init=ortho_init,
            use_sde=False,
            log_std_init=0.0,
            full_std=True,
            # NOTE(junweiluo): 注释掉这个参数
            # sde_net_arch=None,
            use_expln=False,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=None,
            normalize_images=True,
            optimizer_class=th.optim.Adam,
            optimizer_kwargs=None,

        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """        
        # action log_std
        self.log_std = nn.Parameter(th.ones(self.action_space.shape[0]) * self.log_std_init)

        # ------------------------
        # Shared AC
        # ------------------------
        if self.shared_features_extractor:
            # MlpExtractor 分 actor/critic latent
            self.mlp_extractor = MlpExtractor(
                self.features_extractor.features_dim,
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                device=self.device,
            )
            # heads
            self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, self.action_space.shape[0])
            self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
            self.advantage_net = nn.Linear(self.mlp_extractor.latent_dim_vf, self.action_space.shape[0])
        else:
            # self.features_extractor = self.features_extractor_class(self.observation_space).to(self.device).float()
            self.mlp_extractor = MlpExtractor(
                self.observation_space.shape[0],
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                device=self.device,
            )
            
            self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, self.action_space.shape[0])
            self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
            self.advantage_net = nn.Linear(self.mlp_extractor.latent_dim_pi, self.action_space.shape[0])



        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
                self.advantage_net: 0.01,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        # TODO(junweiluo)
        if (self.lr_vf is not None) and not self.shared_features_extractor:
            self.modules_pi = nn.ModuleList([self.mlp_extractor.policy_net, self.action_net])
            self.optimizer = self.optimizer_class(
                self.modules_pi.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
            )
            self.modules_vf = nn.ModuleList(
                [self.mlp_extractor.value_net, self.value_net, self.advantage_net,]
            )
            self.optimizer_vf = self.optimizer_class(
                self.modules_vf.parameters(), lr=self.lr_vf, **self.optimizer_kwargs
            )
        else:
            self.optimizer = self.optimizer_class(
                self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
            )

    def _get_hermite_univariate_polynomial(self, z: th.Tensor, n: int = 3, ) -> th.Tensor:
        he1 = z
        he2 = z*z - 1.0
        he3 = z*z*z - 3.0*z
        return th.cat([he1, he2, he3], dim=-1) 
        

    def _extract_latent(self, obs: th.Tensor) -> th.Tensor:
        if self.shared_features_extractor:
            feature = self.features_extractor(obs)
            latent_pi, latent_vf = self.mlp_extractor(feature)
            return latent_pi, latent_vf
        else:
            feature = self.features_extractor(obs)
            latent_pi, latent_vf = self.mlp_extractor(feature)
            return latent_pi, latent_vf
    
    def _predict(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:

        latent_pi, _ = self._extract_latent(obs)
        # output action mean
        mean_actions = self.action_net(latent_pi)
        # build Normal action distributin
        distribution = self.action_dist.proba_distribution(
                            mean_actions,
                            self.log_std,
                        )

        return distribution.get_actions(deterministic=deterministic)

    def extract_features_vf(self, obs: th.Tensor) -> th.Tensor:
        # pobs = preprocess_obs(
        #     obs, self.observation_space, normalize_images=self.normalize_images
        # )

        return (
            self.mlp_extractor(self.features_extractor(obs), type="vf")
            if self.shared_features_extractor
            else self.mlp_extractor_vf(self.features_extractor_vf(obs))
        )

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        obs = obs.float()
        latent_pi, latent_vf = self._extract_latent(obs)
        mean_actions = self.action_net(latent_pi)
        # NOTE(junweiluo) 25/11/29: use Gassuain
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        # policies = distribution.distribution.probs
        # log_policies shape is [n_envs]
        # if use distribution.distribution.log_prob(actions), shape will be [n_envs, action_dim]

        log_policies = distribution.log_prob(actions)

        values = self.value_net(latent_vf)

        return actions, mean_actions, log_policies, values

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor, mu: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        obs = obs.float()
        latent_pi, latent_vf = self._extract_latent(obs)
        # distribution = self._get_action_dist_from_latent(latent)
        mean_actions = self.action_net(latent_pi)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        # get log probs from distributions
        log_probs = distribution.log_prob(actions)
 


        # advantage net flow, shape is (B, D)
        adv_latent_feature = self.advantage_net(latent_vf)
        # advantages_raw shape is (B,)
        advantages = (adv_latent_feature * (actions - mu)).sum(-1)

        values = self.value_net(latent_vf)
        return values, advantages, log_probs, distribution.entropy()

    def evaluate_state(
        self, obs: th.Tensor, actions: th.Tensor, mu: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:

        return self.evaluate_actions(obs = obs, actions = actions, mu = mu)

    def predict_policy(
        self, obs: th.Tensor, actions: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:

        latent_pi, _ = self._extract_latent(obs)

        distribution = self._get_action_dist_from_latent(latent_pi)
        # policies = distribution.distribution.probs
        log_policies = distribution.log_prob(actions)

        return log_policies, distribution.entropy()

    def predict_value(
        self, obs: th.Tensor, actions: Optional[th.Tensor] = None, mu: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        obs = obs.float()
        # latent = self.extract_features(obs)
        _, latent_vf = self._extract_latent(obs)

        # advantage net flow, shape is (B, D)
        adv_latent_feature = self.advantage_net(latent_vf) 
        

        if actions is None:
            advantages = adv_latent_feature.sum(-1)
        else:
            advantages = (adv_latent_feature * (actions - mu)).sum(-1)
        # keep \pi-centered advantage constrant
        values = self.value_net(latent_vf)
        
        return values, advantages
