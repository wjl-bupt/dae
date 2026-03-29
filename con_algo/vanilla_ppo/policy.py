# -*- encoding: utf-8 -*-
'''
@File :policy.py
@Created-Time :2025-11-27 20:51:57
@Author  :june
@Description   : proximal policy optimization in PPO.
@Modified-Time : 2025-11-27 20:51:57
'''

import gymnasium as gym
import torch as th
import torch.nn as nn
from con_algo.util import SimBaEncoder
from typing import Optional, Tuple, Type, List, Union, Dict
from con_algo.util import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor


class SimBaFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        block_num: int = 2,
        hidden_dim: int = 256,
        activation = nn.Tanh(),
        **kwargs,
    ):
        input_dim = observation_space.shape[0]
        super().__init__(observation_space, features_dim=hidden_dim)

        self.encoder = SimBaEncoder(
            input_dim=input_dim,
            block_num=block_num,
            hidden_dim=hidden_dim,
            activation=activation,
        )

    def forward(self, observations):
        return self.encoder(observations)

# class VanillaPPOAC(ActorCriticPolicy):
#     def __init__(
#         self,
#         observation_space: gym.spaces.Space,
#         action_space: gym.spaces.Space,
#         lr_schedule: Schedule,
#         lr_schedule_vf: Optional[Schedule] = None,
#         ortho_init: bool = True,
#         features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
#         squash_output : bool = False,
#         shared_features_extractor: bool = True,
#         net_arch : List[Union[int, Dict[str, List[int]]]] = [dict(pi=[64, 64], vf=[64, 64])],
#         activation_fn : Optional[nn.Module] = nn.Tanh,
#         use_sde : bool = False,
#     ):
#         self.shared_features_extractor = shared_features_extractor
#         # if lr_schedule_vf else None
#         self.lr_vf = lr_schedule_vf(1) if lr_schedule_vf else None
        

#         super(VanillaPPOAC, self).__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             activation_fn=activation_fn,
#             net_arch=net_arch,
#             ortho_init=ortho_init,
#             use_sde=use_sde,
#             log_std_init=0.0,
#             full_std=True,
#             # NOTE(junweiluo): 注释掉这个参数
#             # sde_net_arch=None,
#             use_expln=False,
#             squash_output=squash_output,
#             features_extractor_class=features_extractor_class,
#             features_extractor_kwargs=None,
#             normalize_images=True,
#             optimizer_class=th.optim.AdamW,
#             optimizer_kwargs=None,
#         )

#         hidden_dim = 256
#         self.actor_feature_extractor = SimBaEncoder(input_dim = self.observation_space.shape[0], block_num = 2, hidden_dim = hidden_dim, activation=nn.Tanh())
#         self.action_net = nn.Linear(hidden_dim, self.action_space.shape[0])
#         self.log_std = nn.Parameter(th.zeros(self.action_space.shape[0]))
#         self.value_feature_extractor = SimBaEncoder(input_dim = self.observation_space.shape[0], block_num = 2, hidden_dim = hidden_dim, activation=nn.Tanh())
#         self.value_net = nn.Linear(hidden_dim, 1)
        
#         self.action_dist = DiagGaussianDistribution(self.action_space)
        
    
#     def get_latent(self, obs):
#         latent_pi = self.actor_feature_extractor(obs)
#         latent_vf = self.value_feature_extractor(obs)

#         return latent_pi, latent_vf

#     def forward(self, obs, deterministic = False):
        
#         latent_pi, latent_vf = self.get_latent(obs)
#         mu = self.action_net(latent_pi)
#         dist = self.action_dist.proba_distribution(mu, self.log_std)
#         actions = dist.sample()
#         log_probs = dist.log_prob(actions)
#         values = self.value_net(latent_vf)

#         return actions, values, log_probs

#     def evaluate_actions(self, obs, actions):
#         latent_pi, latent_vf = self.get_latent(obs)
#         mu = self.action_net(latent_pi)
#         dist = self.action_dist.proba_distribution(mu, self.log_std)
#         log_probs = dist.log_prob(actions)
#         entropy = dist.entropy()
#         values = self.value_net(latent_vf)

#         return values, log_probs, entropy
