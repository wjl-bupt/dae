# -*- encoding: utf-8 -*-
'''
@File :policy.py
@Created-Time :2026-01-31 14:53:12
@Author  :june
@Description   :Description of this file
@Modified-Time : 2026-01-31 14:53:12
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