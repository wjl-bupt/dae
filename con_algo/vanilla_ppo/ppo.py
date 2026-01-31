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
import torch.nn as nn
from con_algo.util import SimBaNet
from con_algo.util import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy

class VanillaPPOAC(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch = None, activation_fn = nn.Tanh, ortho_init = True, use_sde = False, log_std_init = 0, full_std = True, use_expln = False, squash_output = False, features_extractor_class = ..., features_extractor_kwargs = None, share_features_extractor = True, normalize_images = True, optimizer_class = th.optim.Adam, optimizer_kwargs = None):
        super().__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, ortho_init, use_sde, log_std_init, full_std, use_expln, squash_output, features_extractor_class, features_extractor_kwargs, share_features_extractor, normalize_images, optimizer_class, optimizer_kwargs)

        hidden_dim = 256
        self.actor_feature_extractor = SimBaNet(input_dim = self.observation_space.shape[0], block_num = 2, hidden_dim = hidden_dim)
        self.action_net = nn.Linear(hidden_dim, self.action_space.shape[0])
        self.log_std = nn.Parameter(th.zeros(self.action_space.shape[0]))
        self.value_feature_extractor = SimBaNet(input_dim = self.observation_space.shape[0], block_num = 2, hidden_dim = hidden_dim)
        self.value_net = nn.Linear(hidden_dim, 1)
        
    
    def get_latent(self, obs):
        latent_pi = self.actor_feature_extractor(obs)
        latent_vf = self.value_feature_extractor(obs)

        return latent_pi, latent_vf

    def forward(self, obs, deterministic = False):
        
        latent_pi, latent_vf = self.get_latent(obs)
        mu = self.action_net(latent_pi)
        dist = self.action_dist.distribution(mu, self.log_std)
        actions = dist.sample()
        log_probs = dist.log_probs(actions)
        values = self.value_net(latent_vf)

        return actions, values, log_probs

    def evaluate_actions(self, obs, actions):
        return super().evaluate_actions(obs, actions)