# -*- encoding: utf-8 -*-
'''
@File :policy.py
@Created-Time :2025-11-29 10:10:04
@Author  :june
@Description   : Policy for Mujoco.
@Modified-Time : 2025-11-29 10:10:04
'''

import math
import torch as th
import torch.nn as nn
from functools import partial
from typing import Optional, Tuple, Type, List, Union, Dict, Any
import numpy as np 
import gymnasium as gym
# from functorch import vmap
# from torch.autograd.functional import hessian, jacobian
from torch.func import vmap, hessian, functional_call, jacrev, grad
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from torch.optim import Adam, AdamW
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor, FlattenExtractor, NatureCNN
from copy import deepcopy
from con_algo.util import DiagGaussianDistribution, layer_init, SimBaEncoder
from torch.distributions import Normal, Categorical
from torch.func import jacrev


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
        nheads : int = 2,
        learning_rate_vf: float = 0.00015,
        optimizer_kwargs: dict[str, Any] | None = None,
    ):
        self.nheads = nheads
        self.bins = 50
        self.advantage_max = 10
        self.shared_features_extractor = shared_features_extractor
        # if lr_schedule_vf else None
        self.lr_vf = lr_schedule_vf(1) if lr_schedule_vf else None
        self.learning_rate_vf = learning_rate_vf

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
            optimizer_kwargs=optimizer_kwargs,
        )
        # self.lr_vf = lr_schedule_vf(1) 
        self.high = th.from_numpy(self.action_space.high).float()
        self.low = th.from_numpy(self.action_space.low).float()
        self.action_dist = DiagGaussianDistribution(self.action_space.shape[0])
        
        
        self.pow_vector = th.arange(1, self.nheads + 1, 1).to(self.device)
        even_mask = (self.pow_vector % 2 ==0)
        m = self.pow_vector // 2
        self.const = th.zeros_like(self.pow_vector, dtype=th.float32, device=self.device)
        self.const[even_mask] = (
            th.exp(
                th.lgamma(self.pow_vector[even_mask] + 1)
                - (m[even_mask] * th.log(th.tensor(2.0, device=self.device)))
                - th.lgamma(m[even_mask] + 1)
            )
        )
        self.supports = th.linspace(-self.advantage_max, self.advantage_max, self.bins)
        # self.pow_vector = th.arange(1, self.nheads+2, 2)
        
        self.ema_ex_adv = 0.0
        self.ema_coef = 0.9

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """        
        # self.actor_feature_extractor = nn.Sequential(
        #     nn.Linear(self.observation_space.shape[0], 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 64),
        #     nn.Tanh(),
        # )

        self.advantage_activate_func = nn.Tanh()
        self.activate_func = nn.Tanh()

        hidden_dim = 256
        self.actor_feature_extractor = nn.Sequential(
            # layer_init(nn.Linear(self.observation_space.shape[0], hidden_dim)),
            # self.actor_activate_func,
            # layer_init(nn.Linear(hidden_dim, hidden_dim)),
            # self.actor_activate_func,
            SimBaEncoder(input_dim = self.observation_space.shape[0], block_num = 2,
                         hidden_dim = hidden_dim, activation = self.activate_func)
        )
        self.action_net = nn.Linear(hidden_dim, self.action_space.shape[0])
        # self.log_std = layer_init(nn.Linear(hidden_dim, self.action_space.shape[0]), std=0.01)
        self.log_std = nn.Parameter(th.zeros(self.action_space.shape[0]) * (-0.6931))
        self.value_feature_extractor = SimBaEncoder(
            input_dim = self.observation_space.shape[0], block_num = 2,
            hidden_dim = hidden_dim, activation = self.activate_func
        )
        self.value_net = nn.Linear(hidden_dim, 1)
        self.advantage_head = nn.Sequential(
            # SimBaEncoder(input_dim = self.observation_space.shape[0] + self.action_space.shape[0], block_num = 2,
            #              hidden_dim = hidden_dim, activation = self.activate_func),
            # nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Linear(hidden_dim + self.action_space.shape[0] , hidden_dim * 2),
            self.advantage_activate_func,
            nn.Linear(hidden_dim * 2 , hidden_dim * 2),
            self.advantage_activate_func,
            nn.Linear(hidden_dim * 2, self.action_space.shape[0]),
        )
        self.weights_heads = nn.Linear(hidden_dim, self.action_space.shape[0])

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.actor_feature_extractor: np.sqrt(2),
                self.value_feature_extractor: np.sqrt(2),
                # self.advantage_feature_extractor : np.sqrt(2),
                self.action_net: 0.01,
                self.value_net : 1.0,
                self.advantage_head : 0.1,
                self.weights_heads : 0.01,
                # self.log_sigma_state: 0.01,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        # TODO(junweiluo) (self.lr_vf is not None) and 
        if not self.shared_features_extractor:
            # self.modules_pi = nn.ModuleList([self.actor_feature_extractor, self.action_net, self.log_std])
            self.modules_pi = list(self.actor_feature_extractor.parameters()) \
                + list(self.action_net.parameters()) + [self.log_std]

            self.optimizer = self.optimizer_class(
                self.modules_pi, lr=lr_schedule(1), **self.optimizer_kwargs
            )
            # self.advantage_feature_extractor,
            self.modules_vf = nn.ModuleList(
                [self.value_feature_extractor, 
                 self.value_net, self.advantage_head,
                 self.weights_heads,
            ])
            # self.lr_vf
            # we will use linear decay in ppo.py
            self.optimizer_vf = self.optimizer_class(
                self.modules_vf.parameters(), lr = self.learning_rate_vf,
            )
        else:
            self.optimizer = self.optimizer_class(
                self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
            )
        

    def _extract_latent(self, obs: th.Tensor) -> th.Tensor:
        # feature = self.features_extractor(obs)
        # latent_obs, latent_vf = self.mlp_extractor(feature)

        latent_pi = self.actor_feature_extractor(obs)
        # latent_vf = self.critic_observation_feature_extractor(obs)
        # latent_vf = self.value_feature_extractor(obs)
        return latent_pi, None
    
    def _predict(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        
        latent_pi, _ = self._extract_latent(obs)
        mean_actions, log_std = self.calc_meam_std(latent_pi)
        if deterministic:
            actions = mean_actions
        else:
            dist = self.action_dist.proba_distribution(mean_actions, log_std)
            actions = dist.sample()
        
        return actions

    def calc_meam_std(self, latent_obs):
        mu = self.action_net(latent_obs)
        # log_std = self.log_std(latent_obs)
        log_std = self.log_std
        # log_std = th.clamp(log_std, -4, 2)
        
        return mu, log_std

    def forward(
        self, obs: th.Tensor, deterministic: bool = False,
        epsilon : float = 1e-8,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # obs = obs.float()
        latent_pi, _ = self._extract_latent(obs)
        mean_actions, log_std = self.calc_meam_std(latent_pi)

        dist = self.action_dist.proba_distribution(mean_actions, log_std)
        actions = dist.sample()
        # actions = th.clamp(dist.sample(), th.tensor(self.action_space.low, device=obs.device), th.tensor(self.action_space.high,  device=obs.device))          
        # rsample = reparameterization
        # actions_w_tanh = nn.functional.tanh(actions)

        # log-prob（包含 tanh Jacobian 修正）
        log_policies = dist.log_prob(actions, need_sum = False)
        # log_policies -= th.log(1 - actions_w_tanh.pow(2) + 1e-8)
        # log_policies = log_policies.sum(-1)
            
        values = self.value_net(self.value_feature_extractor(obs))

        return actions, mean_actions, log_policies, values, actions


    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor, 
        mu: th.Tensor, log_std: th.Tensor,
        noise: th.Tensor, epsilon:float = 1e-8,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        latent_pi, _ = self._extract_latent(obs)
        mean_actions = self.action_net(latent_pi)
        new_log_std = self.log_std
        
        dist = self.action_dist.proba_distribution(mean_actions, new_log_std)
        log_policies = dist.log_prob(actions, need_sum = False)
        entropy = dist.entropy()
        # shape is [B, B, D]
        latent_vf = self.value_feature_extractor(obs)
        # latent_adv = self.advantage_feature_extractor(obs)
        # shape is [Batch, act_dim]
        fs = self.advantage_head(th.cat([latent_vf, actions], dim = 1))
        # ws = self.advantage_net(th.cat([latent_vf, actions], dim = 1))
        with th.no_grad():
            sigma = th.exp(log_std)
            scores =  - (actions - mu) / (sigma + 1e-12)
            # scores = scores.mean(dim = 1, keepdim = True)

        def f_single(x, w):
            inp = th.cat([w, x], dim=-1)
            return self.advantage_head(inp)

        # # with th.no_grad():
        J = vmap(jacrev(f_single))(actions, latent_vf.detach())  # [B,K,K]
        # divs = J.squeeze(1)
        divs = J.diagonal(dim1=1,dim2=2)
        
        # 1. 直接做均值或者sum
        # advantages = ((fs * scores + divs - (1 - sigma) * divs.mean(dim = 0, keepdim = True))).mean(1) 
        
        # 2. 使用平方和然后开方的方式，减少极端值影响
        # advantage_components = ((fs * scores + divs - (1 - sigma) * divs.mean(dim = 0, keepdim = True)))
        # advantages = th.sign(advantage_components.mean(1)) * th.sqrt(th.clamp(advantage_components.pow(2).mean(1), min=1e-10))
        
        # 3. 在多输出一个weights头，用于权衡不同的weights
        advantage_components = ((fs * scores + divs - (1 - sigma) * divs.mean(dim = 0, keepdim = True)))
        weights = self.weights_heads(latent_vf)
        advantages = (advantage_components * th.nn.functional.softmax(weights, dim = 1)).sum(1)
        
        values = self.value_net(latent_vf)
        
        return values, advantages, log_policies, entropy, scores, divs, fs
        # return values, advantages, log_probs, distribution.entropy()
    
    def evaluate_state(
        self, obs: th.Tensor, actions: th.Tensor, mu: th.Tensor, log_std: th.Tensor = None, noise: th.Tensor = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:

        return self.evaluate_actions(obs = obs, actions = actions, mu = mu, log_std = log_std, noise = noise)

    def predict_policy(
        self, obs: th.Tensor, actions: Optional[th.Tensor] = None, tanh_w_actions : Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:

        latent_pi, _ = self._extract_latent(obs)

        mean_actions = self.action_net(latent_pi)
        new_log_std = self.log_std
        # new_log_std = th.clamp(new_log_std, -4, 2)
        
        dist = self.action_dist.proba_distribution(mean_actions, new_log_std)
        log_policies = dist.log_prob(actions, need_sum = False)
        # log_policies -= th.log(1 - tanh_w_actions.pow(2) + 1e-8)
        # log_policies = log_policies.sum(-1)
        # policies = distribution.distribution.probs
        # log_policies = distribution.log_prob(actions)

        return log_policies, dist.entropy()

    def predict_value(
        self, obs: th.Tensor, actions: Optional[th.Tensor] = None, 
        mu: Optional[th.Tensor] = None, log_std : Optional[th.Tensor] = None,
        noise: Optional[th.Tensor] = None, return_all : bool = False
    ) -> Tuple[th.Tensor, th.Tensor]:
        obs = obs.float()
        # latent = self.extract_features(obs)
        

        if actions is None:
            advantages = None
            # latent_vf, _ = self._extract_latent(obs)
            values = self.value_net(self.value_feature_extractor(obs))
            return values, advantages
        else:
            values, advantages, log_probs, entropy, scores, divs, fs = self.evaluate_actions(obs, actions, mu, log_std, noise)
            # build \pi-centered advantage constrant
            # advantages = advantages - advantages_mu
            if return_all:
                return values, advantages, scores, divs, fs

            return values, advantages