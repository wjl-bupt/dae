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
from typing import Optional, Tuple, Type, List, Union, Dict
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
from con_algo.util import DiagGaussianDistribution
from torch.distributions import Normal
from con_algo.util import SimBaEncoder



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
        # if lr_schedule_vf else None
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
            optimizer_class=th.optim.AdamW,
            optimizer_kwargs=None,
        )
        # self.lr_vf = lr_schedule_vf(1) 
        self.high = th.from_numpy(self.action_space.high).float()
        self.low = th.from_numpy(self.action_space.low).float()
        self.action_dist = DiagGaussianDistribution(self.action_space.shape[0])

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

        hidden_dim = 64
        self.actor_feature_extractor = SimBaEncoder(input_dim = self.observation_space.shape[0], block_num = 2, hidden_dim = hidden_dim)
        self.action_net = nn.Linear(hidden_dim, self.action_space.shape[0])

        # self.action_net = nn.Linear(64, self.action_space.shape[0])
        self.log_std = nn.Parameter(th.zeros(self.action_space.shape[0]))
        # self.log_std = nn.Linear(64, self.action_space.shape[0])
        # self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        # self.value_feature_extractor = nn.Sequential(
        #     nn.Linear(self.observation_space.shape[0], 64),
        #     nn.LayerNorm(64),
        #     nn.GELU(approximate="tanh"),
        #     nn.Linear(64, 64),
        #     nn.LayerNorm(64),
        #     nn.GELU(approximate="tanh"),
        # )
        # self.value_net = nn.Linear(64, 1)
        # self.advantage_feature_extractor = nn.Sequential(
        #     nn.Linear(64 + self.action_space.shape[0], 64),
        #     nn.LayerNorm(64),
        #     nn.GELU(approximate="tanh"),
        #     nn.Linear(64, 64),
        #     nn.LayerNorm(64),
        #     nn.GELU(approximate="tanh"),
        # )
        self.value_feature_extractor = SimBaEncoder(input_dim = self.observation_space.shape[0], block_num = 2, hidden_dim = hidden_dim)
        self.value_net = nn.Linear(hidden_dim, 1)
        self.advantage_feature_extractor = SimBaEncoder(input_dim = hidden_dim + self.action_space.shape[0], block_num = 2, hidden_dim = hidden_dim)
        self.advantage_net = nn.Linear(hidden_dim, 1)

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                # self.features_extractor: np.sqrt(2),
                # self.mlp_extractor: np.sqrt(2),

                # self.actor_feature_extractor : np.sqrt(2),
                # self.value_feature_extractor : np.sqrt(2),
                # self.advantage_feature_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1.0,
                # self.adv_d : 0.1,
                # self.adv_g : 0.1,
                self.advantage_net: 0.1,
                # 
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        # TODO(junweiluo) (self.lr_vf is not None) and 
        if not self.shared_features_extractor:
            # self.modules_pi = nn.ModuleList([self.actor_feature_extractor, self.action_net, self.log_std])
            self.modules_pi = list(self.actor_feature_extractor.parameters()) + list(self.action_net.parameters()) + [self.log_std]

            self.optimizer = self.optimizer_class(
                self.modules_pi, lr=lr_schedule(1), **self.optimizer_kwargs
            )
            self.modules_vf = nn.ModuleList(
                [self.value_feature_extractor, self.advantage_feature_extractor, self.value_net, self.advantage_net,]
            )
            # self.lr_vf
            self.optimizer_vf = AdamW(
                self.modules_vf.parameters(), lr = 2.5e-4,
            )
        else:
            self.optimizer = self.optimizer_class(
                self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
            )
        

    def _extract_latent(self, obs: th.Tensor) -> th.Tensor:
        # feature = self.features_extractor(obs)
        # latent_pi, latent_vf = self.mlp_extractor(feature)

        latent_pi = self.actor_feature_extractor(obs)
        latent_vf = self.value_feature_extractor(obs)
        return latent_pi, latent_vf
    
    def _predict(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:

        latent_pi, _ = self._extract_latent(obs)
        # output action mean
        mean_actions = self.action_net(latent_pi)
        log_std = self.log_std(latent_pi)
        # build Normal action distributin
        normal = th.distributions.Normal(mean_actions, log_std.exp())
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = th.tanh(x_t)
        actions = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= th.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = th.tanh(mean) * self.action_scale + self.action_bias
        
        return actions

    def calc_meam_std(self, latent_pi):
        mu = self.action_net(latent_pi)
        # log_std = self.log_std(latent_pi)
        log_std = self.log_std
        
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
        obs = obs.float()
        latent_pi, latent_vf = self._extract_latent(obs)
        mean_actions, log_std = self.calc_meam_std(latent_pi)

        dist = self.action_dist.proba_distribution(mean_actions, log_std)
        actions = dist.sample()          # rsample = reparameterization
        # actions_w_tanh = nn.functional.tanh(actions)

        # log-prob（包含 tanh Jacobian 修正）
        log_policies = dist.log_prob(actions)
        # log_policies -= th.log(1 - actions_w_tanh.pow(2) + 1e-8)
        # log_policies = log_policies.sum(-1)
            
        values = self.value_net(latent_vf)

        return actions, mean_actions, log_policies, values, actions


    def hess_diag_single(self, latent_b, action_b):
        """
        latent_b: (latent_dim,)
        action_b: (action_dim,)
        return: (action_dim,)  Hessian diagonal
        """

        def f_action(a):
            return self.advantage_net(
                self.advantage_feature_extractor(th.cat([latent_b, a], dim=-1))
            ).squeeze(-1)

        # g(a) = ∇_a f(a)
        grad_f = grad(f_action)

        # Hessian = jacobian of grad
        H = jacrev(grad_f)(action_b)   # (action_dim, action_dim)

        return th.diagonal(H)



    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor, 
        mu: th.Tensor, log_std: th.Tensor,
        noise: th.Tensor, epsilon:float = 1e-8,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        
        
        obs = obs.float()
        latent_pi, latent_vf = self._extract_latent(obs)
        mean_actions = self.action_net(latent_pi)
        new_log_std = self.log_std
        
        
        dist = self.action_dist.proba_distribution(mean_actions, new_log_std)
        log_policies = dist.log_prob(actions)
        # log_policies -= th.log(1 - actions.pow(2) + 1e-8)
        # log_policies = log_policies.sum(-1)
        entropy = dist.entropy()
        
        # latent_adv = self.advantage_feature_extractor(th.concat([latent_adv, mu], dim = -1))
        # clamp_actions = th.clamp(actions, self.action_space.low.mean().item(), self.action_space.high.mean().item())
        f_a = self.advantage_net(self.advantage_feature_extractor(th.concat([latent_vf, actions], dim = -1)))
        f_zero = self.advantage_net(self.advantage_feature_extractor(th.concat([latent_vf, mu], dim = -1)))
        
        # calc hessian matrix
        # def f_single(latent_b, action_b):
        #     return self.advantage_net(th.cat([latent_b, action_b], dim=-1)).squeeze(-1)

        # def hess_single(latent_b, action_b):
        #     # 只对 action_b 求 Hessian
        #     return hessian(lambda x: f_single(latent_b, x))(action_b)

        # # def adv_single(latent_b, action_b):
        # #     x = th.cat([latent_b, action_b], dim=-1)
        # #     return self.advantage_net(x).squeeze(-1)

        # # grad_adv_single = jacrev(adv_single, argnums=1)
        # # jacobian = vmap(grad_adv_single)(
        # #     latent_adv,
        # #     mu,
        # # )

        hessian_matrix = vmap(self.hess_diag_single)(
            latent_vf, 
            mu,
        )
        hessian_diag = hessian_matrix
        # hessian_diag = th.diagonal(hessian_matrix, dim1=-2, dim2=-1)
        sigma = th.exp(log_std).pow(2)
        trace = 0.5 * th.sum(
            hessian_diag * sigma,
            dim=-1,
            keepdim=True,
        )  
        # sigma = th.diag_embed(th.exp(2 * log_std)).unsqueeze(0).repeat(mu.size(0), 1, 1)
        # hs = (hessian_matrix @ sigma)
        # trace = 0.5 * hs.diagonal(dim1=-2, dim2=-1).mean(-1).unsqueeze(-1)
        # traces = 0.5 * th.sum(hessian_diag * sigma, dim=-1).unsqueeze(-1)
        # taylor_terms = th.sum(jacobian * mu, dim = -1) + 0.5 * th.sum(hessian_diag * sigma, dim=-1)
        # taylor_terms = 0.5 * th.sum(hessian_diag * sigma, dim=-1, keepdim=True)
        # # taylor_terms = th.mean(jacobian * mu, dim = -1, keepdim = True)

        ex_adv =  f_zero + trace
        advantages = f_a - ex_adv
        advantages = advantages.squeeze(-1)

        values = self.value_net(latent_vf)
        
        return values, advantages, log_policies, entropy, ex_adv.squeeze(-1), trace.mean().detach()
        # return values, advantages, log_probs, distribution.entropy()
    
    def evaluate_state(
        self, obs: th.Tensor, actions: th.Tensor, mu: th.Tensor, log_std: th.Tensor = None, noise: th.Tensor = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:

        return self.evaluate_actions(obs = obs, actions = actions, mu = mu, log_std = log_std, noise = noise)

    def predict_policy(
        self, obs: th.Tensor, actions: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:

        latent_pi, _ = self._extract_latent(obs)

        mean_actions = self.action_net(latent_pi)
        new_log_std = self.log_std
        
        
        dist = self.action_dist.proba_distribution(mean_actions, new_log_std)
        log_policies = dist.log_prob(actions)
        # log_policies -= th.log(1 - actions.pow(2) + 1e-8)
        # log_policies = log_policies.sum(-1)
        # policies = distribution.distribution.probs
        # log_policies = distribution.log_prob(actions)

        return log_policies, dist.entropy()

    def predict_value(
        self, obs: th.Tensor, actions: Optional[th.Tensor] = None, 
        mu: Optional[th.Tensor] = None, log_std : Optional[th.Tensor] = None,
        noise: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        obs = obs.float()
        # latent = self.extract_features(obs)
        _, latent_vf = self._extract_latent(obs)

        if actions is None:
            advantages = None
            values = self.value_net(latent_vf)
            return values, advantages
        else:
            values, advantages, log_probs, entropy, ex_adv, trace = self.evaluate_actions(obs, actions, mu, log_std, noise)
            # build \pi-centered advantage constrant
            # advantages = advantages - advantages_mu
            return values, advantages, ex_adv, trace