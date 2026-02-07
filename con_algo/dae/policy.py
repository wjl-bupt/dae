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
            optimizer_class=th.optim.Adam,
            optimizer_kwargs=None,
        )
        # self.lr_vf = lr_schedule_vf(1) 
        self.high = th.from_numpy(self.action_space.high).float()
        self.low = th.from_numpy(self.action_space.low).float()
        self.action_dist = DiagGaussianDistribution(self.action_space.shape[0])
        
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

        self.actor_activate_func = nn.Tanh()
        self.critic_activate_func = nn.ReLU()

        hidden_dim = 256
        self.actor_feature_extractor = SimBaEncoder(input_dim = self.observation_space.shape[0], block_num = 2, hidden_dim = hidden_dim, activation = self.actor_activate_func)
        self.action_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            self.actor_activate_func,
            nn.Linear(64, 64),
            self.actor_activate_func,
        )
        self.action_head = nn.Linear(64, self.action_space.shape[0])

        # self.action_net = nn.Linear(64, self.action_space.shape[0])
        self.log_std = nn.Parameter(th.zeros(self.action_space.shape[0]))
        # self.log_std = nn.Linear(64, self.action_space.shape[0])
        # self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.value_feature_extractor = SimBaEncoder(input_dim = self.observation_space.shape[0], block_num = 2, hidden_dim = hidden_dim, activation = self.critic_activate_func)
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.critic_activate_func,
            nn.Linear(hidden_dim, hidden_dim),
            self.critic_activate_func,
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        # self.advantage_feature_extractor = SimBaEncoder(
        #     input_dim = self.observation_space.shape[0], 
        #     block_num = 2, 
        #     hidden_dim = hidden_dim, 
        #     activation=self.critic_activate_func
        # )
        self.advantage_net = nn.Sequential(
            # SimBaEncoder(input_dim = self.observation_space.shape[0], block_num = 2, hidden_dim = hidden_dim, activation = self.critic_activate_func),
            # self.critic_activate_func,
            nn.Linear(hidden_dim, hidden_dim),
            self.critic_activate_func,
            nn.Linear(hidden_dim, hidden_dim),
            self.critic_activate_func,
            
        )
        self.n_tril = self.action_space.shape[0] * (self.action_space.shape[0] + 1) // 2
        self.drank = 1
        self.advantage_head = nn.Linear(hidden_dim, self.n_tril)
        # self.rank_head = nn.Linear(hidden_dim, self.drank * self.action_space.shape[0])

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

                self.actor_feature_extractor : np.sqrt(2),
                self.value_feature_extractor : np.sqrt(2),
                self.advantage_net : np.sqrt(2),
                self.action_net: np.sqrt(2),
                # self.advantage_feature_extractor: np.sqrt(2),
                self.action_head: 0.01,
                self.value_head: 1.0,
                # self.adv_d : 0.1,
                # self.adv_g : 0.1,
                self.advantage_head: 0.1,
                # self.rank_head: 0.1,
                # 
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        # TODO(junweiluo) (self.lr_vf is not None) and 
        if not self.shared_features_extractor:
            # self.modules_pi = nn.ModuleList([self.actor_feature_extractor, self.action_net, self.log_std])
            self.modules_pi = list(self.actor_feature_extractor.parameters()) + list(self.action_net.parameters()) +  list(self.action_head.parameters())  + [self.log_std]

            self.optimizer = self.optimizer_class(
                self.modules_pi, lr=lr_schedule(1), **self.optimizer_kwargs
            )
            self.modules_vf = nn.ModuleList(
                [self.value_feature_extractor, 
                 self.value_net, self.advantage_net, 
                 self.value_head, self.advantage_head,
                 # self.rank_head,
            ])
            # self.lr_vf
            self.optimizer_vf = Adam(
                self.modules_vf.parameters(), lr = 2.5e-4,
            )
        else:
            self.optimizer = self.optimizer_class(
                self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
            )
        

    def _extract_latent(self, obs: th.Tensor) -> th.Tensor:
        # feature = self.features_extractor(obs)
        # latent_pi, latent_vf = self.mlp_extractor(feature)

        latent_pi = self.action_net(self.actor_feature_extractor(obs))
        latent_vf = self.value_net(self.value_feature_extractor(obs))
        return latent_pi, latent_vf
    
    def _predict(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        

        obs = obs.float()
        latent_pi, _ = self._extract_latent(obs)
        mean_actions, log_std = self.calc_meam_std(latent_pi)
        if deterministic:
            actions = mean_actions
        else:
            dist = self.action_dist.proba_distribution(mean_actions, log_std)
            actions = dist.sample()          
        
        return actions

    def calc_meam_std(self, latent_pi):
        mu = self.action_head(latent_pi)
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
        actions = dist.sample()          
        # rsample = reparameterization
        # actions_w_tanh = nn.functional.tanh(actions)

        # log-prob（包含 tanh Jacobian 修正）
        log_policies = dist.log_prob(actions)
        # log_policies -= th.log(1 - actions_w_tanh.pow(2) + 1e-8)
        # log_policies = log_policies.sum(-1)
            
        values = self.value_head(latent_vf)

        return actions, mean_actions, log_policies, values, actions


    def hess_diag_single(self, latent_b, action_b):
        """
        latent_b: (latent_dim,)
        action_b: (action_dim,)
        return: (action_dim,)  Hessian diagonal
        """

        def f_action(a):
            return self.advantage_head(
                self.advantage_net(self.advantage_feature_extractor(th.cat([latent_b, a], dim=-1)))
            ).squeeze(-1)

        # g(a) = ∇_a f(a)
        grad_f = grad(f_action)

        # Hessian = jacobian of grad
        H = jacrev(grad_f)(action_b)   # (action_dim, action_dim)

        return th.diagonal(H)

    def jacobian_single(self, latent_b, action_b):
        return self.advantage_head(
            self.advantage_net(self.advantage_feature_extractor(th.cat([latent_b, action_b], dim=-1)))
        ).squeeze(-1)


    def calc_hessian_diag(self, latent_vf: th.Tensor, actions: th.Tensor, sigma : th.Tensor) -> th.Tensor:
        hessian_diag = vmap(self.hess_diag_single)(
            latent_vf, 
            actions,
        )
        hessian = 0.5 * th.sum(
            hessian_diag * sigma,
            dim=-1, keepdim = True,
        )
        
        return hessian
    
    def calc_jacrevian_diag(self, latent_vf, zero_anchor: th.Tensor, mu: th.Tensor) -> th.Tensor:
        jacrevian_diag = vmap(self.jacobian_single)(
            latent_vf, 
            zero_anchor,
        ).unsqueeze(-1)
        jacrevian = 0.5 * th.mean(
            jacrevian_diag * mu,
            dim=-1, keepdim = True,
        )
        
        return jacrevian


    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor, 
        mu: th.Tensor, log_std: th.Tensor,
        noise: th.Tensor, epsilon:float = 1e-8,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        
        
        obs = obs.float()
        latent_pi, latent_vf = self._extract_latent(obs)
        mean_actions = self.action_head(latent_pi)
        new_log_std = self.log_std
        
        
        dist = self.action_dist.proba_distribution(mean_actions, new_log_std)
        log_policies = dist.log_prob(actions)
        # log_policies -= th.log(1 - actions.pow(2) + 1e-8)
        # log_policies = log_policies.sum(-1)
        entropy = dist.entropy()
        
        # .view(-1, self.action_space.shape[0], self.action_space.shape[0])
        tril = self.advantage_head(self.advantage_net(latent_vf))
        
        # full-rank
        hs = th.zeros(
            (actions.shape[0], self.action_space.shape[0], self.action_space.shape[0]),
            device=actions.device,
        )
        indices = th.tril_indices(row=self.action_space.shape[0], col=self.action_space.shape[0])
        hs[:, indices[0], indices[1]] = tril
        hs_diag = th.diagonal(hs, dim1=1, dim2=2)
        # hs = hs + hs.transpose(1, 2) - th.diag_embed(th.diagonal(hs, dim1=1, dim2=2))
        hs = hs - th.diag_embed(hs_diag) + th.diag_embed(nn.functional.softplus(hs_diag))
        hs = hs @ hs.transpose(-1, -2)
        
        # low-rank
        # diag = self.advantage_head(self.advantage_net(latent_vf))
        # diag = nn.functional.softplus(diag)
        # D = th.diag_embed(diag)  # (B, action_dim, action_dim)
        # hs = D
        # ranks = self.rank_head(self.advantage_net(latent_vf)).view(
        #     -1, self.action_space.shape[0], self.drank)
        # ranks2 =  ranks @ ranks.transpose(-1, -2)  # (B, action_dim, action_dim)
        
        # hs = D + ranks2  # (B, action_dim, action_dim)
        
        
        
        # 只保证对称（不保证正定）
        # hs = 0.5 * (hs + hs.transpose(1, 2))

        # 保证正定 + 对称
        # hs = th.tril(hs.view(-1, self.action_space.shape[0], self.action_space.shape[0]))
        # diag = th.diagonal(hs, dim1=1, dim2=2)
        # hs = hs - th.diag_embed(diag) + th.diag_embed(nn.functional.softplus(diag))
        # # tanh_mu = nn.functional.tanh(mu / th.sqrt(1 + (th.pi * log_std.exp().pow(2) / 8)))

        var = th.exp(2 * log_std)
        delta = (actions - mu).unsqueeze(-1)  # (B, action_dim, 1)
        
        # diag implement
        # delta = delta.squeeze(-1)
        # raw_advantages =  -0.5 * (hs * delta.pow(2)).sum(dim=1)
        # ex_adv = -0.5 * (hs * var).sum(dim=1)


        raw_advantages = - 0.5 * th.matmul(
            delta.transpose(1, 2),
            th.matmul(hs @ hs.transpose(1, 2), delta)
        ).squeeze(-1).squeeze(-1)
        # # E[A(s,a)]
        # # var = (1 - tanh_mu.pow(2)) / (1 + (2 / th.sqrt(1 + (th.pi * var / 4))))
        # ex_adv = - 0.5 * th.sum(
        #     th.diagonal(hs, dim1=1, dim2=2) * var,
        #     dim=1
        # )
        ex_adv = -0.5 * th.sum(
            var.view(1, self.action_space.shape[0], 1)* hs.pow(2),
            dim=(1, 2)
        )
        # sigma = th.diag_embed(var)
        # M = hs @ hs.transpose(1, 2)  
        # ex_adv = -0.5 * th.einsum(
            #  "bij,bij->b", M, Sigma
        # )
        varA = 0.5 * th.sum(
            hs.pow(2) * (var.view(1, self.action_space.shape[0], 1) * var.view(1, 1, self.action_space.shape[0])),
            dim=(1, 2)
        )
        stdA = th.sqrt(varA + epsilon).detach()
        
        
        
        
        advantages = raw_advantages - ex_adv

        values = self.value_head(latent_vf)
        
        return values, advantages, log_policies, entropy, ex_adv, stdA
        # return values, advantages, log_probs, distribution.entropy()
    
    def evaluate_state(
        self, obs: th.Tensor, actions: th.Tensor, mu: th.Tensor, log_std: th.Tensor = None, noise: th.Tensor = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:

        return self.evaluate_actions(obs = obs, actions = actions, mu = mu, log_std = log_std, noise = noise)

    def predict_policy(
        self, obs: th.Tensor, actions: Optional[th.Tensor] = None, tanh_w_actions : Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:

        latent_pi, _ = self._extract_latent(obs)

        mean_actions = self.action_head(latent_pi)
        new_log_std = self.log_std
        
        
        dist = self.action_dist.proba_distribution(mean_actions, new_log_std)
        log_policies = dist.log_prob(actions)
        # log_policies -= th.log(1 - tanh_w_actions.pow(2) + 1e-8)
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
            values = self.value_head(latent_vf)
            return values, advantages, 0
        else:
            values, advantages, log_probs, entropy, ex_adv, stdA = self.evaluate_actions(obs, actions, mu, log_std, noise)
            # build \pi-centered advantage constrant
            # advantages = advantages - advantages_mu
            return values, advantages, ex_adv, stdA