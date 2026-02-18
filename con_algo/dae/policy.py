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
from con_algo.util import DiagGaussianDistribution, layer_init
from torch.distributions import Normal




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

        self.actor_activate_func = nn.SiLU()
        self.activate_func = nn.SiLU()

        hidden_dim = 64
        self.observation_feature_extractor = nn.Sequential(
            layer_init(nn.Linear(self.observation_space.shape[0], hidden_dim)),
            self.actor_activate_func,
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            self.actor_activate_func,
        )
        
        self.action_feature_extractor = nn.Sequential(
            layer_init(nn.Linear(self.action_space.shape[0], hidden_dim)),
            self.activate_func,
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            self.activate_func,
        )
        
        self.action_net = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, self.action_space.shape[0]), std=0.01),
        )
        self.log_std = nn.Parameter(th.zeros(self.action_space.shape[0]))
        self.value_net = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.advantage_net = nn.Sequential(
            layer_init(nn.Linear(hidden_dim * 2, hidden_dim), std=np.sqrt(2)),
            self.activate_func,
            layer_init(nn.Linear(hidden_dim, hidden_dim), std=np.sqrt(2)),
            self.activate_func,
            layer_init(nn.Linear(hidden_dim, 1), std=0.1)
        )

        self.critic_observation_feature_extractor = None
        # Setup optimizer with initial learning rate
        # TODO(junweiluo) (self.lr_vf is not None) and 
        if not self.shared_features_extractor:
            self.critic_observation_feature_extractor = nn.Sequential(
                layer_init(nn.Linear(self.observation_space.shape[0], hidden_dim), std = np.sqrt(2)),
                self.activate_func,
                layer_init(nn.Linear(hidden_dim, hidden_dim), std = np.sqrt(2)),
                self.activate_func,
            )
            # self.modules_pi = nn.ModuleList([self.actor_feature_extractor, self.action_net, self.log_std])
            self.modules_pi = list(self.observation_feature_extractor.parameters()) \
                + list(self.action_net.parameters()) + [self.log_std] + list(self.action_feature_extractor.parameters()) \
                + list(self.advantage_net.parameters())

            self.optimizer = self.optimizer_class(
                self.modules_pi, lr=lr_schedule(1), **self.optimizer_kwargs
            )
            self.modules_vf = nn.ModuleList(
                [self.critic_observation_feature_extractor, 
                 self.value_net,
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
        latent_pi = self.observation_feature_extractor(obs)
        if self.critic_observation_feature_extractor is None:
            return latent_pi, None
        else:
            latent_vf = self.critic_observation_feature_extractor(obs)
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

    def calc_meam_std(self, latent_obs):
        mu = self.action_net(latent_obs)
        # log_std = self.log_std(latent_obs)
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
        
        if latent_vf is None:
            values = self.value_net(latent_pi)
        else:
            values = self.value_net(latent_vf)

        return actions, mean_actions, log_policies, values, actions


    def hess_diag_single(self, latent_b, action_b):
        """
        latent_b: (latent_dim,)
        action_b: (action_dim,)
        return: (action_dim,)  Hessian diagonal
        """

        def f_action(a):
            return \
                self.advantage_net(
                    th.cat([latent_b, self.action_feature_extractor(a)], dim=-1)
                ).squeeze(-1)

        # g(a) = ∇_a f(a)
        grad_f = grad(f_action)

        # Hessian = jacobian of grad
        H = jacrev(grad_f)(action_b)   # (action_dim, action_dim)

        return th.diagonal(H)

    def jacobian_single(self, latent_b, action_b):
        return self.advantage_net(
            th.cat([latent_b, self.action_feature_extractor(action_b)], dim=-1)
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
            jacrevian_diag * (mu - zero_anchor),
            dim=-1, keepdim = True,
        )
        
        return jacrevian


    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor, 
        mu: th.Tensor, log_std: th.Tensor,
        noise: th.Tensor, epsilon:float = 1e-8,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        
        
        obs = obs.float()
        latent_adv = self.observation_feature_extractor(obs)
        with th.no_grad():
            anchor_actions = mu
        anchor_embeddings = self.action_feature_extractor(anchor_actions) 
        action_embeddings = self.action_feature_extractor(actions)
        f_a = self.advantage_net(th.concat([latent_adv, action_embeddings], dim = -1))
        f_anchor = self.advantage_net(th.concat([latent_adv, anchor_embeddings], dim = -1))
        sigma = th.exp(log_std).pow(2)
        trace = self.calc_hessian_diag(latent_adv.detach(), mu, sigma)
        # trace = self.calc_jacrevian_diag(latent_vf.detach(), anchor_actions, mu)
        ex_adv =  f_anchor + trace
        
        advantages = f_a - ex_adv
        advantages = advantages.squeeze(-1)
        
        return advantages, ex_adv.squeeze(-1), trace.squeeze(-1)
        # return values, advantages, log_probs, distribution.entropy()
    
    def evaluate_state(
        self, obs: th.Tensor, actions: th.Tensor, mu: th.Tensor, log_std: th.Tensor = None, noise: th.Tensor = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:

        obs = obs.float()
        latent_pi, latent_vf = self._extract_latent(obs)
        new_mu = self.action_net(latent_pi)
        new_log_std = self.log_std
        new_dist = self.action_dist.proba_distribution(new_mu, new_log_std)
        new_log_policies = new_dist.log_prob(actions)
        entropy = new_dist.entropy()

        with th.no_grad():
            anchor_actions = mu
        
        anchor_embeddings = self.action_feature_extractor(anchor_actions)
        action_embeddings = self.action_feature_extractor(actions)
        f_a = self.advantage_net(th.concat([latent_pi, action_embeddings], dim = -1))
        f_anchor = self.advantage_net(th.concat([latent_pi, anchor_embeddings], dim = -1))
        sigma = th.exp(log_std).pow(2)
        # with th.no_grad():
        trace = self.calc_hessian_diag(latent_pi.detach(), mu, sigma)
        # trace = self.calc_jacrevian_diag(latent_vf.detach(), anchor_actions, mu)
        ex_adv =  f_anchor + trace
        
        advantages = f_a - ex_adv
        advantages = advantages.squeeze(-1)
        
        if latent_vf is None:
            values = self.value_net(latent_pi)
        else:
            values = self.value_net(latent_vf)
        
        return values, advantages, new_log_policies, entropy, ex_adv.squeeze(-1), trace.squeeze(-1)

    def predict_policy(
        self, obs: th.Tensor, actions: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:

        latent_pi = self.observation_feature_extractor(obs)
        mean_actions = self.action_net(latent_pi)
        new_log_std = self.log_std
        dist = self.action_dist.proba_distribution(mean_actions, new_log_std)
        log_policies = dist.log_prob(actions)
        # log_policies -= th.log(1 - tanh_w_actions.pow(2) + 1e-8)
        # log_policies = log_policies.sum(-1)
        # policies = distribution.distribution.probs
        # log_policies = distribution.log_prob(actions)

        return log_policies, dist.entropy()

    def predict_advantage(
        self, obs: th.Tensor, actions: Optional[th.Tensor] = None, 
        mu: Optional[th.Tensor] = None, log_std : Optional[th.Tensor] = None,
        noise: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        obs = obs.float()
        # latent = self.extract_features(obs)
        

        if actions is None:
            advantages = None
            _, latent_vf = self._extract_latent(obs)
            values = self.value_net(latent_vf)
            return values, advantages
        else:
            advantages, ex_adv, trace = self.evaluate_actions(obs, actions, mu, log_std, noise)
            # build \pi-centered advantage constrant
            # advantages = advantages - advantages_mu
            return advantages, ex_adv, trace
    
    
    def predict_value(
        self, obs: th.Tensor
    ):
        latent_pi, latent_vf = self._extract_latent(obs)
        if latent_vf is None:
            values = self.value_net(latent_pi)
        else:
            values = self.value_net(latent_vf)
        
        # 返回none主要是对齐接口
        return values, None