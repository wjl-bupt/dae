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
# from functorch import vmap
# from torch.autograd.functional import hessian, jacobian
from torch.func import vmap, hessian, functional_call
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor, FlattenExtractor, NatureCNN
from copy import deepcopy
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

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """        

        if self.shared_features_extractor:
            # MlpExtractor 分 actor/critic latent
            self.mlp_extractor = MlpExtractor(
                self.features_extractor.features_dim,
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                device=self.device,
            )
        else:
            self.features_extractor = self.features_extractor_class(self.observation_space).to(self.device).float()
            self.mlp_extractor = MlpExtractor(
                self.observation_space.shape[0],
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                device=self.device,
            )

        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, self.action_space.shape[0])
        # self.log_std = nn.Parameter(th.zeros(self.action_space.shape[0]))
        self.log_std = nn.Linear(self.mlp_extractor.latent_dim_pi, self.action_space.shape[0])
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.advantage_net = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_vf + self.action_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.register_buffer(
            "action_scale",
            th.tensor(
                (self.action_space.high - self.action_space.low) / 2.0,
                dtype=th.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            th.tensor(
                (self.action_space.high + self.action_space.low) / 2.0,
                dtype=th.float32,
            ),
        )

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
                self.advantage_net: 0.1,
                # 
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        # TODO(junweiluo) (self.lr_vf is not None) and 
        if not self.shared_features_extractor:
            self.modules_pi = nn.ModuleList([self.mlp_extractor.policy_net, self.action_net, ])
            self.optimizer = self.optimizer_class(
                self.modules_pi.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
            )
            self.modules_vf = nn.ModuleList(
                [self.mlp_extractor.value_net, self.value_net, self.advantage_net,]
            )
            
            # self.lr_vf
            self.optimizer_vf = self.optimizer_class(
                self.modules_vf.parameters(), lr=self.lr_vf, **self.optimizer_kwargs
            )
        else:
            self.optimizer = self.optimizer_class(
                self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
            )
        

    def _extract_latent(self, obs: th.Tensor) -> th.Tensor:
        # if self.shared_features_extractor:
        #     feature = self.features_extractor(obs)
        #     latent_pi, latent_vf = self.mlp_extractor(feature)
        #     return latent_pi, latent_vf
        # else:
        feature = self.features_extractor(obs)
        latent_pi, latent_vf = self.mlp_extractor(feature)
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
        log_std = self.log_std(latent_pi)
        log_std = th.tanh(log_std)
        # LOG_STD_MAX = 2
        # LOG_STD_MIN = -5
        # log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1) 
        
        return mu, log_std


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
        # NOTE(junweiluo) 25/11/29: use Gassuain
        # distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        # actions = distribution.get_actions(deterministic=deterministic)

        # log_policies = distribution.log_prob(actions)
        
        # NOTE(junweiluo): reparameter action
        # noise = th.randn_like(mean_actions)
        # std = self.log_std.exp()
        # # log_prob = -0.5 * (((action - mu) / std) ** 2 + 2*torch.log(std) + torch.log(torch.tensor(2*3.1415926)))
        # actions = mean_actions + noise * self.log_std.exp()
        # log_policies = -0.5 * (((actions - mean_actions) / (std + 1e-10)) ** 2 + 2*th.log(std) + th.log(th.tensor(2*th.pi)))
        
        # NOTE(junweiluo):使用tanh的版本
        # build Normal action distributin
        normal = th.distributions.Normal(mean_actions, log_std.exp())
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = th.tanh(x_t)
        actions = y_t * self.action_scale + self.action_bias
        log_policies = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_policies -= th.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_policies = log_policies.sum(-1)
        # mean_actions = th.tanh(mean_actions) * self.action_scale + self.action_bias
        values = self.value_net(latent_vf)

        return actions, mean_actions, log_policies, values, x_t
    
    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor, 
        mu: th.Tensor, log_std: th.Tensor,
        noise: th.Tensor, epsilon:float = 1e-8,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        
        
        obs = obs.float()
        latent_pi, latent_vf = self._extract_latent(obs)
        mean_actions, log_std = self.calc_meam_std(latent_pi)
        normal = th.distributions.Normal(mean_actions, log_std.exp())
        # x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = th.tanh(noise)
        # actions = y_t * self.action_scale + self.action_bias
        log_policies = normal.log_prob(noise)
        # Enforcing Action Bound
        log_policies -= th.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_policies = log_policies.sum(-1)
        mean_actions = th.tanh(mean_actions) * self.action_scale + self.action_bias
        
        # NOTE(junweiluo): 

        zero_point = th.zeros_like(actions)
        f_a = self.advantage_net(th.concat([latent_vf, actions], dim = -1))
        f_zero = self.advantage_net(th.concat([latent_vf, zero_point], dim = -1))
        
        # calc hessian matrix
        def f_single(latent_b, action_b):
            return self.advantage_net(th.cat([latent_b, action_b], dim=-1)).squeeze(-1)

        def hess_single(latent_b, action_b):
            # 只对 action_b 求 Hessian
            return hessian(lambda x: f_single(latent_b, x))(action_b)

        hessian_matrix = vmap(hess_single)(latent_vf, zero_point)
        traces = th.einsum("bii->b", hessian_matrix).unsqueeze(-1)
        
        advantages = f_a - f_zero - 0.5 * traces
        advantages = advantages.squeeze(-1)
        

        values = self.value_net(latent_vf)
        
        return values, advantages, log_policies, normal.entropy()
        # return values, advantages, log_probs, distribution.entropy()
    
    def evaluate_state(
        self, obs: th.Tensor, actions: th.Tensor, mu: th.Tensor, log_std: th.Tensor = None, noise: th.Tensor = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:

        return self.evaluate_actions(obs = obs, actions = actions, mu = mu, log_std = log_std, noise = noise)

    def predict_policy(
        self, obs: th.Tensor, actions: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:

        latent_pi, _ = self._extract_latent(obs)

        distribution = self._get_action_dist_from_latent(latent_pi)
        # policies = distribution.distribution.probs
        log_policies = distribution.log_prob(actions)

        return log_policies, distribution.entropy()

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
        else:
            values, advantages, log_probs, entropy = self.evaluate_actions(obs, actions, mu, log_std, noise)
            # build \pi-centered advantage constrant
            # advantages = advantages - advantages_mu
        # keep \pi-centered advantage constrant
        values = self.value_net(latent_vf)
        
        return values, advantages
