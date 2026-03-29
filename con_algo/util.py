
import torch as th
import torch.nn as nn
import os
import numpy as np
from time import time   
from torch.func import vmap, hessian, functional_call, jacrev, grad
from stable_baselines3.common.callbacks import BaseCallback
from torch.distributions import Normal
from typing import Optional, TypeVar
from stable_baselines3.common.distributions import sum_independent_dims
SelfDiagGaussianDistribution = TypeVar("SelfDiagGaussianDistribution", bound="DiagGaussianDistribution")


class DiagGaussianDistribution:
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None
        self.distribution = None

    def proba_distribution(
        self: SelfDiagGaussianDistribution, mean_actions: th.Tensor, log_std: th.Tensor
    ) -> SelfDiagGaussianDistribution:
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: th.Tensor, need_sum = True) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        if need_sum:
            return sum_independent_dims(log_prob)
        else:
            return log_prob


    def entropy(self, need_sum = True) -> Optional[th.Tensor]:
        if need_sum:
            return sum_independent_dims(self.distribution.entropy())
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        return self.distribution.mean

    def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer


class PeriodicCheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_training_start(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # num_timesteps 是“全局 timesteps”
        if self.num_timesteps % self.save_freq == 0 or not hasattr(self, "_saved_init"):
            # time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = os.path.join(
                self.save_path,
                f"model_{self.num_timesteps}.pth"
            )
            self.model.save(path)
            self._saved_init = True
            if self.verbose > 0:
                print(f"Saved model to {path}")
        return True

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim = None, activation = nn.SiLU(),*args, **kwargs):
        super().__init__(*args, **kwargs)
        if hidden_dim == None:
            hidden_dim = input_dim * 4
        # self.layer_norm = nn.LayerNorm(input_dim)
        self.block = nn.Sequential(
            nn.LayerNorm(input_dim),
            layer_init(nn.Linear(input_dim, hidden_dim)),
            # nn.GELU(approximate="tanh"),
            activation,
            layer_init(nn.Linear(hidden_dim, input_dim)),
            # activation,
        )

        # self.block = nn.Sequential(

        # )
    
    def forward(self, x):
        identity = x
        block_out = self.block(x)

        return identity + block_out

class SimBaEncoder(nn.Module):
    def __init__(self, input_dim, block_num, hidden_dim, activation = nn.SiLU(),*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = activation
        self.pre_encoder = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            activation,
        )
        self.residual_encoder = nn.Sequential(*[ResidualBlock(input_dim = hidden_dim, activation=activation) for _ in range(block_num)])
        self.ln2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        fc_out = self.pre_encoder(x)
        feature_ = self.residual_encoder(fc_out)
        
        return self.activation(self.ln2(feature_))
