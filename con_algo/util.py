
import torch as th
import torch.nn as nn
import os
import numpy as np
from time import time   
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


    def entropy(self) -> Optional[th.Tensor]:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.sample()

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


class EvalCallback(BaseCallback):
    def __init__(self, eval_env, n_eval_episodes=64, eval_freq=100_000,
                 gamma=0.99, lambda_gae=0.95, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.last_eval_step = 0

    def _on_step(self) -> bool:
        # 每 eval_freq 步评估一次
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            total_returns = []

            for ep in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                rewards = []

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
                    rewards.append(reward)

                total_returns.append(np.sum(rewards))

            # 返回评估指标
            mean_return = np.mean(total_returns)
            std_return = np.std(total_returns)

            # 可打印
            if self.verbose > 0:
                print(f"[Step {self.num_timesteps}] Eval mean: {mean_return:.2f}, std: {std_return:.2f}")

            # TensorBoard log
            self.logger.record("eval/ep_return_mean", mean_return, self.num_timesteps)
            self.logger.record("eval/ep_return_std", std_return, self.num_timesteps)

            # 可选保存模型

        return True



class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim = None, activation = nn.SiLU(),*args, **kwargs):
        super().__init__(*args, **kwargs)
        if hidden_dim == None:
            hidden_dim = input_dim * 4
        # self.layer_norm = nn.LayerNorm(input_dim)
        self.block = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            # nn.GELU(approximate="tanh"),
            activation,
            nn.Linear(hidden_dim, input_dim),
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
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation,
        )
        self.residual_encoder = nn.Sequential(*[ResidualBlock(input_dim = hidden_dim, activation=activation) for _ in range(block_num)])
        self.ln2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        fc_out = self.pre_encoder(x)
        feature_ = self.residual_encoder(fc_out)
        
        return self.activation(self.ln2(feature_))
