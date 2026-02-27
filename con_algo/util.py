
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

class Actor(nn.Module):
    def __init__(self, action_dim, hidden_dim = 256, activate_func = nn.SiLU()):
        super().__init__()
        self.actor_activate_func = activate_func
        self.observation_feature_extractor = nn.Sequential(
            layer_init(nn.Linear(self.observation_space.shape[0], hidden_dim)),
            self.actor_activate_func,
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            self.actor_activate_func,
        )
        self.action_net = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )
        self.log_std = nn.Parameter(th.zeros(action_dim))

    def forward(self, x):
        feature_ = self.observation_feature_extractor(x)
        mu = self.action_net(feature_)
        log_std = self.log_std
        return mu, log_std

class Critic(nn.Module):
    def __init__(self, hidden_dim = 256, activate_func = nn.SiLU()):
        super().__init__()
        self.critic_activate_func = activate_func
        self.observation_feature_extractor = nn.Sequential(
            layer_init(nn.Linear(self.observation_space.shape[0], hidden_dim)),
            self.critic_activate_func,
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            self.critic_activate_func,
        )
        self.action_feature_extractor = nn.Sequential(
            layer_init(nn.Linear(self.action_space.shape[0], hidden_dim)),
            self.activate_func,
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            self.activate_func,
        )
        self.value_net = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.advantage_net = nn.Sequential(
            layer_init(nn.Linear(hidden_dim * 2, hidden_dim), std=np.sqrt(2)),
            self.activate_func,
            layer_init(nn.Linear(hidden_dim, hidden_dim), std=np.sqrt(2)),
            self.activate_func,
            layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        )

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


    def forward(self, x, actions = None, mu = None, log_std = None):
        feature_ = self.observation_feature_extractor(x)
        if actions is not None:
            action_feature_ = self.action_feature_extractor(actions)
            mu_feature_ = self.action_feature_extractor(mu)
            feature = th.cat([feature_, action_feature_], dim=-1)
            mu_feature = th.cat([feature_, mu_feature_], dim=-1)
            f_a = self.advantage_net(feature)
            f_mu = self.advantage_net(mu_feature)
            sigma = th.exp(log_std).pow(2)
            trace = self.calc_jacrevian_diag(feature_, mu, sigma)
            ex_adv = f_mu + trace
            advantage = f_a - ex_adv
        else:
            advantage = None
        value = self.value_net(feature_)
        return value, advantage

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
