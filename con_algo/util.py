import torch as th
import torch.nn as nn
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

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

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


