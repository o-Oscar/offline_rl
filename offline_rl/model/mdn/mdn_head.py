import numpy as np
import torch as th
import torch.nn as nn

ONEOVERSQRT2PI = 1.0 / np.sqrt(2 * np.pi)
LOG2PI = np.log(2 * np.pi)


class MixtureDensityHead(nn.Module):
    def __init__(self, input_shape: int, num_gaussian: int, output_size: int):
        super().__init__()
        self.input_shape = input_shape
        self.num_gaussian = num_gaussian
        self.output_size = output_size
        self.output_shape = self.input_shape[:-1] + (
            self.num_gaussian,
            self.output_size,
        )

        self.pi_linear = nn.Linear(self.input_shape[-1], self.num_gaussian)
        self.sigma_linear = nn.Linear(self.input_shape[-1], self.num_gaussian)
        self.mu_linear = nn.Linear(
            self.input_shape[-1], self.num_gaussian * self.output_size
        )

    def forward(self, x: th.Tensor):
        """
        x : (*, input_size)

        returns : (*, output_size, num_gaussian)
        """
        pi = self.pi_linear(x)
        sigma = nn.ELU()(self.sigma_linear(x)) + 1 + 1e-15
        mu = self.mu_linear(x).view(self.output_shape)
        return pi, sigma, mu

    def compute_density(
        self, pi: th.Tensor, sigma: th.Tensor, mu: th.Tensor, t: th.Tensor
    ):
        """
        pi : (*, num_gaussian)
        sigma : (*, num_gaussian)
        mu : (*, num_gaussian, output_size)
        t : (*, output_size)

        returns : (*, )
        """
        t = th.unsqueeze(t, -2)

        a = pi.softmax(dim=-1)
        dists = th.sum(th.square((t - mu)), axis=-1)
        log_exponential = -0.5 * dists / sigma

        compound = a * ONEOVERSQRT2PI / sigma * th.exp(log_exponential)
        return th.sum(compound, dim=-1)

    def compute_log_density(
        self, pi: th.Tensor, sigma: th.Tensor, mu: th.Tensor, t: th.Tensor
    ):
        """
        pi : (*, num_gaussian)
        sigma : (*, num_gaussian)
        mu : (*, num_gaussian, output_size)
        t : (*, output_size)

        returns : (*, )
        """
        t = th.unsqueeze(t, -1)

        log_frac = th.log(ONEOVERSQRT2PI / sigma)
        dists = th.sum(th.square((t - mu)), axis=-1)
        log_exponential = -0.5 * dists / sigma
        log_mix = pi - th.logsumexp(pi, keepdims=True, dim=-1)
        return th.logsumexp(log_frac + log_exponential + log_mix, dim=-1)

    def density(self, x: th.Tensor, target: th.Tensor):
        pi, sigma, mu = self(x)
        return self.compute_density(pi, sigma, mu, target)

    def log_density(self, x: th.Tensor, target: th.Tensor):
        pi, sigma, mu = self(x)
        return self.compute_log_density(pi, sigma, mu, target)
