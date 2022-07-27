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
            self.output_size,
            self.num_gaussian,
        )

        self.pi_linear = nn.Linear(
            self.input_shape[-1], self.num_gaussian * self.output_size
        )
        self.sigma_linear = nn.Linear(
            self.input_shape[-1], self.num_gaussian * self.output_size
        )
        self.mu_linear = nn.Linear(
            self.input_shape[-1], self.num_gaussian * self.output_size
        )

    def apply_view(self, x: th.Tensor):
        return x.view(self.output_shape)

    def forward(self, x: th.Tensor):
        """
        x : (*, input_size)

        returns : (*, output_size, num_gaussian)
        """
        pi = self.pi_linear(x).view(self.output_shape)
        sigma = nn.ELU()(self.sigma_linear(x).view(self.output_shape)) + 1 + 1e-15
        mu = self.mu_linear(x).view(self.output_shape)
        return pi, sigma, mu

    def density(self, pi, sigma, mu, t):
        """
        a : (*, output_size, num_gaussian)
        sigma : (*, output_size, num_gaussian)
        mu : (*, output_size, num_gaussian)
        t : (*, output_size)

        returns : (*, )
        """
        t = th.unsqueeze(t, 1)
        a = th.softmax(pi)
        compound = (
            a * ONEOVERSQRT2PI / sigma * th.exp(-0.5 * th.square((t - mu) / sigma))
        )
        return th.sum(compound, dim=-1)

    def log_density(self, a, sigma, mu, t):
        """
        a : (*, output_size, num_gaussian)
        sigma : (*, output_size, num_gaussian)
        mu : (*, output_size, num_gaussian)
        t : (*, output_size)

        returns : (*, )
        """
        log_frac = th.log(ONEOVERSQRT2PI / sigma)
        log_exponential = -0.5 * th.square((t - mu) / sigma)
        log_mix = a - th.logsumexp(a, dim=-1, keepdim=True)
        return th.logsumexp(log_frac + log_exponential + log_mix, dim=-1)
