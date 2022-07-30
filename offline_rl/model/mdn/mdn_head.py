import numpy as np
import torch as th
import torch.nn as nn

ONEOVERSQRT2PI = 1.0 / np.sqrt(2 * np.pi)
LOG2PI = np.log(2 * np.pi)


class MixtureDensityHead(nn.Module):
    def __init__(self, input_shape: int, num_gaussian: int, output_size: int):
        super().__init__()
        self.base_shape = input_shape[:-1]
        self.input_size = input_shape[-1]
        self.num_gaussian = num_gaussian
        self.output_size = output_size
        self.output_shape = self.base_shape + (
            self.num_gaussian,
            self.output_size,
        )
        self.mu_1_shape = self.base_shape + (
            self.num_gaussian,
            1,
            self.output_size,
        )
        self.mu_2_shape = self.base_shape + (
            1,
            self.num_gaussian,
            self.output_size,
        )

        self.pi_linear = nn.Linear(self.input_size, self.num_gaussian)
        self.sigma_linear = nn.Linear(self.input_size, self.num_gaussian)
        self.mu_linear = nn.Linear(
            self.input_size, self.num_gaussian * self.output_size
        )

        self.bias = th.Tensor(
            np.linspace(0, 1, self.num_gaussian).reshape(1, self.num_gaussian, 1)
        )

    def forward(self, x: th.Tensor):
        """
        x : (*, input_size)

        returns : (*, output_size, num_gaussian)
        """
        pi = self.pi_linear(x)
        # sigma = nn.ELU()(self.sigma_linear(x)) + 1 + 1e-15
        # sigma = th.sigmoid(self.sigma_linear(x)) + 1e-2
        sigma = th.sigmoid(self.sigma_linear(x)) * 0 + 1 / self.num_gaussian
        mu = self.mu_linear(x).view(self.output_shape) * 0 + self.bias
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
        log_exponential = -0.5 * dists / th.square(sigma)

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

        # sigma = sigma.detach()
        log_frac = th.log(ONEOVERSQRT2PI / sigma)
        dists = th.sum(th.square((t - mu)), axis=-1)
        log_exponential = -0.5 * dists / th.square(sigma)
        log_mix = pi - th.logsumexp(pi, keepdims=True, dim=-1)
        return th.logsumexp(log_frac + log_exponential + log_mix, dim=-1)

    def compute_standard_dist_old(self, pi: th.Tensor, sigma: th.Tensor, mu: th.Tensor):
        """
        pi : (*, num_gaussian)
        sigma : (*, num_gaussian)
        mu : (*, num_gaussian, output_size)

        returns : (*, )
        """
        mu1 = mu.view(
            self.base_shape
            + (
                1,
                self.num_gaussian,
                self.output_size,
            )
        )
        mu2 = mu.view(
            self.base_shape
            + (
                self.num_gaussian,
                1,
                self.output_size,
            )
        )
        sigma_1 = sigma.view(
            self.base_shape
            + (
                1,
                self.num_gaussian,
            )
        )
        sigma_2 = sigma.view(
            self.base_shape
            + (
                self.num_gaussian,
                1,
            )
        )
        mu_dists = th.sum(th.square(mu1 - mu2), axis=-1)
        # mu_dists = th.sqrt(th.sum(th.square(mu1 - mu2), axis=-1) + 1e-15)
        sigmas = 2 * sigma_1 * sigma_2 / (sigma_1 + sigma_2)
        xs = 1 + mu_dists / sigmas.detach()
        fs = 1 / th.square(xs) - 1 / xs
        a = th.sum(fs, dim=(-1, -2))
        b = th.sum(th.square(sigma / 0.3 - 1), axis=-1)
        return th.sum(fs, dim=(-1, -2)) + th.sum(th.square(sigma / 0.3 - 1), axis=-1)

    def compute_standard_dist(self, pi: th.Tensor, sigma: th.Tensor, mu: th.Tensor):
        """
        pi : (*, num_gaussian)
        sigma : (*, num_gaussian)
        mu : (*, num_gaussian, output_size)

        returns : (*, )
        """
        xs = sigma / 0.3
        fs = th.log(xs) + 0.5 / th.square(xs)
        fs = fs  # * (1 - pi.softmax(dim=-1))
        return th.sum(fs, dim=(-1))

    def density(self, x: th.Tensor, target: th.Tensor):
        pi, sigma, mu = self(x)
        return self.compute_density(pi, sigma, mu, target)

    def log_density(self, x: th.Tensor, target: th.Tensor):
        pi, sigma, mu = self(x)
        return self.compute_log_density(pi, sigma, mu, target)

    def standard_dist(self, x: th.Tensor):
        pi, sigma, mu = self(x)
        return self.compute_standard_dist(pi, sigma, mu)
