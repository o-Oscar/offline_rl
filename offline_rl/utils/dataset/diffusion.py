from typing import Callable

import numpy as np


def line(n):
    x = np.linspace(-1, 1, num=n)
    y = x * 0
    return np.stack([x, y], axis=1)


def curve(n):
    x = np.linspace(-1, 1, num=n)
    epsilon = np.random.normal(size=n)
    y = x**2  # + epsilon * (x**2 * 0.2 + 0.03)
    return np.stack([x, y], axis=1)


def cross(n):
    ds1 = curve(n // 2)
    ds2 = curve(n // 2)
    ds2 *= np.array([1, -1]).reshape([1, 2])
    return np.concatenate([ds1, ds2], axis=0)


def xor(n):
    x = np.concatenate([np.ones((n // 2, 1)), np.zeros((n // 2, 1))])
    y = 1 - x
    return np.concatenate([x, y], axis=1)


def get_diffusion_params_basic(T):
    # betas = np.linspace(1e-4, 0.02, T)
    betas = np.linspace(
        1e-4, 3 / T, T
    )  # I thought it was good. mabe not good enough for training
    # betas = np.linspace(1e-4, 0.3 / T, T)
    alphas = 1 - betas
    alphas_bar = np.cumprod(alphas)
    betas_tilde = (1 - alphas_bar[:-1]) / (1 - alphas_bar[1:]) * betas[1:]
    betas_tilde = np.concatenate([[0], betas_tilde])

    return alphas, alphas_bar, betas, betas_tilde


def get_diffusion_params_cos(T):
    xs = np.linspace(1e-4, 1 - 1e-4, T)
    alphas_bar = (np.cos(xs * np.pi) + 1) / 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = np.concatenate([[alphas_bar[0]], alphas])
    betas = 1 - alphas
    betas_tilde = (1 - alphas_bar[:-1]) / (1 - alphas_bar[1:]) * betas[1:]
    betas_tilde = np.concatenate([[0], betas_tilde])

    return alphas, alphas_bar, betas, betas_tilde


def get_diffusion_params_linear(T):
    xs = np.linspace(1e-4, 1 - 1e-4, T)
    alphas_bar = 1 - xs
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = np.concatenate([[alphas_bar[0]], alphas])
    betas = 1 - alphas
    betas_tilde = (1 - alphas_bar[:-1]) / (1 - alphas_bar[1:]) * betas[1:]
    betas_tilde = np.concatenate([[0], betas_tilde])

    return alphas, alphas_bar, betas, betas_tilde


def get_diffusion_params(*args, **kwargs):
    return get_diffusion_params_basic(*args, **kwargs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # xs = np.linspace(0, 10)
    # ys = 1 - np.exp(-xs)
    # plt.plot(xs, ys)
    # plt.show()
    # exit()

    T = 1000
    alphas, alphas_bar, betas, betas_tilde = get_diffusion_params(T)

    plt.figure()
    plt.plot(alphas)
    plt.title("alphas")
    plt.figure()
    plt.plot(alphas_bar)
    plt.title("alphas_bar")
    plt.figure()
    plt.plot(betas)
    plt.title("betas")
    plt.figure()
    plt.plot(betas_tilde)
    plt.title("betas_tilde")
    plt.show()
