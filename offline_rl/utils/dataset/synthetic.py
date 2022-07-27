import numpy as np
import torch as th


def one_dim_wrapper(base):
    def wrapped(n):
        x, y = base(n)
        return np.expand_dims(x, axis=1), np.expand_dims(y, axis=1)


def torch_wrapper(base):
    def wrapped(n):
        x, y = base(n)
        return th.Tensor(x, axis=1), th.Tensor(y, axis=1)


def linear(n):
    x = np.linspace(0, 1, num=n)
    epsilon = np.random.normal(size=n)
    y = 5 * x + np.square(x) * epsilon
    return x, y


def curve_hard(n):
    x = np.linspace(-1, 1, num=n)
    epsilon = np.random.normal(size=n)
    y = x**2 + epsilon * (x**2 * 0.2 + 0.03)
    return x, y


def curve_simple(n):
    x = np.linspace(-10, 10, num=n)
    epsilon = np.random.normal(size=n)
    y = x**2 + epsilon * np.abs(x)
    return x / 10, y


curve = curve_simple


def cross(n):
    x1, y1 = curve(n // 2)
    x2, y2 = curve(n // 2)
    return np.concatenate([x1, x2]), np.concatenate([y1, -y2])


def step(n):
    x = np.linspace(0, 1, num=n)
    epsilon = np.random.normal(size=n)
    y1 = 3 * x + epsilon * 0.15
    y2 = x + epsilon * 0.15
    return x, np.where(x < 0.3, y1, y2)
