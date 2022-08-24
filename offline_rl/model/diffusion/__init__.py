import numpy as np
import torch as th
from offline_rl.model.base import BaseModel
from offline_rl.utils.dataset.diffusion import get_diffusion_params


class DiffusionNet(BaseModel):
    def __init__(self):
        super().__init__()

        self.diffusion_steps = 1000
        (
            self.alphas,
            self.alphas_bar,
            self.betas,
            self.betas_tilde,
        ) = get_diffusion_params(self.diffusion_steps)
        self.sigmas = self.betas
        self.sigmas = self.betas_tilde

        self.predicts_x0 = True

    def loss(self, x0s: th.Tensor, infos: th.Tensor = None):
        t = np.random.randint(self.diffusion_steps, dtype=np.int32)
        return self.loss_t(x0s, t, infos)

    def loss_t(self, x0s: th.Tensor, t: int, infos: th.Tensor):
        epsilon = np.random.normal(size=x0s.shape)
        epsilon_th = th.Tensor(epsilon)

        x_fac = np.sqrt(self.alphas_bar[t])
        epsilon_fac = np.sqrt(1 - self.alphas_bar[t])

        xts = x_fac * x0s + epsilon_fac * epsilon_th

        output = self(xts, t / self.diffusion_steps, self.alphas_bar[t], infos)
        target = epsilon_fac * epsilon_th

        return th.mean(th.square(output - target))

    def generate(self, batch_size: int):

        xts = np.random.normal(size=(batch_size,) + self.sample_shape)
        xts_th = th.Tensor(xts)
        for t in reversed(range(0, self.diffusion_steps, 1)):
            print(t)
            z = th.Tensor(np.random.normal(size=xts.shape))
            if t == 0:
                z *= 0

            # epsilon = self(xts_th, 0.1)
            epsilon = self(
                xts_th, t / self.diffusion_steps, self.alphas_bar[t]
            ) / np.sqrt(1 - self.alphas_bar[t])

            x_fac = 1 / np.sqrt(self.alphas[t])
            epsilon_fac = (1 - self.alphas[t]) / np.sqrt(1 - self.alphas_bar[t])

            xts_th = x_fac * (xts_th - epsilon_fac * epsilon) + self.sigmas[t] * z
        return xts_th

    def generate_ddim(
        self, batch_size: int, speedup_fac: int = 10, infos: th.Tensor = None
    ):
        xts = np.random.normal(size=(batch_size,) + self.sample_shape)
        xts_th = th.Tensor(xts)
        used_tm1 = list(reversed(range(0, self.diffusion_steps, speedup_fac)))
        # used_t = [self.diffusion_steps] + used_tm1[:-1]
        used_t = used_tm1[:-1]
        used_tm1 = used_tm1[1:]
        for tm1, t in zip(used_tm1, used_t):
            print(t)
            # epsilon = self(xts_th, 0.1)
            epsilon = self(
                xts_th, t / self.diffusion_steps, self.alphas_bar[t], infos=infos
            ) / np.sqrt(1 - self.alphas_bar[t])

            pred_x_0 = (xts_th - np.sqrt(1 - self.alphas_bar[t]) * epsilon) / np.sqrt(
                self.alphas_bar[t]
            )

            dir_to_xt = np.sqrt(1 - self.alphas_bar[tm1]) * epsilon

            xts_th = np.sqrt(self.alphas_bar[tm1]) * pred_x_0 + dir_to_xt
        return xts_th

    def generate_partial(
        self, x0s: np.ndarray, diffusion_steps: int
    ):  # Not working properly
        x0s_th = th.Tensor(x0s)

        epsilon = np.random.normal(size=x0s.shape)
        epsilon_th = th.Tensor(epsilon)

        x_fac = np.sqrt(self.alphas_bar[diffusion_steps])
        epsilon_fac = np.sqrt(1 - self.alphas_bar[diffusion_steps])
        xts_th = x_fac * x0s_th + epsilon_fac * epsilon_th

        for t in reversed(range(diffusion_steps)):
            z = th.Tensor(np.random.normal(size=xts_th.shape))
            if t == 0:
                z *= 0

            epsilon = self(xts_th, t / self.diffusion_steps, self.alphas_bar[t])

            x_fac = 1 / np.sqrt(self.alphas[t])
            epsilon_fac = (1 - self.alphas[t]) / np.sqrt(1 - self.alphas_bar[t])

            xts_th = x_fac * (xts_th - epsilon_fac * epsilon) + self.sigmas[t] * z
        return xts_th

    def embed(self, t: th.Tensor, feature_nb: int, axis: int):
        to_return = []
        for i in range(feature_nb):
            phase = t * np.pi * (2**i)
            to_return.append(np.sin(phase))
            to_return.append(np.cos(phase))

        return np.concatenate(to_return, axis=axis)
