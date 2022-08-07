import numpy as np
import torch as th
import torch.nn as nn
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

    def loss(self, x0s: th.Tensor):
        t = np.random.randint(self.diffusion_steps, dtype=np.int32)
        return self.loss_t(x0s, t)

    def loss_t(self, x0s: th.Tensor, t: int):
        epsilon = np.random.normal(size=x0s.shape)
        epsilon_th = th.Tensor(epsilon)

        x_fac = np.sqrt(self.alphas_bar[t])
        epsilon_fac = np.sqrt(1 - self.alphas_bar[t])

        xts = x_fac * x0s + epsilon_fac * epsilon_th

        output = self(xts, t / self.diffusion_steps, self.alphas_bar[t])
        # output = self.calc_x0(xts, t / self.diffusion_steps, self.alphas_bar[t])
        target = epsilon_fac * epsilon_th
        # target = epsilon_th
        # target = x0s

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


class DiffusionResNet(DiffusionNet):
    def __init__(self):
        super().__init__()

        self.resnet = SimpleResNet()

        self.sample_shape = (7, 3, 3)
        self.output_shape = (1,) + self.sample_shape
        self.feature_shape = (1, 1) + self.sample_shape[1:]

        xs = np.linspace(0, 1, self.sample_shape[-1])
        xs, ys = np.meshgrid(xs, xs)
        xs = xs.reshape(self.feature_shape)
        ys = ys.reshape(self.feature_shape)
        xs = self.embed(xs, 4, 1)
        ys = self.embed(ys, 4, 1)
        self.pos_th = th.Tensor(np.concatenate([xs, ys], axis=1))

    def calc_x0(self, x: th.Tensor, t: float, alpha_bar: float):
        t_embeded = self.embed(t * np.ones(self.feature_shape), 4, 1)
        t_embeded = th.Tensor(t_embeded)

        batch_ones = th.ones(x.shape[:1] + (1,) * len(self.sample_shape))

        inp = th.concat([x, self.pos_th * batch_ones, t_embeded * batch_ones], axis=1)
        # t_embed = t * np.ones(x.shape[:1] + self.feature_shape[1:])
        # inp = th.concat([x, self.pos_th * batch_ones, th.Tensor(t_embed)], axis=1)
        # inp = th.concat([x, th.Tensor(t_embed)], axis=1)

        return self.resnet(inp)

    def forward(self, x: th.Tensor, t: float, alpha_bar: float):
        return x - np.sqrt(alpha_bar) * self.calc_x0(x, t, alpha_bar)

        # return self.calc_x0(x, t, alpha_bar)  # / np.sqrt(1 - alpha_bar)


from offline_rl.model import FCNN


class DiffusionFCNN(DiffusionNet):
    def __init__(self):
        super().__init__()

        self.model = FCNN([8, 32, 32, 7], last_activation=False)
        self.sample_shape = (7, 1, 1)

    def forward(self, x: th.Tensor, t: float, alpha_bar: float):
        inp = x.reshape((x.shape[0], 7))
        t_embeded = t * np.ones((x.shape[0], 1))
        t_embeded = th.Tensor(t_embeded)
        inp = th.concat([inp, t_embeded], axis=1)

        x0 = self.model(inp).reshape(x.shape)

        return x - np.sqrt(alpha_bar) * x0


class SimpleResNet(nn.Module):
    def __init__(self):
        super().__init__()

        sample_channels = 7
        in_channels = sample_channels + 8 + 16  # + 16  # 8 + 16
        compute_channels = 64

        self.layer_0 = nn.Conv2d(
            in_channels,
            compute_channels,
            1,
            padding="same",
            padding_mode="reflect",
        )
        self.layer_1 = nn.Conv2d(
            in_channels + compute_channels,
            compute_channels,
            1,
            padding="same",
            padding_mode="reflect",
        )
        # self.layer_11 = nn.Conv2d(
        #     in_channels + compute_channels,
        #     compute_channels,
        #     1,
        #     padding="same",
        #     padding_mode="reflect",
        # )
        self.layer_2 = nn.Conv2d(
            in_channels + compute_channels,
            sample_channels,
            1,
            padding="same",
            padding_mode="reflect",
        )

    def forward(self, x):
        inp = x
        x = th.relu(self.layer_0(x))
        x = th.concat([x, inp], axis=1)
        x = th.relu(self.layer_1(x))
        x = th.concat([x, inp], axis=1)
        # x = th.relu(self.layer_11(x))
        # x = th.concat([x, inp], axis=1)
        x = self.layer_2(x)

        return x
