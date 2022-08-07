import numpy as np
import torch as th
import torch.nn as nn
from offline_rl.model.backbone.convolutionnal import CNN2d, ResNet, SimpleResNet
from offline_rl.model.backbone.fcnn import FCNN
from offline_rl.model.backbone.unet import ResUNet, UNet
from offline_rl.model.base import BaseModel
from offline_rl.model.mdn.mdn_head import MixtureDensityHead
from offline_rl.utils.dataset.diffusion import get_diffusion_params


class SimpleMDN(BaseModel):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.backbone = FCNN([input_size, 100, 100])
        self.head = MixtureDensityHead(
            (
                -1,
                self.backbone.sizes[-1],
            ),
            30,
            output_size,
        )

    def forward(self, x: th.Tensor):
        features = self.backbone(x)
        return self.head(features)

    def density(self, x: th.Tensor, target: th.Tensor):
        features = self.backbone(x)
        return self.head.density(features, target)

    def log_density(self, x: th.Tensor, target: th.Tensor):
        features = self.backbone(x)
        return self.head.log_density(features, target)

    def standard_dist(self, x: th.Tensor):
        features = self.backbone(x)
        return self.head.standard_dist(features)


class ConvMDN(BaseModel):
    def __init__(self, input_size: tuple, output_size: int):
        super().__init__()

        cnn_layers = 2
        assert input_size[0] % (2**cnn_layers) == 0
        assert input_size[1] % (2**cnn_layers) == 0

        self.cnn = CNN2d([input_size[-1], 16, 16], [3, 3])

        cnn_out_size = (
            input_size[0]
            // (2**cnn_layers)
            * input_size[1]
            // (2**cnn_layers)
            * self.cnn.channel_sizes[-1]
        )

        self.fcnn = FCNN((cnn_out_size, 128))

        self.head = MixtureDensityHead(
            (
                -1,
                self.fcnn.sizes[-1],
            ),
            30,
            output_size,
        )

    def forward(self, x: th.Tensor):
        conv_features = th.flatten(self.cnn(x), 1)
        print(conv_features.shape)
        flat_features = self.fcnn(conv_features)
        print(flat_features.shape)
        return self.head(flat_features)

    def density(self, x: th.Tensor, target: th.Tensor):
        conv_features = th.flatten(self.cnn(x), 1)
        flat_features = self.fcnn(conv_features)
        return self.head.density(flat_features, target)

    def log_density(self, x: th.Tensor, target: th.Tensor):
        conv_features = th.flatten(self.cnn(x), 1)
        flat_features = self.fcnn(conv_features)
        return self.head.log_density(flat_features, target)

    def standard_dist(self, x: th.Tensor):
        conv_features = th.flatten(self.cnn(x), 1)
        flat_features = self.fcnn(conv_features)
        return self.head.standard_dist(flat_features)


class DiffusionNet(BaseModel):
    def __init__(self):
        super().__init__()

        self.diffusion_steps = 100
        (
            self.alphas,
            self.alphas_bar,
            self.betas,
            self.betas_tilde,
        ) = get_diffusion_params(self.diffusion_steps)
        self.sigmas = self.betas
        self.sigmas = self.betas_tilde

    def get_random_ts(self, x0s: th.Tensor):
        flat_ts = np.random.randint(
            self.diffusion_steps, size=x0s.shape[0], dtype=np.int32
        )
        expanded_ts = np.expand_dims(flat_ts, axis=(1, 2, 3))
        target_shape = x0s.shape[:3] + (1,)
        full_ts = expanded_ts * np.ones(target_shape)
        ts_th = th.Tensor(full_ts / self.diffusion_steps)
        return expanded_ts, ts_th

    def from_unit_t(self, x0s: th.Tensor, t: float):
        target_shape = x0s.shape[:3] + (1,)
        return th.Tensor(np.ones(target_shape) * t) / self.diffusion_steps

    def loss(self, x0s: th.Tensor):
        ts, ts_th = self.get_random_ts(x0s)

        epsilon = np.random.normal(size=x0s.shape)
        epsilon_th = th.Tensor(epsilon)

        x_fac = th.Tensor(np.sqrt(self.alphas_bar[ts]))
        epsilon_fac = th.Tensor(np.sqrt(1 - self.alphas_bar[ts]))

        xts = x_fac * x0s + epsilon_fac * epsilon_th

        outputs = self(xts, ts_th)
        to_return = 0
        for output in outputs:
            to_return += th.mean(th.square(output - epsilon_th))

        return to_return

    def generate(self, batch_size: int):

        xts = np.random.normal(size=(batch_size, 8, 8, 7))
        xts_th = th.Tensor(xts)
        for t in reversed(range(self.diffusion_steps)):
            z = th.Tensor(np.random.normal(size=xts.shape))
            if t == 0:
                z *= 0

            ts_th = self.from_unit_t(xts_th, t)
            epsilon = self(xts_th, ts_th)[-1]

            x_fac = 1 / np.sqrt(self.alphas[t])
            epsilon_fac = (1 - self.alphas[t]) / np.sqrt(1 - self.alphas_bar[t])

            xts_th = x_fac * (xts_th - epsilon_fac * epsilon) + self.sigmas[t] * z
        return xts_th

    def generate_partial(self, x0s: np.ndarray, diffusion_steps: int):
        x0s_th = th.Tensor(x0s)

        epsilon = np.random.normal(size=x0s.shape)
        import matplotlib.pyplot as plt

        # plt.hist(epsilon.flatten())
        # plt.show()
        # exit()
        epsilon_th = th.Tensor(epsilon)

        x_fac = np.sqrt(self.alphas_bar[diffusion_steps])
        epsilon_fac = np.sqrt(1 - self.alphas_bar[diffusion_steps])
        # print(self.alphas_bar[diffusion_steps])
        xts_th = x_fac * x0s_th + epsilon_fac * epsilon_th
        # return xts_th

        for t in reversed(range(diffusion_steps)):
            z = th.Tensor(np.random.normal(size=xts_th.shape))
            if t == 0:
                z *= 0

            ts_th = self.from_unit_t(xts_th, t)
            epsilon = self(xts_th, ts_th)[-1]

            x_fac = 1 / np.sqrt(self.alphas[t])
            epsilon_fac = (1 - self.alphas[t]) / np.sqrt(1 - self.alphas_bar[t])

            xts_th = x_fac * (xts_th - epsilon_fac * epsilon) + self.sigmas[t] * z
        return xts_th

    def embed_t(self, t: th.Tensor, n_sin):
        # return t
        to_return = []
        for i in range(n_sin):
            phase = t * np.pi * (2**i)
            to_return.append(th.sin(phase))
            to_return.append(th.cos(phase))

        return th.concat(to_return, axis=-1)


class DiffusionUNet(DiffusionNet):
    def __init__(self, input_size: int, diffusion_steps: int):
        super().__init__()
        self.input_size = input_size

        in_channels = 8
        unet_channels = 32

        self.unet = ResUNet(
            # self.unet = UNet(
            in_size=input_size,
            in_channels=in_channels,
            compute_channels=unet_channels,
            depth=2,
        )

        self.last_layer = nn.Conv2d(
            unet_channels * 4,
            in_channels - 1,
            1,
            padding="same",
            padding_mode="reflect",
        )

        self.diffusion_steps = diffusion_steps
        (
            self.alphas,
            self.alphas_bar,
            self.betas,
            self.betas_tilde,
        ) = get_diffusion_params(self.diffusion_steps)
        self.sigmas = self.betas

    def forward(self, x: th.Tensor, t_th: th.Tensor):
        inp = th.concat([x, t_th], dim=-1)
        inp = th.transpose(inp, 1, 3)
        unet_out = self.unet(inp)
        result = self.last_layer(unet_out)
        return [th.transpose(result, 1, 3)]


class DiffusionMultiUNet(DiffusionNet):
    def __init__(self, input_size: int, diffusion_steps: int):
        super().__init__()
        self.input_size = input_size

        in_channels = 8
        unet_channels = 16
        unets = []
        for i in range(3):
            unet_in_size = in_channels if i == 0 else (in_channels + unet_channels * 2)
            unets.append(
                UNet(
                    in_size=input_size,
                    in_channels=unet_in_size,
                    compute_channels=unet_channels,
                    depth=2,
                )
            )
        self.unets = nn.ModuleList(unets)

        self.last_layer = nn.Conv2d(
            unet_channels * 2,
            in_channels - 1,
            1,
            padding="same",
            padding_mode="reflect",
        )

        self.diffusion_steps = diffusion_steps
        (
            self.alphas,
            self.alphas_bar,
            self.betas,
            self.betas_tilde,
        ) = get_diffusion_params(self.diffusion_steps)
        self.sigmas = self.betas

    def forward(self, x: th.Tensor, t_th: th.Tensor):
        # t_th = th.ones((x.shape[0], 1, 8, 8)) * t
        inp = th.concat([x, t_th], dim=-1)
        inp = th.transpose(inp, 1, 3)
        all_unet_out = []
        for i, unet in enumerate(self.unets):
            if i == 0:
                cur_features = inp
            else:
                cur_features = th.concat([inp, unet_out.detach()], axis=1)
            unet_out = unet(cur_features)
            all_unet_out.append(unet_out)

        all_results = [self.last_layer(unet_out) for unet_out in all_unet_out]
        return [th.transpose(result, 1, 3) for result in all_results]


class DiffusionResNet(DiffusionNet):
    def __init__(self):
        super().__init__()

        # self.resnet = ResNet()
        self.resnet = SimpleResNet()

    def forward(self, x: th.Tensor, t_th: th.Tensor):
        # inp = th.concat([x, t_th], dim=-1)
        inp = x
        inp = th.transpose(inp, 1, 3)
        th_embed = self.embed_t(t_th, 4)
        th_embed = th.transpose(th_embed, 1, 3)

        resnet_out = self.resnet(inp, th_embed)
        return [th.transpose(resnet_out, 1, 3)]
