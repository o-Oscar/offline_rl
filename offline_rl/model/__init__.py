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
