import numpy as np
import torch as th
import torch.nn as nn
from offline_rl.model.backbone.fcnn import FCNN
from offline_rl.model.mdn.mdn_head import MixtureDensityHead


class SimpleMDN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.backbone = FCNN([input_size, 10, 10])
        self.head = MixtureDensityHead(
            (
                -1,
                10,
            ),
            2,
            output_size,
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(self.backbone(features))

    def density(self, x: th.Tensor, target: th.Tensor):
        features = self.backbone(x)
        return self.head.density(features, target)

    def log_density(self, x: th.Tensor, target: th.Tensor):
        features = self.backbone(x)
        return self.head.log_density(features, target)
