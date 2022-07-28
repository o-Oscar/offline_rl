import numpy as np
import torch as th
import torch.nn as nn
from offline_rl.model.backbone.fcnn import FCNN
from offline_rl.model.base import BaseModel
from offline_rl.model.mdn.mdn_head import MixtureDensityHead


class SimpleMDN(BaseModel):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.backbone = FCNN([input_size, 100, 100])
        self.head = MixtureDensityHead(
            (
                -1,
                self.backbone.sizes[-1],
            ),
            4,
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
