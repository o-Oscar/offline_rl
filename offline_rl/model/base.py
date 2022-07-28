from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.save_path = None

    def update_save_path(self, save_path: Path = None):
        if save_path is not None:
            self.save_path = save_path
        if self.save_path is None:
            raise ValueError("No save path provided")

    def save(self, save_path: Path = None):
        self.update_save_path(save_path)
        th.save(self.state_dict(), self.save_path)

    def load(self, save_path: Path = None):
        self.update_save_path(save_path)
        self.load_state_dict(th.load(self.save_path))
