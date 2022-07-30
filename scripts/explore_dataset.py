import json
from curses import meta
from pathlib import Path

import gym
import gym_minigrid
import numpy as np
from offline_rl.dataset import Dataset, load_dataset

save_path = Path("results/dataset/dataset_0")
ds = load_dataset(save_path)
env = gym.make(ds.metadata["env_name"])

print(ds)

ds.plot_rollout(2)
