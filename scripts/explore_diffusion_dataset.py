import json
from curses import meta
from pathlib import Path

import gym
import gym_minigrid
import matplotlib.pyplot as plt
import numpy as np
from offline_rl.utils.dataset.diffusion import xor

ds = xor(10000)

scale = 0.2
diffused = ds * (1 - scale) + np.random.normal(0, np.sqrt(scale), size=ds.shape)

plt.plot(diffused[:, 0], diffused[:, 1], ".")
plt.plot(ds[:, 0], ds[:, 1], ".")
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.show()
