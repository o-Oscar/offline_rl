# import gym_minigrid
import matplotlib.pyplot as plt
import numpy as np
from offline_rl.utils.dataset.diffusion import get_diffusion_params

# plt.plot(xs, ys)
# plt.show()


# import numpy as np

# xs = np.linspace(0, 5)
# ys = 1 / np.square(xs + 1) - 1 / (xs + 1)


diffusion_steps = 1000
(
    alphas,
    alphas_bar,
    betas,
    betas_tilde,
) = get_diffusion_params(diffusion_steps)

sigma2 = 1 - alphas_bar

for t in range(10, diffusion_steps, 100):
    xs = np.linspace(-1, 1, 100)
    ys = np.exp(-np.square(xs - 1) / 2 / sigma2[t])
    zs = np.exp(-np.square(xs + 1) / 2 / sigma2[t])

    e = (ys - zs) / (ys + zs)

    plt.plot(xs, e)
plt.show()
