# import gym_minigrid
# import matplotlib.pyplot as plt
# import numpy as np

# xs = np.linspace(0, 5)
# ys = 1 / np.square(xs + 1) - 1 / (xs + 1)


# plt.plot(xs, ys)
# plt.show()


import numpy as np

a = np.array([0, 0, 0, 0, 1, 0])
r = np.cumsum(a[::-1], 0)[::-1]
print(r)
