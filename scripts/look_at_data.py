import matplotlib.pyplot as plt
from offline_rl.utils.dataset.synthetic import cross, curve, linear, step

x, y = cross(10000)
plt.plot(x, y, ".")
plt.show()
