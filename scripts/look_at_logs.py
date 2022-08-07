from pathlib import Path

import matplotlib.pyplot as plt
from offline_rl.utils.logger import Logger

logs_path = Path("results/loggers/")
all_logger_path = list(logs_path.glob("*"))
all_logger_path = sorted(all_logger_path)[-2:]


all_loggers = []
for path in all_logger_path:
    all_loggers.append(Logger(path))


plt.figure()
for logger in all_loggers:
    logger.plot("full_loss")
plt.yscale("log")
plt.legend()

plt.figure()
for logger in all_loggers:
    logger.plot_wall("full_loss")
plt.yscale("log")
plt.legend()

plt.show()