from pathlib import Path

import matplotlib.pyplot as plt
from offline_rl.utils.logger import Logger

logs_path = Path("results/loggers/")
all_logger_path = list(logs_path.glob("*"))
all_logger_path = sorted(all_logger_path)

# change here what you want to select
all_logger_path = [all_logger_path[0], all_logger_path[-1]]


all_loggers = []
for path in all_logger_path:
    all_loggers.append(Logger(path))
    all_loggers[-1].summerize()

logger: Logger

plt.figure()
for logger in all_loggers:
    logger.plot("full_loss", filter_sigma=100)
plt.yscale("log")
plt.legend()

plt.figure()
for logger in all_loggers:
    logger.plot_wall("full_loss", filter_sigma=100)
plt.yscale("log")
plt.legend()

plt.show()
