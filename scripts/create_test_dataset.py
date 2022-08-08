import itertools
import json
from pathlib import Path

import gym
import gym.spaces
import gym_minigrid
import numpy as np
from gym_minigrid.wrappers import *
from offline_rl.dataset import one_hot_to_string

# env_name = "MiniGrid-Empty-Random-5x5-v0"
env_name = "MiniGrid-Empty-8x8-v0"

metadata = {"env_name": env_name}


def create_empty_dataset(n_states):

    rollout_nb = 1
    obs_shape = (7, 8, 8)

    all_lengths = np.zeros([rollout_nb], dtype=np.int32) + (n_states - 1)
    all_states = np.zeros((rollout_nb, n_states) + obs_shape, dtype=np.int32)
    all_actions = np.zeros((rollout_nb, n_states - 1), dtype=np.int32)
    all_rewards = np.zeros((rollout_nb, n_states - 1), dtype=np.float32)

    # adding the walls
    all_states[:, :, 4, 0, :] = 1
    all_states[:, :, 4, -1, :] = 1
    all_states[:, :, 4, :, 0] = 1
    all_states[:, :, 4, :, -1] = 1

    # adding the goal
    all_states[:, :, 5, -2, -2] = 1

    return all_lengths, all_states, all_actions, all_rewards


# -> datasets with the agent occupying only one position with the four dirs
save_path = Path("results/dataset/test_dataset_0")
save_path.mkdir(parents=True, exist_ok=True)
with open(save_path / "metadata.json", "w") as f:
    json.dump(metadata, f)

all_lengths, all_states, all_actions, all_rewards = create_empty_dataset(4)

all_states[0, 0, 0, 1, 1] = 1
all_states[0, 1, 1, 1, 1] = 1
all_states[0, 2, 2, 1, 1] = 1
all_states[0, 3, 3, 1, 1] = 1

all_states[:, :, -1] = 1 - np.sum(all_states, axis=2)

np.save(save_path / "all_lengths.npy", all_lengths)
np.save(save_path / "all_states.npy", all_states)
np.save(save_path / "all_actions.npy", all_actions)
np.save(save_path / "all_rewards.npy", all_rewards)

# -> datasets with the agent occupying only the first four positions always facing the same dir
save_path = Path("results/dataset/test_dataset_1")
save_path.mkdir(parents=True, exist_ok=True)
with open(save_path / "metadata.json", "w") as f:
    json.dump(metadata, f)

all_lengths, all_states, all_actions, all_rewards = create_empty_dataset(4)

all_states[0, 0, 0, 1, 1] = 1
all_states[0, 1, 0, 2, 1] = 1
all_states[0, 2, 0, 1, 2] = 1
all_states[0, 3, 0, 2, 2] = 1

all_states[:, :, -1] = 1 - np.sum(all_states, axis=2)

np.save(save_path / "all_lengths.npy", all_lengths)
np.save(save_path / "all_states.npy", all_states)
np.save(save_path / "all_actions.npy", all_actions)
np.save(save_path / "all_rewards.npy", all_rewards)

# -> datasets with the agent occupying only the first four positions with all dirs
save_path = Path("results/dataset/test_dataset_2")
save_path.mkdir(parents=True, exist_ok=True)
with open(save_path / "metadata.json", "w") as f:
    json.dump(metadata, f)

all_lengths, all_states, all_actions, all_rewards = create_empty_dataset(16)

for dir in range(4):
    all_states[0, 0 + 4 * dir, dir, 1, 1] = 1
    all_states[0, 1 + 4 * dir, dir, 2, 1] = 1
    all_states[0, 2 + 4 * dir, dir, 1, 2] = 1
    all_states[0, 3 + 4 * dir, dir, 2, 2] = 1

all_states[:, :, -1] = 1 - np.sum(all_states, axis=2)

np.save(save_path / "all_lengths.npy", all_lengths)
np.save(save_path / "all_states.npy", all_states)
np.save(save_path / "all_actions.npy", all_actions)
np.save(save_path / "all_rewards.npy", all_rewards)

# -> datasets with the agent occupying only four positions always facing the same dir
save_path = Path("results/dataset/test_dataset_3")
save_path.mkdir(parents=True, exist_ok=True)
with open(save_path / "metadata.json", "w") as f:
    json.dump(metadata, f)

all_lengths, all_states, all_actions, all_rewards = create_empty_dataset(4)

for dir in range(1):
    all_states[0, 0 + 4 * dir, dir, 1, 2] = 1
    all_states[0, 1 + 4 * dir, dir, 2, 4] = 1
    all_states[0, 2 + 4 * dir, dir, 6, 3] = 1
    all_states[0, 3 + 4 * dir, dir, 5, 6] = 1

all_states[:, :, -1] = 1 - np.sum(all_states, axis=2)

np.save(save_path / "all_lengths.npy", all_lengths)
np.save(save_path / "all_states.npy", all_states)
np.save(save_path / "all_actions.npy", all_actions)
np.save(save_path / "all_rewards.npy", all_rewards)

# -> datasets with the agent occupying only four positions with all dirs
save_path = Path("results/dataset/test_dataset_4")
save_path.mkdir(parents=True, exist_ok=True)
with open(save_path / "metadata.json", "w") as f:
    json.dump(metadata, f)

all_lengths, all_states, all_actions, all_rewards = create_empty_dataset(16)

for dir in range(4):
    all_states[0, 0 + 4 * dir, dir, 1, 2] = 1
    all_states[0, 1 + 4 * dir, dir, 2, 4] = 1
    all_states[0, 2 + 4 * dir, dir, 6, 3] = 1
    all_states[0, 3 + 4 * dir, dir, 5, 6] = 1

all_states[:, :, -1] = 1 - np.sum(all_states, axis=2)

np.save(save_path / "all_lengths.npy", all_lengths)
np.save(save_path / "all_states.npy", all_states)
np.save(save_path / "all_actions.npy", all_actions)
np.save(save_path / "all_rewards.npy", all_rewards)

# -> datasets with the agent occupying any position with one direction
save_path = Path("results/dataset/test_dataset_5")
save_path.mkdir(parents=True, exist_ok=True)
with open(save_path / "metadata.json", "w") as f:
    json.dump(metadata, f)

all_lengths, all_states, all_actions, all_rewards = create_empty_dataset(36)

id = 0
for dir in range(1):
    for px, py in itertools.product(range(1, 7), range(1, 7)):
        all_states[0, id, dir, px, py] = 1
        all_states[0, id, dir, px, py] = 1
        all_states[0, id, dir, px, py] = 1
        all_states[0, id, dir, px, py] = 1
        id += 1

all_states[:, :, -1] = 1 - np.max(all_states, axis=2)

np.save(save_path / "all_lengths.npy", all_lengths)
np.save(save_path / "all_states.npy", all_states)
np.save(save_path / "all_actions.npy", all_actions)
np.save(save_path / "all_rewards.npy", all_rewards)

# -> datasets with the agent occupying any position with any direction
save_path = Path("results/dataset/test_dataset_6")
save_path.mkdir(parents=True, exist_ok=True)
with open(save_path / "metadata.json", "w") as f:
    json.dump(metadata, f)

all_lengths, all_states, all_actions, all_rewards = create_empty_dataset(36 * 4)

id = 0
for dir in range(4):
    for px, py in itertools.product(range(1, 7), range(1, 7)):
        all_states[0, id, dir, px, py] = 1
        all_states[0, id, dir, px, py] = 1
        all_states[0, id, dir, px, py] = 1
        all_states[0, id, dir, px, py] = 1
        id += 1

all_states[:, :, -1] = 1 - np.max(all_states, axis=2)

np.save(save_path / "all_lengths.npy", all_lengths)
np.save(save_path / "all_states.npy", all_states)
np.save(save_path / "all_actions.npy", all_actions)
np.save(save_path / "all_rewards.npy", all_rewards)
