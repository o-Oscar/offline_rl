import json
from curses import meta
from pathlib import Path

import gym
import gym_minigrid
import numpy as np
from offline_rl.utils.dataset.formating import one_hot_to_map


def load_dataset(save_path: Path):
    with open(save_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    all_lengths = np.load(save_path / "all_lengths.npy")
    all_states = np.load(save_path / "all_states.npy")
    all_actions = np.load(save_path / "all_actions.npy")
    all_rewards = np.load(save_path / "all_rewards.npy")

    return metadata, all_lengths, all_states, all_actions, all_rewards


save_path = Path("results/dataset/dataset_0")
metadata, all_lengths, all_states, all_actions, all_rewards = load_dataset(save_path)
env = gym.make(metadata["env_name"])

print(metadata)
print(all_lengths)
print(all_states.shape)
print(all_actions.shape)
print(all_rewards.shape)

for state, action in zip(all_states[0, : all_lengths[0] + 1], all_actions[0]):
    print(one_hot_to_map(state))
    print(action)
