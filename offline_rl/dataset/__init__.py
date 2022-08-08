import dataclasses
import json
from pathlib import Path

import numpy as np


def one_hot_to_string(onehot: np.ndarray):
    to_print = ""
    for col in onehot.transpose((2, 1, 0)):
        for cell in col:
            if cell[0] == 1:
                to_print += ">"
            elif cell[1] == 1:
                to_print += "V"
            elif cell[2] == 1:
                to_print += "<"
            elif cell[3] == 1:
                to_print += "^"
            elif cell[4] == 1:
                to_print += "W"
            elif cell[5] == 1:
                to_print += "G"
            else:
                to_print += " "
            to_print += " "

        to_print += "\n"
    return to_print[:-1]


@dataclasses.dataclass
class Dataset:
    metadata: dict
    all_lengths: np.ndarray
    all_states: np.ndarray
    all_actions: np.ndarray
    all_rewards: np.ndarray

    def plot_rollout(self, rollout_idx: int) -> None:
        for state, action, reward in zip(
            self.all_states[rollout_idx, : self.all_lengths[rollout_idx]],
            self.all_actions[rollout_idx],
            self.all_rewards[rollout_idx],
        ):
            print(one_hot_to_string(state))
            print("action : ", action)
            print("reward : ", reward)
            print()
        # todo : print the last state
        print(
            one_hot_to_string(
                self.all_states[rollout_idx, self.all_lengths[rollout_idx]]
            )
        )
        print("rollout len :", self.all_lengths[rollout_idx])

    def __str__(self) -> str:
        to_return = ""
        to_return += str(self.metadata)
        to_return += "   number of rollouts : " + str(self.all_lengths.shape[0])
        to_return += "   states shape : " + str(self.all_states.shape[0])
        to_return += "   actions shape : " + str(self.all_actions.shape[0])
        to_return += "   rewards shape : " + str(self.all_rewards.shape[0])
        return to_return

    def __len__(self):
        return np.sum(self.all_lengths)

    def generate_rollout_reward_dataset(self):
        dataset_states = np.zeros(
            (len(self),) + self.all_states.shape[2:], dtype=np.float32
        )
        dataset_cum_rewards = np.zeros((len(self),), dtype=np.float32)
        dataset_len = 0
        for rollout_id in range(self.all_states.shape[0]):
            rollout_len = self.all_lengths[rollout_id]
            rollout_states = self.all_states[rollout_id]
            rollout_rewards = self.all_rewards[rollout_id, :rollout_len]

            lid = dataset_len
            hid = dataset_len + rollout_len
            dataset_states[lid:hid] = rollout_states[:rollout_len]
            dataset_cum_rewards[lid:hid] = np.cumsum(rollout_rewards[::-1], 0)[::-1]

            dataset_len += rollout_len
        return dataset_states, dataset_cum_rewards.reshape((-1, 1))

    def generate_all_states_dataset(self):
        n_states = np.sum(self.all_lengths + 1)
        dataset_states = np.zeros(
            (n_states,) + self.all_states.shape[2:], dtype=np.float32
        )
        dataset_len = 0
        for rollout_id in range(self.all_states.shape[0]):
            rollout_len = self.all_lengths[rollout_id]
            rollout_states = self.all_states[rollout_id]

            lid = dataset_len
            hid = dataset_len + rollout_len + 1
            dataset_states[lid:hid] = rollout_states[: rollout_len + 1]

            dataset_len += rollout_len
        return dataset_states


def load_dataset(save_path: Path):
    with open(save_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    all_lengths = np.load(save_path / "all_lengths.npy")
    all_states = np.load(save_path / "all_states.npy")
    all_actions = np.load(save_path / "all_actions.npy")
    all_rewards = np.load(save_path / "all_rewards.npy")

    return Dataset(metadata, all_lengths, all_states, all_actions, all_rewards)
