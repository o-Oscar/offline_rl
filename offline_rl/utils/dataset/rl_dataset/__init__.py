import dataclasses
import json
from pathlib import Path

import numpy as np
from offline_rl.utils.dataset.rl_dataset.formating import one_hot_to_string


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
        print(self.all_rewards[rollout_idx])
        print("rollout len :", self.all_lengths[rollout_idx])

    def __str__(self) -> str:
        to_return = ""
        to_return += str(self.metadata)
        to_return += "   number of rollouts : " + str(self.all_lengths.shape[0])
        to_return += "   states shape : " + str(self.all_states.shape[0])
        to_return += "   actions shape : " + str(self.all_actions.shape[0])
        to_return += "   rewards shape : " + str(self.all_rewards.shape[0])
        return to_return


def load_dataset(save_path: Path):
    with open(save_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    all_lengths = np.load(save_path / "all_lengths.npy")
    all_states = np.load(save_path / "all_states.npy")
    all_actions = np.load(save_path / "all_actions.npy")
    all_rewards = np.load(save_path / "all_rewards.npy")

    return Dataset(metadata, all_lengths, all_states, all_actions, all_rewards)
