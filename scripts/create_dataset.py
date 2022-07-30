import json
from pathlib import Path

import gym
import gym.spaces
import gym_minigrid
import numpy as np
from gym_minigrid.wrappers import *
from offline_rl.dataset import one_hot_to_string

# from tqdm import tqdm


class CustomWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.shape = (self.env.width, self.env.height, 6)
        self.observation_space = gym.spaces.Box(
            np.zeros(self.shape), np.ones(self.shape)
        )

    def observation(self, obs):
        to_return = np.zeros(self.shape)
        to_return[self.env.agent_pos[0], self.env.agent_pos[1], self.env.agent_dir] = 1

        walls = np.where(obs["image"][:, :, 0] == 2)
        for wall in zip(*walls):
            to_return[wall + (4,)] = 1

        goals = np.where(obs["image"][:, :, 0] == 8)
        for goal in zip(*goals):
            to_return[goal + (5,)] = 1

        return to_return


# env_name = "MiniGrid-Empty-Random-5x5-v0"
env_name = "MiniGrid-Empty-8x8-v0"

save_path = Path("results/dataset/dataset_0")
save_path.mkdir(parents=True, exist_ok=True)
metadata = {"env_name": env_name}
with open(save_path / "metadata.json", "w") as f:
    json.dump(metadata, f)


env = gym.make(env_name)
env = FullyObsWrapper(env)
# env = ImgObsWrapper(env)
env = CustomWrapper(env)


def rollout(env, max_steps):
    obs = env.reset()

    all_states = [obs]
    all_actions = []
    all_rewards = []
    for step in range(max_steps):
        action = np.random.randint(3)
        obs, rew, done, infos = env.step(action)

        all_states.append(obs)
        all_actions.append(action)
        all_rewards.append(rew)

        if done:
            return np.array(all_states), np.array(all_actions), np.array(all_rewards)

    return np.array(all_states), np.array(all_actions), np.array(all_rewards)


rollout_nb = 100

all_lengths = np.zeros([rollout_nb], dtype=np.int32)
all_states = np.zeros(
    (rollout_nb, env.max_steps + 1) + env.observation_space.shape, dtype=np.int32
)
all_actions = np.zeros((rollout_nb, env.max_steps), dtype=np.int32)
all_rewards = np.zeros((rollout_nb, env.max_steps), dtype=np.float32)

for rollout_id in range(rollout_nb):
    # for rollout_id in tqdm(range(rollout_nb)):
    states, actions, rewards = rollout(env, env.max_steps)
    rollout_len = actions.shape[0]

    all_lengths[rollout_id] = rollout_len
    all_states[rollout_id, : rollout_len + 1] = states
    all_actions[rollout_id, :rollout_len] = actions
    all_rewards[rollout_id, :rollout_len] = rewards

np.save(save_path / "all_lengths.npy", all_lengths)
np.save(save_path / "all_states.npy", all_states)
np.save(save_path / "all_actions.npy", all_actions)
np.save(save_path / "all_rewards.npy", all_rewards)
