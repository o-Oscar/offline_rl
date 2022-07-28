import gym
import gym_minigrid
from gym_minigrid.wrappers import *


class CustomWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces["image"]

    def observation(self, obs):
        to_return = np.zeros((self.env.width, self.env.height, 6))
        to_return[self.env.agent_pos[0], self.env.agent_pos[1], self.env.agent_dir] = 1

        walls = np.where(obs["image"][:, :, 0] == 2)
        for wall in zip(*walls):
            to_return[wall + (4,)] = 1

        goals = np.where(obs["image"][:, :, 0] == 8)
        for goal in zip(*goals):
            to_return[goal + (5,)] = 1

        return to_return


def one_hot_to_map(onehot: np.ndarray):
    to_print = ""
    for col in onehot.transpose((1, 0, 2)):
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
        to_print += "\n"
    return to_print


# env = gym.make("MiniGrid-Empty-Random-5x5-v0")
env = gym.make("MiniGrid-Empty-8x8-v0")
env = FullyObsWrapper(env)
# env = ImgObsWrapper(env)
env = CustomWrapper(env)
res = env.reset()


for action in [2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2]:
    obs, rew, done, infos = env.step(action)

    print(one_hot_to_map(obs))
    print(rew, done, infos)
    print()


# for i in range(10):
#     res, _, _, _ = env.step(np.random.randint(3))
#     print(res[:, :, 0])
