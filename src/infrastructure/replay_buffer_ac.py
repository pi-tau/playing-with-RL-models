import numpy as np
import torch

from src import core


class NonReplayBuffer(core.Buffer):
    """
    """

    def __init__(self):
        """
        """
        self._prev_obs = None
        self._container = []

    def add_first(self, timestep):
        observation, reward, done, info = timestep
        self._prev_obs = observation

    def add(self, action, timestep, is_last=False):
        observation, reward, done, info = timestep

        # TODO
        # This hack is done in order to make sure v(terminal) = 0 !
        if is_last:
            observation = np.zeros_like(observation)

        item = (self._prev_obs, action, reward, observation)
        self._container.append(item)
        self._prev_obs = observation

    def draw(self, num_samples=None, device=torch.device("cpu")):
        prev_obs, actions, rewards, next_obs = zip(*self._container)
        prev_obs = torch.FloatTensor(np.array(prev_obs)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_obs = torch.FloatTensor(np.array(next_obs)).to(device)
        self.flush()
        return prev_obs, actions, rewards, next_obs

    def flush(self):
        self._container.clear()

    def __len__(self):
        return len(self._container)

#