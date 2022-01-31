import numpy as np
import torch

from src import core


class EpisodeBuffer(core.Buffer):
    """Episode buffer.
    A buffer object used to store trajectories of entire episodes.

    Attributes:
        _observations (list[list[np.Array]]): A list of lists of numpy arrays. Each nested
            list stores the observed states during a single episode.
        _actions (list[list[int]]): A list of lists of ints. Each nested list stores the
            actions taken by the agent during a single episode.
        _rewards (list[list[float]]): A list of lists of floats. Each nested list stores
            the rewards obtained by the agent during a single episode.
        _done (list[list[bool]]):A list of lists of booleans. Each nested list stores
            boolean flags indicating wether the episode has finished.
        _current_observations (list[np.Array]): A list of numpy arrays storing the
            observations from the current episode.
        _current_actions (list[int]): A list of ints storing the actions taken by the
            agent during the current episode.
        _current_rewards (list[float]): A list of floats storing the rewards obtained by
            the agent during the current episode.
        _current_done (list[bool]): A list of booleans indicating whether the current
            episode has finished.
    """

    def __init__(self):
        self._observations = []
        self._actions = []
        self._rewards = []
        self._done = []

    def add_first(self, timestep):
        """Add an initial time-step to the buffer.
        Start a new episode beginning with the initial observation. Prepare containers
        for observations, actions and rewards for the current episode.
        """
        observation, reward, done, info = timestep

        self._current_observations = []
        self._current_actions = []
        self._current_rewards = []
        self._current_done = []
        self._current_observations.append(observation)

    def add(self, action, timestep, is_last=False):
        """Add a time-step and the action selected by the agent to the buffer.
        Append observations, actions and rewards experiences to the containers for the
        on-going episode. If this is the last time step, store the episode trajectory
        and empty the containers for the current episode.

        Args:
            action (int): The index of the action selected by the agent.
            timestep (core.TimeStep): A namedtuple containing:
                observation (np.Array): A numpy array representing the observable state of
                    the environment.
                reward (float): The immediate reward received by the agent on transitioning
                    into the new state.
                done (bool): A boolean value indicating whether the episode has finished.
                info (dict}: A json style dict with relevant information.
        """
        observation, reward, done, info = timestep
        self._current_observations.append(observation)
        self._current_actions.append(action)
        self._current_rewards.append(reward)
        self._current_done.append(done)

        if is_last:
            self._observations.append(self._current_observations[:-1])
            self._actions.append(self._current_actions)
            self._rewards.append(self._current_rewards)
            self._done.append(self._current_done)
            self._current_observations = None
            self._current_actions = None
            self._current_rewards = None
            self._current_done = None

    def draw(self, num_samples=None, device=torch.device("cpu")):
        """Draw a batch of episodes from the buffer and place them on device.
        Pad shorter episodes with 0s according to the longest episode in the batch.

        Args:
            num_samples (int, optional): Number of episodes in the batch. Default value
                is None, returning all episodes in the buffer.
            device (torch.device): Determine which device to place the batch upon, CPU or GPU.

        Returns:
            observations (torch.Tensor): A tensor of shape (b, t, size), giving the
                observations from the batch of episodes, where b = batch size,
                t = number of time steps, size = shape of the environment observable.
            actions (torch.Tensor): A tensor of shape (b, t), giving the actions selected
                by the agent during the interaction with the environment.
            rewards (torch.Tensor): A tensor of shape (b, t), giving the rewards obtained
                during by the agent.
            masks (torch.Tensor): A tensor of shape (b, t), of boolean values, that masks
                out the part of the episode after it has finished.
        """
        max_len_episode = max(len(episode) for episode in self._observations)
        padded_observations = np.array([[obs for obs in episode] + [np.zeros_like(episode[-1]).tolist()
            for _ in range(max_len_episode-len(episode))]
            for episode in self._observations])
        padded_actions = [[act for act in episode] + [0]*(max_len_episode-len(episode))
            for episode in self._actions]
        padded_rewards = [[r for r in episode] + [0]*(max_len_episode-len(episode))
            for episode in self._rewards]
        observations = torch.FloatTensor(padded_observations).to(device)
        actions = torch.LongTensor(padded_actions).to(device)
        rewards = torch.FloatTensor(padded_rewards).to(device)

        # if done[i] is False and done[i+1] is True, then the trajectory should be masked
        # out at and after step i+2.
        padded_done = [[d for d in episode] + [True]*(max_len_episode-len(episode))
            for episode in self._done]
        done = torch.BoolTensor(padded_done).to(device)
        masks = ~torch.cat((done[0:1], done[1:] & done[:-1]), dim=0)

        # Drawing from the `EpisodeBuffer` deletes the drawn experiences.
        # This type of buffer is used only with offline policy gradient agents.
        self.flush()

        return observations, actions, rewards, masks

    def flush(self):
        """Flush the buffer."""
        self._observations.clear()
        self._actions.clear()
        self._rewards.clear()
        self._done.clear()

    def __len__(self):
        return len(self._observations)

#