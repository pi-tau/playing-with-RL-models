import numpy as np
from src.agents.agent import Agent
from src.infrastructure.logging import NullLogger


class DQNAgent(Agent):

    def __init__(self, actor, learner, buffer,
                 min_experiences, Q_update_every,
                 initial_eps, final_eps, eps_decay_range, logger=None):
        """
        Deep Q-Learning Agent with Experience Replay buffer.

        Attributes:
            _actor (core.Actor):
                Responsible for adding experince replays to `self._buffer`
            _learner (core.Learner):
                Holds and updates Q-function parameters.
            _buffer (core.Buffer):
                Experience Replay buffer. Shared with `self._actor`
            min_experiences (int):
                Minimum number of experiences in `self._buffer` required for
                update on Q-function parameters to fire.
            Q_update_every (int):
                Controls how often the Q-function parameters are updated.
        """
        # `actor`, `learner` and `buffer` arguments are binded to `self` here
        super().__init__(actor, learner, buffer)
        self._actor.epsilon = initial_eps
        self.min_experiences = min_experiences
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.eps_decay_range = eps_decay_range
        self.Q_update_every = Q_update_every
        self.n = 0  # counts new observations since last self.update() call
        self.total_experiences = 0
        self._logger = logger if logger is not None else NullLogger

    def observe(self, action, timestep, is_last=False):
        # Add observation to Experience Replay buffer
        self._actor.observe(action, timestep, is_last)
        self.n += 1
        self.total_experiences += 1
        if is_last:
            self._logger.add_return(self, self._actor._return)
            self._logger.add_episode_length(self, self._actor._steps)

    def update(self):
        """ Updates Q-function parameters and the epsilon policy parameter. """
        # Check if update is meaningfull
        if (len(self._buffer) < self.min_experiences or
            self.n < self.Q_update_every):
            return False
        # Update Q-network parameters
        self._learner.step(self._buffer)
        self._actor.Qnetwork = self._learner.Qnetwork
        self.n = 0
        # Update policy epsilon
        alpha = min(1.0, self.total_experiences / self.eps_decay_range)
        r = self.initial_eps - self.final_eps
        self._actor.epsilon = self.final_eps + alpha * r
        # Log Q-network stats
        self._logger.add_mean_Q(self)
        self._logger.add_buffer_capacity(self, len(self._buffer))
        return True
