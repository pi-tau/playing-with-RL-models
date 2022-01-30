import numpy as np
from src.agents.agent import Agent
from src.infrastructure.learner_log import DQNLearnerLogger


class DQNAgent(Agent):

    def __init__(self, actor, learner, buffer,
                 min_experiences, update_every, n_steps):
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
            update_every (int):
                Controls how often the Q-function parameters are updated.
            n_steps (int):
                Controls how many learner steps happen on call to `self.update()`
        """
        # `actor`, `learner` and `buffer` arguments are binded to `self` here
        super().__init__(actor, learner, buffer)
        self.min_experiences = min_experiences
        self.update_every = update_every
        self.n_steps = n_steps
        self._n = 0  # counts new observations since last self.update() call
        self._total_experiences = 0
        self._logger = DQNLearnerLogger(self)

    def observe_first(self, timestep):
        self._actor.observe_first(timestep)

    def observe(self, action, timestep, is_last=False):
        # Add observation to Experience Replay buffer
        self._actor.observe(action, timestep, is_last)
        self._n += 1
        self._total_experiences += 1
        # Check if Experience Replay buffer contains enough samples
        # and that `self._fit_every` steps have past since last fit.
        if (len(self._buffer) > self.min_experiences and
            self._n >= self.update_every):
            self.update()

    def update(self):
        """ Updates Q-function parameters and the epsilon policy parameter. """
        self._learner.step(self._buffer, self.n_steps)
        x = np.round(self._total_experiences / 1_000_000, 1)
        self._actor.epsilon = 1.0 - min(0.9, x)
        self._actor.Qnetwork = self._learner.Qnetwork
        self._n = 0
        self._logger.add_mean_Q()
        self._logger.plot_mean_Q('figures/mean_Q_value.png')
