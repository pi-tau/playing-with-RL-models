import sys
sys.path.append("../..")

import numpy as np

from src import core
from src.envs.qubits.rdm_environment import QubitsEnvironment

class Environment(core.Environment):
    """An RL environment for a quantum system."""

    def __init__(self, num_qubits=2, epsi=1e-3):
        """Initialize an environment object."""
        self._system = QubitsEnvironment(num_qubits, epsi=epsi, batch_size=1)
        self.shape = self._observe(self._system._states).shape

    def reset(self):
        """Resets the environment to the initial state.

        Returns:
            timestep (core.TimeStep): A namedtuple containing:
                observation (np.Array): A numpy array representing the observable initial
                    state of the environment.
                reward (float): 0.
                done (bool): False.
                info (dict}: {}.
        """
        self._system.set_random_states()
        return core.TimeStep(self._observe(self._system._states), 0, False, {})

    def actions(self):
        """Return a list with the ids of the legal actions for the current state."""
        return None

    def num_actions(self):
        """The total number of actions in the environment."""
        return self._system.num_actions

    def observable_shape(self):
        """The shape of the numpy array representing the observable state of the environment."""
        return self.shape

    def step(self, actID):
        next_state, reward, done = self._system.step([actID])
        info = {}
        return core.TimeStep(self._observe(next_state), reward[0], done[0], info)

    def _observe(self, state):
        """Constructs a numpy array representing the observable state of the environment.

        Args:
            state (system.state): The game state to be observed.

        Returns:
            observable (np.Array): A 1D numpy array of shape (size,). The size of the
                array depends on the size of the quantum system.
        """
        obs = state.reshape(-1)
        obs =  np.hstack([obs.real, obs.imag])
        return obs.astype(np.float32)

#