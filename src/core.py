import abc
from collections import namedtuple

import torch


"""Returned with every call to `step` and `reset` on an environment.
A `TimeStep` contains the data emitted by an environment at each step of interaction.
A `TimeStep` holds a an `observation` (np.Array), and associated `reward`, `done` and `info`.
"""
TimeStep = namedtuple('TimeStep', ['observation', 'reward', 'done', 'info'])

ReplayExperience = namedtuple('ReplayExperience',
                               ['current', 'action', 'reward', 'next', 'done'])


class Environment(abc.ABC):
    """Abstract base class for Python RL environments.
    Observations are described with `np.Array` objects.
    """

    @abc.abstractmethod
    def step(self, actID):
        """Updates the environment according to the action and returns a `TimeStep`.

        Args:
            actID (int): The index of the action selected by the agent.

        Returns:
            timestep (core.TimeStep): A namedtuple containing:
                observation (np.Array): A numpy array representing the observable state of
                    the environment.
                reward (float): The immediate reward received by the agent on transitioning
                    into the new state.
                done (bool): A boolean value indicating whether the episode has finished.
                info (dict}: A json style dict with relevant information.
        """

    @abc.abstractmethod
    def reset(self):
        """Resets the environment to the initial state.

        Returns:
            timestep (core.TimeStep): A namedtuple containing:
                observation (np.Array): A numpy array representing the observable initial
                    state of the environment.
                reward (float): 0.
                done (bool): False.
                info (dict}: A json style dict with relevant information.
        """

    @abc.abstractmethod
    def actions(self):
        """Return a list with the ids of the legal actions for the current state."""

    @abc.abstractmethod
    def num_actions(self):
        """The total number of possible actions in the environment."""

    @abc.abstractmethod
    def shape(self):
        """The shape of the numpy array representing the observable state of the environment."""

    @abc.abstractmethod
    def close(self):
        """Stop the environment engine."""

class Actor(abc.ABC):
    """ Abstract actor object.
    This interface defines an API for an Actor. The actor uses a policy network to select
    actions and interact with the environment.
    The interface provides methods for observing the state of the environment at each step.
    """

    @abc.abstractmethod
    def select_action(self, observation, legal):
        """Select an action using some policy."""

    @abc.abstractmethod
    def observe_first(self, timestep):
        """Observe the first time-step from the agent-environment interaction loop."""

    @abc.abstractmethod
    def observe(self, action, timestep):
        """Observe a time-step from the agent-environment interaction loop."""

    def update(self):
        """This method may implement logic for updating the policy network of the actor.
        However, this is usually done using a `core.Learner` object.

        # TODO:
        # Fetch the updated weights of the policy network after the learner has updated
        # them. The learner will be updating the weights asynchronously.
        """


class Learner(abc.ABC):
    """Abstract learner object.
    This interface defines an API for a Lerner. The learner object implements a learning
    loop. A single step of learning should be implemented via the `step` method.

    All objects implementing this interface should be able to take in an external dataset
    as a parameter to the `step` method and run updates using data from this dataset.
    The dataset is provided by a `core.Buffer` object.

    # TODO:
    # Data will be read from this dataset asynchronously and this is primarily useful when
    # the dataset is filled by an external process.
    """

    @abc.abstractmethod
    def step(self, buffer):
        """Perform a single step of learning update."""

    def save(self):
        raise NotImplementedError("Method 'save' is not implemented.")

    def restore(self, state):
        raise NotImplementedError("Method 'restore' is not implemented.")


class Buffer(abc.ABC):
    """Abstract buffer object.
    This interface defines an API for a buffer. The buffer stores data from passed
    experiences.
    The interface provides methods for adding and drawing from the buffer.
    """

    @abc.abstractmethod
    def add_first(self, timestep):
        """Add an initial time-step to the buffer."""

    @abc.abstractmethod
    def add(self, action, timestep, is_last):
        """Add a time-step and the action selected by the agent to the buffer."""

    @abc.abstractmethod
    def draw(self, num_samples, device):
        """Draw experiences from the buffer and place them on device."""

    @abc.abstractmethod
    def flush(self):
        """Flush the buffer."""
#