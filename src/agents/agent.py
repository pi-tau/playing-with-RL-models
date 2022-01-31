from src import core

class Agent(core.Actor):
    """Agent class which combines acting and learning.
    This provides an implementation of the `Agent` interface which acts and learns.
    It takes as input instances of both `core.Actor` and `core.Learner` classes, and
    implements the policy, observation, and update methods which
    defer to the underlying actor and learner.

    The only real logic implemented by this class is that it controls the number of
    observations to make before running a learner step. This is done by passing the number
    of `min_observations` to use and a ratio of
    `observations_per_step` := num_actor_actions / num_learner_steps.
    Note that the number of `observations_per_step` can also be in the range[0, 1]
    in order to allow the agent to take more than 1 learner step per action.

    Attributes:
        _actor (core.Actor): An actor object used to interact with the environment.
        _learner (core.Learner): A learner object used to update the policy network.
        _buffer (core.Buffer): A buffer object used to store past experiences.
        _min_observations (int):
        _steps_per_observation (int):
    """

    def __init__(self, actor, learner, buffer, min_observations, steps_per_observation):
        self._buffer = buffer
        self._learner = learner
        self._actor = actor
        self._min_observations = min_observations
        self._steps_per_observation = steps_per_observation

    def select_action(self, observation, illegal=None):
        """The agent selects an action by delegating to the actor."""
        return self._actor.select_action(observation, illegal)

    def observe_first(self, timestep):
        """The agent observes the first time-step by delegating to the actor."""
        self._actor.observe_first(timestep)

    def observe(self, action, timestep, is_last=False):
        """The agent observes time-steps by delegating to the actor."""
        self._actor.observe(action, timestep, is_last)

    def update(self):
        """Update the policy network by calling the `core.Learner.step()` method multiple
        times. The number of update steps made depends on the number of experiences stored
        in the buffer.
        """
        num_steps = self._calc_num_steps()
        for _ in range(num_steps):
            self._learner.step(self._buffer)
        # TODO: Asynchronous update of policy network weights.
        # if num_steps > 0:
        #     self._actor.update()

    def _calc_num_steps(self):
        num_observations = len(self._buffer)
        if num_observations < self._min_observations:
            return 0
        if self._steps_per_observation == None:
            return 1
        return int(num_observations * self._steps_per_observation)

#