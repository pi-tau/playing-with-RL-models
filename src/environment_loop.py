class EnvironmentLoop:
    """A simple RL environment loop.
    This takes `Environment` and `Agent` instances and coordinates their interaction.
    The agent is updated at every step of the feedback loop if `should_update=True`.

    This can be used as:
        loop = EnvironmentLoop(environment, agent, should_update=True)
        loop.run(num_episodes, steps)
    """

    def __init__(self, agent, environment, should_update=True):
        """Initialize an environment loop object.

        Args:
            agent (Agent): An agent object.
            environment (core.Environment): An environment object.
            should_update (bool, optional): If True, update the agent at every step of the
                feedback loop. Default value is True.
        """
        self._agent = agent
        self._environment = environment
        self._should_update = should_update

    def run(self, episodes, steps=1_000_000):
        """Run the agent-environment feedback loop.

        Args:
            episodes (int): Number of episodes to run.
            steps (int, optional): Maximum number of steps for each episode.
                Default value is 1_000_000.
        """
        for _ in range(episodes):
            # At the begining of each episode reset the environment and observe the
            # initial state.
            timestep = self._environment.reset()
            self._agent.observe_first(timestep)

            for i in range(steps):
                # Select an action for the agent to perform.
                legal = self._environment.actions()
                illegal = list(set(range(self._environment.num_actions())) - set(legal))
                action = self._agent.select_action(timestep.observation, illegal)

                # Perform the action selected by the agent.
                timestep = self._environment.step(action)
                is_last = (timestep.done or i == (steps-1))

                # Observe the next timestep from the environment.
                self._agent.observe(action, timestep, is_last)

                # Maybe update the agent policy network.
                if self._should_update:
                    self._agent.update()

                # If the episode is finished break the loop.
                if timestep.done:
                    break

#