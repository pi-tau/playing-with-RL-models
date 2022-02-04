from tqdm import tqdm

class EnvironmentLoop:
    """A simple RL environment loop.
    This takes `Environment` and `Actor` instances and coordinates their interaction.
    The actor is updated at every step of the feedback loop if `should_update=True`.

    This can be used as:
        loop = EnvironmentLoop(environment, actor, should_update=True)
        loop.run(num_episodes, steps)
    """

    def __init__(self, actor, environment, should_update=True):
        """Initialize an environment loop object.

        Args:
            actor (core.Actor): An actor object.
            environment (core.Environment): An environment object.
            should_update (bool, optional): If True, update the actor at every step of the
                feedback loop. Default value is True.
        """
        self._actor = actor
        self._environment = environment
        self._should_update = should_update

    def should_update(self, should_update):
        self._should_update = should_update

    def run(self, episodes, steps=1_000_000):
        """Run the actor-environment feedback loop.

        Args:
            episodes (int): Number of episodes to run.
            steps (int, optional): Maximum number of steps for each episode.
                Default value is 1_000_000.
        """
        running_return = 0.0
        for e in tqdm(range(episodes)):
            # At the begining of each episode reset the environment and observe the
            # initial state.
            timestep = self._environment.reset()
            self._actor.observe_first(timestep)
            episode_return = 0.

            for i in range(steps):
                # Select an action for the actor to perform.
                legal = self._environment.actions()
                action = self._actor.select_action(timestep.observation, legal)

                # Perform the action selected by the actor.
                timestep = self._environment.step(action)
                is_last = (timestep.done or i == (steps-1))
                episode_return += timestep.reward

                # Observe the next timestep from the environment.
                self._actor.observe(action, timestep, is_last)

                # Maybe update the actor policy network.
                if self._should_update:
                    self._actor.update()

                # If the episode is finished break the loop.
                if timestep.done:
                    break

            # Keep track of the running return.
            if running_return is None: running_return = episode_return
            else: running_return = 0.99 * running_return + 0.01 * episode_return

            # Printout logging information.
            if e == 0 or (e + 1) % 100 == 0:
                tqdm.write(f"Episode ({e+1}/{episodes}); Running return: {running_return:.3f}")

        self._environment.close()

#