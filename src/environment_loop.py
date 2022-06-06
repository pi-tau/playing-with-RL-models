import sys
from tqdm import tqdm

class EnvironmentLoop:
    """A simple RL environment loop.
    This takes `Environment` and `Actor` instances and coordinates their interaction.
    The actor is updated at every step of the feedback loop if `should_update=True`.

    This can be used as:
        `loop = EnvironmentLoop(environment, actor, should_update=True)`
        `loop.run(num_episodes, steps)`

    Attributes:
        actor (core.Actor): An actor object used to interact with the environment.
        environment (core.Environment): An environment object.
        run_history (dict): A dictionary storing information containing:
            returns (list(float)): A list of the returns history.
            running_return (list(float)): A list of the running returns history.
            nsteps (list(float)): A list of the number of episode steps history.
        stdout (file): File object (stream) used for standard output of logging information.
    """

    def __init__(self, actor, environment, should_update=True, stdout=sys.stdout):
        """Initialize an environment loop object.

        Args:
            actor (core.Actor): An actor object.
            environment (core.Environment): An environment object.
            should_update (bool, optional): If True, update the actor at every step of the
                feedback loop. Default value is True.
            stdout (file, optional): File object (stream) used for standard output of
                logging information. Default value is `sys.stdout`.
        """
        self.actor = actor
        self.environment = environment
        self.run_history = {}
        self.stdout = stdout
        self._should_update = should_update

    def should_update(self, should_update):
        self._should_update = should_update

    def run(self, episodes, steps=1_000_000, log_every=100, demo_every=None):
        """Run the actor-environment feedback loop.

        Args:
            episodes (int): Number of episodes to run.
            steps (int, optional): Maximum number of steps for each episode.
                Default value is 1_000_000.
            log_every (int, optional): Printout logging information every `log_every`
                episodes. Default value is 100.
            demo_every (int, optional): Run a graphical demo simulation every `demo_every`
                episodes. If `demo_every` is None, then no graphical simulations are run.
                Default value is None.
        """
        stdout = self.stdout
        self.run_history = {
            "returns": [],
            "running_return": [],
            "nsteps": [],
        }

        running_return = None
        total_return = 0.0
        best_return = -float("inf")
        total_steps = 0
        for e in tqdm(range(episodes)):
            verbose = (e+1) % log_every == 0

            # At the beginning of each episode reset the environment and observe the
            # initial state.
            timestep = self.environment.reset()
            self.actor.observe_first(timestep)
            episode_return = 0.

            for i in range(steps):
                # Select an action for the actor to perform.
                legal = self.environment.actions()
                action = self.actor.select_action(timestep.observation, legal)

                # Perform the action selected by the actor.
                timestep = self.environment.step(action)
                is_last = (timestep.done or i == (steps-1))
                episode_return += timestep.reward

                # Observe the next timestep from the environment.
                self.actor.observe(action, timestep, is_last)

                # Maybe update the actor policy network.
                if self._should_update:
                    self.actor.update(verbose)

                # If the episode is finished break the loop.
                if timestep.done:
                    break

            # Keep track of the running return, the total return, the best return and the
            # total number of steps.
            if running_return is None: running_return = episode_return
            else: running_return = 0.99 * running_return + 0.01 * episode_return
            total_return += episode_return
            best_return = max(best_return, episode_return)
            total_steps += i

            # Bookkeeping.
            self.run_history["returns"].append(episode_return)
            self.run_history["running_return"].append(running_return)
            self.run_history["nsteps"].append(i)

            # Printout logging information.
            if verbose:
                mean_return = total_return / log_every
                avg_steps = total_steps // log_every
                tqdm.write("Episode ({}/{}); Running/Mean/Best return: {:.2f}/{:.2f}/{:.2f}; Avg steps: {}".format(
                    e+1, episodes, running_return, mean_return, best_return, avg_steps), file=stdout)
                total_return = 0.0
                best_return = -float("inf")
                total_steps = 0

            # Maybe play a demo.
            if demo_every is not None and (e + 1) % demo_every == 0:
                self.environment.graphics(True)
                timestep = self.environment.reset()
                for _ in range(steps):
                    legal = self.environment.actions()
                    action = self.actor.select_action(timestep.observation, legal)
                    timestep = self.environment.step(action)
                    if timestep.done:
                        break
                self.environment.graphics(False)

        self.environment.close()

#