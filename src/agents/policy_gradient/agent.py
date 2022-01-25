from src.agents.actors import FeedForwardActor
from src.agents.agent import Agent
from src.agents.policy_gradient.learning import PGLearner


class PGAgent(Agent):
    """PG agent.
    This class implements a simple policy gradient agent.
    The agent interacts with the environment generating full episodes of experiences via
    a behavior policy. The episodes are stored in an `EpisodeBuffer` object and passed to
    the learner to update the policy network.
    """

    def __init__(self, policy_network, buffer,
                 learning_rate=1e-4, lr_decay=1.0, reg=0.0, clip_grad=10.0):
        """Initialize a PG Agent instance.

        Args:
            policy_network (core.Network): A network object used as a behavior policy.
            buffer (core.Buffer): A buffer object used to store episodes of experiences.
            learning_rate (float, optional): Learning rate parameter. Default value is 1e-4.
            lr_decay (float, optional): Learning rate decay parameter. Default value is 1.0.
            reg (float, optional): L2 regularization strength. Default values is 0.0.
            clip_grad (float, optional): Gradient clipping parameter. Default value is 10.0.
        """
        optim = {
            "learning_rate":learning_rate,
            "lr_decay":lr_decay,
            "reg":reg,
            "clip_grad":clip_grad,
        }
        self._buffer = buffer
        self._learner = PGLearner(policy_network, optim)
        self._actor = FeedForwardActor(policy_network, self._buffer)
        self._min_observations = 0
        self._steps_per_observation = None

#