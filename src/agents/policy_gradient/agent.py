import sys

import torch.nn.functional as F

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

    def __init__(self, policy_network, buffer, use_baseline=True, discount=0.9, learning_rate=1e-4,
                 lr_decay=1.0, reg=0.0, clip_grad=None, stdout=sys.stdout):
        """Initialize a PG Agent instance.

        Args:
            policy_network (core.Network): A network object used as a behavior policy.
            buffer (core.Buffer): A buffer object used to store episodes of experiences.
            use_baseline (bool, optional): If True the learning step uses baselined returns.
                If False, the learning step uses cumulative returns. Default value is True.
            discount (float, optional): Discount factor for future rewards.
                Default values is 0.9.
            learning_rate (float, optional): Learning rate parameter. Default value is 1e-4.
            lr_decay (float, optional): Learning rate decay parameter. Default value is 1.0.
            reg (float, optional): L2 regularization strength. Default values is 0.0.
            clip_grad (float, optional): Gradient clipping parameter. Default value is None.
        """
        config = {
            "use_baseline"  : use_baseline,
            "discount"      : discount,
            "learning_rate" : learning_rate,
            "lr_decay"      : lr_decay,
            "reg"           : reg,
            "clip_grad"     : clip_grad,
        }
        # The policy for the agent uses the scores returned by the network to compute a
        # boltzmann probability distribution over the actions from the action space. 
        policy = lambda obs: F.softmax(policy_network(obs), dim=-1)
        device = policy_network.device
        self._actor = FeedForwardActor(policy, buffer, device)
        self._learner = PGLearner(policy_network, config, stdout)
        self._buffer = buffer
        self._min_observations = 0
        self._steps_per_observation = None

#