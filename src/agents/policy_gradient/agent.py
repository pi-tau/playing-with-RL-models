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

    Attributes:
        actor (core.Actor): An actor object used to interact with an environment.
        learner (core.Learner): A learner object used for updating the policy of the actor.
        buffer (core.Buffer): A buffer object used for storing observations made from the
            actor.
    """

    def __init__(self, policy_network, buffer, use_reward_to_go=True, discount=0.9,
                 batch_size=1, learning_rate=1e-4, lr_decay=1.0, decay_steps=1, reg=0.0,
                 clip_grad=None, stdout=sys.stdout):
        """Initialize a PG Agent instance.

        Args:
            policy_network (core.Network): A network object used as a behavior policy.
            buffer (core.Buffer): A buffer object used to store episodes of experiences.
            use_reward_to_go (bool, optional): If True the learning step uses cumulative
                returns. If False, the learning step uses total baselined returns.
                Default value is True.
            discount (float, optional): Discount factor for future rewards.
                Default values is 0.9.
            batch_size (int, optional): The number of episodes to be run before performing
                one gradient update step. This variable gives the number of different
                episode trajectories used to approximate the value of the policy gradient.
                Default value is 1.
            learning_rate (float, optional): Learning rate parameter. Default value is 1e-4.
            lr_decay (float, optional): Learning rate decay parameter. Default value is 1.0.
            reg (float, optional): L2 regularization strength. Default values is 0.0.
            clip_grad (float, optional): Gradient clipping parameter. Default value is None.
        """
        config = {
            "use_reward_to_go"  : use_reward_to_go,
            "discount"          : discount,
            "learning_rate"     : learning_rate,
            "lr_decay"          : lr_decay,
            "decay_steps"       : decay_steps,
            "reg"               : reg,
            "clip_grad"         : clip_grad,
        }
        # The policy for the actor uses the scores returned by the network to compute a
        # boltzmann probability distribution over the actions from the action space.
        def policy(observation, legal):
            logits = policy_network(observation)
            if legal is not None:
                num_actions = policy_network.output_layer.out_features
                illegal = list(set(range(num_actions)) - set(legal))
                logits[illegal] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            return probs

        device = policy_network.device
        self.actor = FeedForwardActor(policy, buffer, device)
        self.learner = PGLearner(policy_network, config, stdout)
        self.buffer = buffer

        # Prefill the buffer with `batch_size` different episodes simulated using the current
        # policy and only then perform one-update step.
        self.min_observations = batch_size
        self.observations_per_step = None
        self._num_observations = 0

#