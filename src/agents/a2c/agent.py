import sys

import torch.nn.functional as F

from src.agents.actors import FeedForwardActor
from src.agents.agent import Agent
from src.agents.actor_critic.learning import ACLearner
from src.infrastructure.episode_buffer import EpisodeBuffer
from src.infrastructure.replay_buffer_ac import NonReplayBuffer

class ACAgent(Agent):
    """AC agent.
    This class implements a simple actor-critic agent.
    The agent interacts with the environment generating steps of experiences via a
    behavior policy. The observation steps are stored in a `ReplayBuffer` object and are
    passed to the learner to update the policy and value networks.

    Attributes:
        actor (core.Actor): An actor object used to interact with an environment.
        learner (core.Learner): A learner object used for updating the policy of the actor.
        buffer (core.Buffer): A buffer object used for storing observations made from the
            actor.
    """

    def __init__(self, policy_network, value_network, discount=0.9, observations_per_step=1,
                 policy_lr=1e-4, policy_lr_decay=1.0, policy_decay_steps=1, policy_reg=0.0,
                 value_lr=1e-4, value_lr_decay=1.0, value_decay_steps=1, value_reg=0.0,
                 batch_size=1, clip_grad=None, stdout=sys.stdout):
        """Initialize an AC Agent instance.

        Args:
            policy_network (Network): A network object used as a behavior policy.
            value_network (Network): A network object used to evaluate environment states.
            discount (float, optional): Discount factor for future rewards.
                Default values is 0.9.
            observations_per_step (int, optional): Number of observations the agent has to
                make before performing one learning step. Default value is 1.
            policy_lr (float, optional): Learning rate parameter for the policy network.
                Default value is 1e-4.
            policy_lr_decay (float, optional): Learning rate decay parameter for the policy
                network. Default value is 1.0.
            policy_decay_steps (int, optional): Every `policy_decay_steps` decay the learning
                rate of the policy network by `policy_lr_decay`. Default value is 1.
            policy_reg (float, optional): L2 regularization strength for the policy network.
                Default values is 0.0.
            value_lr (float, optional): Learning rate parameter for the value network.
                Default value is 1e-4.
            value_lr_decay (float, optional): Learning rate decay parameter for the value
                network. Default value is 1.0.
            value_decay_steps (int, optional): Every `value_decay_steps` decay the learning
                rate of the value network by `value_lr_decay`. Default value is 1.
            value_reg (float, optional): L2 regularization strength for the value network.
                Default values is 0.0.
            batch_size (int, optional): Batch size parameter used for gradient updates of
                the value network. Default value is 1.
            clip_grad (float, optional): Gradient clipping parameter. Default value is None.
        """
        super().__init__()
        config = {
            "discount"              : discount,
            "policy_lr"             : policy_lr,
            "policy_lr_decay"       : policy_lr_decay,
            "policy_decay_steps"    : policy_decay_steps,
            "policy_reg"            : policy_reg,
            "value_lr"              : value_lr,
            "value_lr_decay"        : value_lr_decay,
            "value_decay_steps"     : value_decay_steps,
            "value_reg"             : value_reg,
            "batch_size"            : batch_size,
            "clip_grad"             : clip_grad,
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
        # self.buffer = EpisodeBuffer()
        self.buffer = NonReplayBuffer()
        self.actor = FeedForwardActor(policy, self.buffer, device)
        self.learner = ACLearner(policy_network, value_network, config, stdout)
        self.observations_per_step = observations_per_step

#