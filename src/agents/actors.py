import random

import torch
import torch.nn.functional as F

from src import core


class FeedForwardActor(core.Actor):
    """A feed-forward actor.
    An actor based on a feed-forward policy which takes non-batched observations
    and outputs non-batched actions. It also allows adding experiences to a buffer.

    Attributes:
        _policy (func): A mapping that takes an observation from the environment and
            outputs a probability distribution over the action space.
        _buffer_client (core.Buffer): A client used to pass observations to the buffer.
        _device (torch.device): Determine which device to place observations upon, CPU or GPU.
    
    # TODO: The actor uses a client in order to add data asynchronously.
    """

    def __init__(self, policy, buffer_client, device):
        self._policy = policy
        self._buffer_client = buffer_client   # allows to send data to the replay buffer
        self._device = device

    @torch.no_grad()
    def select_action(self, observation, legal=None):
        """Return the action selected by the policy.
        Using the policy function compute a probability distribution over the action space.
        Select the next action by sampling from the probability distribution. If the
        sampled action is illegal, then output a random legal action.

        Args:
            observation (np.Array):A numpy array of shape (state_size,), giving the
                current state of the environment.
            legal (list[int]): A list of indices of the legal actions for the agent.

        Returns:
            act (int): The index of the action selected by the policy.
        """
        # Convert the observations to torch tensors and move them to device.
        observation = torch.from_numpy(observation).float()
        observation = observation.to(self._device)

        # Use the policy to compute a probability distribution over the actions and select
        # the next action probabilistically.        
        probs = self._policy(observation)
        action = torch.multinomial(probs, 1).squeeze(dim=-1).item()

        # If the selected action is illegal, then select a random legal action.
        if legal is not None and action not in legal:
            return random.choice(legal)
        return action

    def observe_first(self, timestep):
        """Observe a time-step from the agent-environment interaction loop."""
        self._buffer_client.add_first(timestep)

    def observe(self, action, timestep, is_last=False):
        """Observe a time-step from the agent-environment interaction loop."""
        self._buffer_client.add(action, timestep, is_last)

#