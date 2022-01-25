import torch
import torch.nn.functional as F

from src import core


class FeedForwardActor(core.Actor):
    """A feed-forward actor.
    An actor based on a feed-forward policy which takes non-batched observations
    and outputs non-batched actions. It also allows adding experiences to a buffer.

    Attributes:
        _policy (core.Network): A network object representing the policy of the actor.
        _buffer_client (core.Buffer): A client used to pass observations to the buffer.
    
    # TODO: The actor uses a client in order to add data asynchronously.
    """

    def __init__(self, policy, buffer_client):
        self._policy = policy
        self._buffer_client = buffer_client   # allows to send data to the replay buffer

    @torch.no_grad()
    def select_action(self, observation, illegal=None):
        """Return the action selected by the policy.
        Using the scores returned by the network compute a boltzmann probability
        distribution over the actions from the action space. Select the next action
        probabilistically, or deterministically returning the action with the highest
        probability.

        Args:
            observation (np.Array):A numpy array of shape (state_size,), giving the
                current state of the environment.
            illegal (list[int]): A list of indices of the illegal actions for the agent.

        Returns:
            act (int): The index of the action selected by the policy.
        """
        # Compute the logits for each action by running the observation through the policy
        # network.
        observation = torch.from_numpy(observation).float()
        observation = observation.to(self._policy.device)
        logits = self._policy(observation)

        # Mask-out illegal actions.
        if illegal is not None:
            logits[illegal] = float("-inf")
        
        # Compute the probability distribution over the action set using a softmax on the
        # logits. Sample the next action from the probability distribution. 
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(dim=-1).item()

    def observe_first(self, timestep):
        """Observe a time-step from the agent-environment interaction loop."""
        self._buffer_client.add_first(timestep)

    def observe(self, action, timestep, is_last=False):
        """Observe a time-step from the agent-environment interaction loop."""
        self._buffer_client.add(action, timestep, is_last)

#