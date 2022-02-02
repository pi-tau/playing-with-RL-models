import numpy as np
import torch

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
    def select_action(self, observation, illegal=[]):
        """Return the action selected by the policy.
        Using the policy function compute a probability distribution over the action space.
        Select the next action by sampling from the probability distribution.

        Args:
            observation (np.Array):A numpy array of shape (state_size,), giving the
                current state of the environment.
            illegal (list[int], optional): A list of indices of the illegal actions for
                the agent. Default value is [], meaning all actions are legal.

        Returns:
            act (int): The index of the action selected by the policy.
        """
        # Convert the observations to torch tensors and move them to device.
        observation = torch.from_numpy(observation).float()
        observation = observation.to(self._device)

        # Use the policy to compute a probability distribution over the actions and select
        # the next action probabilistically.
        probs = self._policy(observation, illegal)
        action = torch.multinomial(probs, 1).squeeze(dim=-1).item()
        return action

    def observe_first(self, timestep):
        """Observe a time-step from the agent-environment interaction loop."""
        self._buffer_client.add_first(timestep)

    def observe(self, action, timestep, is_last=False):
        """Observe a time-step from the agent-environment interaction loop."""
        self._buffer_client.add(action, timestep, is_last)


class DQNActor(core.Actor):

    def __init__(self, Qnetwork, buffer_client, epsilon=1.0, cache_limit=1):
        self._Qnetwork = Qnetwork.to(torch.device('cpu'))
        self.epsilon = epsilon
        self.buffer_client = buffer_client
        # self._buffer_cache = []
        # self._cache_limit = cache_limit
        self._last_timestep = None
        self._return = 0
        self._steps = 0

    @property
    def Qnetwork(self):
        return self._Qnetwork

    @Qnetwork.setter
    def Qnetwork(self, Qnet):
        self._Qnetwork = Qnet.to(torch.device('cpu'))

    def select_action(self, observation, legal):
        observation = np.expand_dims(observation, 0)
        observation = torch.from_numpy(observation).float()
        observation = observation.to(self.Qnetwork.device)

        with torch.no_grad():
            qvalues = self.Qnetwork(observation).cpu().numpy().squeeze()
            qvalues = qvalues[legal]
            argmax = np.argmax(qvalues)
        if np.random.uniform(0.0, 1.0) < self.epsilon:
            i = np.random.randint(0, len(legal))
            while i == argmax:
                i = np.random.randint(0, len(legal))
            return legal[i]
        else:
            return legal[argmax]

    def observe_first(self, timestep):
        self._last_timestep = timestep
        self._return = timestep.reward
        self._steps = 0

    def observe(self, action, timestep, is_last=False):
        replay_experience = DQNActor.make_replay_experience(
            self._last_timestep,
            action,
            timestep
        )
        self.buffer_client.add(replay_experience)
        self._last_timestep = timestep
        self._return += timestep.reward
        self._steps += 1
        # self._buffer_cache.append(replay_experience)
        # if len(self._buffer_cache) >= self._cache_limit:
        #     self.buffer_client.add(self._buffer_cache)
        #     self._buffer_cache.clear()

    @staticmethod
    def make_replay_experience(timestep0, action, timestep1):
        return core.ReplayExperience(
            timestep0.observation,    # current state
            action,                   # action
            timestep1.reward,         # reward
            timestep1.observation,    # next state
            timestep1.done            # done
        )
#