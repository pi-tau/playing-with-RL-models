import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src import core


class PGLearner(core.Learner):
    """PG learner.
    This class implements the learning logic for a policy gradient agent.
    The updates the policy network of the agent by performing single update steps
    using a dataset of full episodes produced by the agent.
    This is an offline and on-policy learning algorithm.

    Attributes:
        _policy_network (Network): The agent behavior policy to be optimized.
        _config (dict): A dictionary with optimization parameters (learning rate, etc.).
        _optimizer (torch.optim): A torch optimizer object used to perform gradient
            descent updates. The learner uses the Adam optimizer.
        _scheduler (torch.optim.lr_scheduler): A torch scheduler object used to schedule
            the value of the learning rate.
        _running_return (float): Keeping track of the running return from each episode
            during training.
        _stdout (file): File object (stream) used for standard output of logging
            information. Default value is `sys.stdout`.
    """

    def __init__(self, policy_network, config, stdout=sys.stdout):
        """Initialize a policy gradient learner object.

        Args:
            policy_network (Network): A network object used as a behavior policy.
            config (dict): A dictionary with configuration parameters containing:
                use_reward_to_go (bool) : If True the learning step uses cumulative
                    returns. If False, the learning step uses total baselined returns
                discount (float): Discount factor for future rewards.
                learning_rate (float): Learning rate parameter.
                lr_decay (float): Learning rate decay parameter.
                reg (float): L2 regularization strength.
                clip_grad (float): Parameter for gradient clipping by norm.
            stdout (file, optional): File object (stream) used for standard output of
                logging information. Default value is `sys.stdout`.
        """
        self._policy_network = policy_network
        self._config = config
        self._optimizer = torch.optim.Adam(
            self._policy_network.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["reg"],
        )
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, step_size=1, gamma=config["lr_decay"])
        self._running_return = None
        self._stdout = stdout

    def step(self, buffer, verbose=True):
        """Perform a single policy gradient update step.
        Draw a batch of episodes from the buffer and compute an estimate of the gradient
        of the policy. Subtract a baseline or use the `reward-to-go` trick to reduce the
        variance of the estimate.

        Args:
            buffer (core.Buffer): A buffer object used to store episodes of experiences.
            verbose (bool, optional): If True, printout logging information. Default value is True.
        """
        stdout = self._stdout
        device = self._policy_network.device
        discount = self._config["discount"]
        clip_grad = self._config["clip_grad"]
        eps = torch.finfo(torch.float32).eps

        # Fetch trajectories from the buffer.
        observations, actions, rewards, masks = buffer.draw(device=device)

        # Compute discounted baselined returns, or compute discounted returns-to-go.
        if self._config["use_reward_to_go"]:
            q_values = self._discounted_cumulative_returns(rewards, discount)
            # Baseline the discounted returns.
            # q_values -= torch.sum(q_values, dim=0) / torch.maximum(
            #     torch.sum(masks, dim=0), torch.Tensor([1.]).to(device))
            # Normalize the discounted returns.
            q_values = (q_values - torch.mean(q_values)) / (torch.std(q_values) + eps)
        else:
            q_values = self._discounted_baselined_returns(rewards, discount)


        # Compute the loss.
        logits = self._policy_network(observations)
        nll = F.cross_entropy(logits.permute(0,2,1), actions, reduction="none")
        weighted_nll = torch.mul(masks * nll, q_values)
        loss = torch.mean(torch.sum(weighted_nll, dim=1))

        # Perform backward pass.
        self._optimizer.zero_grad()
        loss.backward()
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad) for p in self._policy_network.parameters()]))
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self._policy_network.parameters(), clip_grad)
        self._optimizer.step()
        self._scheduler.step()

        # Keep track of the running return.
        mean_return = torch.mean(torch.sum(rewards, dim=-1))
        if self._running_return is None: self._running_return = mean_return
        else: self._running_return = 0.99*self._running_return + 0.01*mean_return

        if verbose:
            probs = F.softmax(logits, dim=-1)
            probs = torch.maximum(probs, torch.tensor(eps))
            avg_policy_ent = -torch.mean(torch.sum(probs*torch.log(probs), dim=-1))
            tqdm.write("#-------------------------------------------------#", file=stdout)
            tqdm.write(f"Mean return:        {torch.mean(torch.sum(rewards, dim=1)): .4f}", file=stdout)
            tqdm.write(f"Best return:        {max(torch.sum(rewards, dim=1)): .1f}", file=stdout)
            tqdm.write(f"Avg num of steps:   {torch.mean(torch.sum(masks, dim=1, dtype=float)): .0f}", file=stdout)
            tqdm.write(f"Longest episode:    {max(torch.sum(masks, dim=1, dtype=float)): .0f}", file=stdout)
            tqdm.write(f"Pseudo loss:        {loss.item(): .5f}", file=stdout)
            tqdm.write(f"Grad norm:          {total_norm: .5f}", file=stdout)
            tqdm.write(f"Avg policy entropy: {avg_policy_ent: .3f}", file=stdout)
            tqdm.write(f"Total num of steps: {torch.sum(masks): .0f}", file=stdout)
            tqdm.write(f"Running return:     {self._running_return:.4f}", file=stdout)

    @torch.no_grad()
    def _discounted_cumulative_returns(self, rewards, discount):
        """Compute the discounted cumulative reward-to-go at every time-step `t`.

        "Don't let the past destract you"
        Taking a gradient step pushes up the log-probabilities of each action in
        proportion to the sum of all rewards ever obtained. However, agents should only
        reinforce actions based on rewards obtained after they are taken.
        Check out: https://spinningup.openai.com/en/latest/spinningup/extra_pg_proof1.html

        Multiplying the rewards by a discount factor can be interpreted as encouraging the
        agent to focus more on the rewards that are closer in time. This can also be
        thought of as a means for reducing variance, because there is more variance
        possible when considering rewards that are further into the future.

        The cumulative return at time-step `t` is computed as the sum of all future
        rewards starting from the current time-step.
        The discounted cumulative returns for a batch of episodes can be computed as a
        matrix multiplication between the rewards matrix and a special toeplitz matrix.

        toeplitz = [1       0       0       0       ...     0       0       0]
                   [g       1       0       0       ...     0       0       0]
                   [g^2     g       1       0       ...     0       0       0]
                   [g^3     g^2     g       1       ...     0       0       0]
                   [...                                                      ]
                   [g^(n-2) g^(n-3) g^(n-4) g^(n-5) ...     g       1       0]
                   [g^(n-1) g^(n-2) g^(n-3) g^(n-4) ...     g^2     g       1]

        Args:
            rewards (torch.Tensor): Tensor of shape (episodes, steps), containing the
                rewards obtained at every step.
            discount (float): Discount factor for future rewards.

        Returns:
            discounted_returns (torch.Tensor): Tensor of shape (episodes, steps), giving
                the discounted cumulative returns for each time-step of every episode.
        """
        _, steps = rewards.shape
        toeplitz = [[discount ** j for j in range(i,-1,-1)] + [0]*(steps-i-1) for i in range(steps)]
        toeplitz = torch.FloatTensor(toeplitz).to(self._policy_network.device)
        discounted_returns = torch.matmul(rewards, toeplitz)
        return discounted_returns

    @torch.no_grad()
    def _discounted_baselined_returns(self, rewards, discount):
        """Compute the discounted baselined return for every episode in the batch.

        The return for an episode is compute as the discounted cumulative sum of all
        rewards received during that episode.
        The baseline is compute as the mean of the returns of all episodes in the batch.

        Subtracting the baseline from the total return has the effect of centering the
        return, giving positive values to good episodes and negative values to bad
        episodes.

        Args:
            rewards (torch.Tensor): Tensor of shape (batch_size, steps), containing the
                rewards obtained at every step.
            discount (float): Discount factor for future rewards.

        Returns:
            discounted_returns (torch.Tensor): Tensor of shape (batch_size, 1), giving the
                discounted baselined return for every episode.
        """
        _, steps = rewards.shape
        device = self._policy_network.device
        discounts = torch.FloatTensor([discount ** i for i in range(steps)]).to(device)
        discounted_returns = torch.sum(torch.mul(rewards, discounts), dim=-1, keepdim=True)
        discounted_returns -= torch.mean(torch.sum(rewards, dim=-1))
        return discounted_returns

#