import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src import core


class PGLearner(core.Learner):
    """PG learner.
    This class implements the learning logic for a policy gradient agent.
    It updates the policy network of the agent by performing single update steps using a
    dataset of full episodes produced by the agent.
    This is an offline and on-policy learning algorithm.

    Attributes:
        policy_network (Network): The agent behavior policy to be optimized.
        config (dict): A dictionary with optimization parameters (learning rate, etc.).
        optimizer (torch.optim): A torch optimizer object used to perform gradient
            descent updates. The learner uses the Adam optimizer.
        scheduler (torch.optim.lr_scheduler): A torch scheduler object used to schedule
            the value of the learning rate.
        running_return (float): Keeping track of the running return from each episode
            during training.
        stdout (file): File object (stream) used for standard output of logging
            information. Default value is `sys.stdout`.
        train_history (dict): A dictionary storing information during training containing:
            mean_return (list(float)): A list of the mean returns history.
            running_return (list(float)): A list of the running returns history.
    """

    def __init__(self, policy_network, config, stdout=sys.stdout):
        """Initialize a policy gradient learner object.

        Args:
            policy_network (Network): A network object used as a behavior policy.
            config (dict): A dictionary with configuration parameters containing:
                discount (float): Discount factor for future rewards.
                learning_rate (float): Learning rate parameter.
                lr_decay (float): Learning rate decay parameter.
                decay_steps (int): Every `decay_steps` decay the learning rate by `lr_decay`.
                reg (float): L2 regularization strength.
                clip_grad (float): Parameter for gradient clipping by norm.
            stdout (file, optional): File object (stream) used for standard output of
                logging information. Default value is `sys.stdout`.
        """
        self.policy_network = policy_network
        self.config = config
        self.running_return = None
        self.stdout = stdout
        self.train_history = {"mean_return":[], "running_return":[]}

        # Initialize the policy network optimizer.
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["reg"],
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=config["decay_steps"], gamma=config["lr_decay"])

    def step(self, buffer, verbose=True):
        """Perform a single policy gradient update step.

        The gradient of the policy is given by the policy gradient theorem as:
            dJ = E[ d log Policy(a_t | s_t) * R_t ]

        To get the gradients of the weights we will backpropagate a "pseudo-loss" given by
        integrating the above equation:
            J_pseudo = E[ log Policy(a_t | s_t) * R_t ]

        Compute an estimate of the pseudo-loss using Monte-Carlo samples.
        Draw a batch of observed episodes from the buffer.
        For each time-step `t` of each episode of the batch do the following:
            - compute the return `R_t` as the discounted reward to go
            - compute the log-probability of selecting action `a_t` while in state `s_t`

        Compute the "pseudo-loss" as the mean accross all time-steps from all episodes of
        the batch.

        Args:
            buffer (core.Buffer): A buffer object used to store episodes of experiences.
            verbose (bool, optional): If True, printout logging information.
                Default value is True.
        """
        stdout = self.stdout
        device = self.policy_network.device
        discount = self.config["discount"]
        clip_grad = self.config["clip_grad"]
        eps = torch.finfo(torch.float32).eps

        # Fetch trajectories from the buffer.
        observations, actions, rewards, masks = buffer.draw(device=device)

        # Compute the discounted cumulative returns and normalize them.
        q_values = self._discounted_cumulative_returns(rewards, masks, discount)
        q_values = self._normalized_returns(q_values, masks)

        # Compute the loss.
        logits = self.policy_network(observations)
        nll = F.cross_entropy(logits.permute(0,2,1), actions, reduction="none")
        weighted_nll = torch.mul(masks * nll, q_values)
        loss = torch.sum(weighted_nll) / torch.sum(masks)

        # Perform backward pass.
        self.optimizer.zero_grad()
        loss.backward()
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad) for p in self.policy_network.parameters()]))
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), clip_grad)
        self.optimizer.step()
        self.scheduler.step()

        # Keep track of the running return.
        mean_return = torch.mean(torch.sum(rewards, dim=-1))
        if self.running_return is None: self.running_return = mean_return
        else: self.running_return = 0.99*self.running_return + 0.01*mean_return

        # Bookkeeping.
        self.train_history["mean_return"].append(mean_return)
        self.train_history["running_return"].append(self.running_return)

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
            tqdm.write(f"Running return:     {self.running_return:.4f}", file=stdout)

    @torch.no_grad()
    def _discounted_cumulative_returns(self, rewards, masks=None, discount=1.0):
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
            masks (torch.Tensor. optional): A tensor of shape (episodes, steps), of boolean
                values, that mask out the part of an episode after it has finished.
                Default value is None, meaning there is no masking.
            discount (float, optional): Discount factor for future rewards.
                Default values is 1.0.

        Returns:
            discounted_returns (torch.Tensor): Tensor of shape (episodes, steps), giving
                the discounted cumulative returns for each time-step of every episode.
        """
        if masks is None:
            masks = torch.ones_like(rewards)
        _, steps = rewards.shape
        device = self.policy_network.device
        toeplitz = [[discount ** j for j in range(i,-1,-1)] + [0]*(steps-i-1) for i in range(steps)]
        toeplitz = torch.FloatTensor(toeplitz).to(device)
        discounted_returns = torch.matmul(rewards, toeplitz)
        return discounted_returns * masks

    @torch.no_grad()
    def _normalized_returns(self, returns, masks=None):
        """Normalize the cumulative rewards to go.

        One way to reduce the variance of the estimate of the policy gradient is to
        subtract a bias term in the formula:
            dJ = E[ d log Policy(a_t | s_t) (R_t - b) ]

        A near optimal choice for a baseline is the expected value of R_t from state s_t.
        To make a Monte-Carlo estimate of the expected value of the return R_t from a
        given state s_t we would need to run multiple trajectories starting from that
        state. However, configuring the environment to start from a particular state s_t
        may not be possible.
        A slightly worse choice for a baseline is the expected value R_t at time-step `t`.
        To make a Monte-Carlo estimate of the value R_t at time-step `t` we simply need
        to take the average from the batch for that time-step.
        Finally, the quantity (R_i,t - E[R_i,t]) is scaled by the standard deviation of
        the return at time-step `t`. The reason for this is that at different time-steps
        we may have wildly different returns producing wildly different gradient updates.
        To provide for a more stable learning process we want to scale the returns at
        every time-step so that the they are normalized.

        Args:
            returns (torch.Tensor): Tensor of shape (episodes, steps), containing the
                returns obtained at every step.
            masks (torch.Tensor. optional): A tensor of shape (episodes, steps), of boolean
                values, that mask out the part of an episode after it has finished.
                Default value is None, meaning there is no masking.

        Returns:
            normalized_returns (torch.Tensor): Tensor of shape (episodes, steps), giving
                the normalized returns for each time-step of every episode.
        """
        if masks is None:
            masks = torch.ones_like(returns)
        # When working with a batch of trajectories, only the active trajectories are
        # considered for calculating the mean and the std.
        device = self.policy_network.device
        eps = torch.finfo(torch.float32).eps
        masked_means = torch.sum(returns, dim=0) / torch.maximum(
                        torch.sum(masks, dim=0), torch.Tensor([1]).to(device))
        masked_vars = torch.sum(torch.square(returns-masked_means), dim=0) / torch.maximum(
                        torch.sum(masks, dim=0), torch.Tensor([1]).to(device))
        masked_stds = torch.maximum(torch.sqrt(masked_vars), torch.Tensor([eps]).to(device))
        return (returns - masked_means) / masked_stds

#