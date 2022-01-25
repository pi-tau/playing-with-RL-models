import sys

import torch
import torch.nn.functional as F

from src import core


class PGLearner(core.Learner):
    """PG learner.
    This class implements the learning logic for a policy gradient agent.
    The updates the policy network of the agent by performing single update steps
    using a dataset of full episodes produced by the agent.
    This is an online and on-policy learning algorithm.

    Attributes:
        _policy_network (core.Network): The agent behavior policy to be optimized.
        _optim (dict): A dictionary with optimization parameters (learning rate, etc.).
        _optimizer (torch.optim): A torch optimizer object used to perform gradient
            descent updates. The learner uses the Adam optimizer.
        _scheduler (torch.optim.lr_scheduler): A torch scheduler object used to schedule
            the value of the learning rate.
        _stdout (file, optional): File object (stream) used for standard output of logging
            information. Default value is `sys.stdout`.
    """

    def __init__(self, policy_network, optim, stdout=sys.stdout):
        """Initialize a policy gradient learner object.

        Args:
            policy_network (core.Network): A network object used as a behavior policy.
            optim (dict): A dictionary with optimization parameters (learning rate, etc.).
            stdout (file, optional): File object (stream) used for standard output of
                logging information. Default value is `sys.stdout`.
        """
        self._policy_network = policy_network
        self._optim = optim
        self._optimizer = torch.optim.Adam(
            self._policy_network.parameters(),
            lr=optim["learning_rate"],
            weight_decay=optim["reg"],
        )
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, step_size=1, gamma=optim["lr_decay"])
        self._stdout = stdout

    def step(self, buffer, verbose=True):
        """Perform a single policy gradient update step.
        Draw a batch of episodes from the buffer and compute an estimate of the gradients
        of the policy network.
        Use the `reward-to-go` trick to reduce the variance of the estimate.
        Subtract a baseline to reduce the variance of the estimate.

        Args:
            buffer (core.Buffer): A buffer object used to store episodes of experiences.
            verbose (bool, optional): If True, printout logging information. Default value is True.
        """
        stdout = self._stdout
        device = self._policy_network.device

        # Fetch trajectories from the buffer.
        observations, actions, rewards, masks = buffer.draw(device=device)

        # Compute the loss.
        logits = self._policy_network(observations)
        q_values = self._reward_to_go(rewards)
        q_values -= self._reward_baseline(rewards, masks)
        nll = F.cross_entropy(logits.permute(0,2,1), actions, reduction="none")
        weighted_nll = torch.mul(masks * nll, q_values)
        loss = torch.mean(torch.sum(weighted_nll, dim=1))

        # Perform backward pass.
        self._optimizer.zero_grad()
        loss.backward()
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad) for p in self._policy_network.parameters()]))
        torch.nn.utils.clip_grad_norm_(
            self._policy_network.parameters(), self._optim["clip_grad"])
        self._optimizer.step()
        self._scheduler.step()

        if verbose:
            probs = F.softmax(logits, dim=-1)
            avg_policy_ent = -torch.mean(torch.sum(probs*torch.log(probs), axis=-1))
            print(f"Mean return:        {torch.mean(torch.sum(rewards, axis=1)): .4f}", file=stdout)
            print(f"Best return:        {max(torch.sum(rewards, axis=1)): .4f}", file=stdout)
            print(f"Avg num of steps:   {torch.mean(torch.sum(masks, axis=1, dtype=float)): .0f}", file=stdout)
            print(f"Longest episode:    {max(torch.sum(masks, axis=1, dtype=float)): .0f}", file=stdout)
            print(f"Pseudo loss:        {loss.item(): .5f}", file=stdout)
            print(f"Grad norm:          {total_norm: .5f}", file=stdout)
            print(f"Avg policy entropy: {avg_policy_ent: .3f}", file=stdout)
            print(f"Total num of steps: {torch.sum(masks): .0f}", file=stdout)

    def _sum_to_go(self, t):
        """Sum-to-go returns the sum of the values starting from the current index. Given
        an array `arr = {a_0, a_1, ..., a_(T-1)}` the sum-to-go is an array `s` such that:
            `s[0] = a_0 + a_1 + ... + a_(T-1)`
            `s[1] = a_1 + ... + a_(T-1)`
            ...
            `s[i] = a_i + a_(i+1) + ... + a_(T-1)`

        Args:
            t (torch.Tensor): Tensor of shape (N1, N2, ..., Nk, steps), where the values
                to be summed are along the last dimension.

        Returns:
            sum_to_go (torch.Tensor): Tensor of shape (N1, N2, ..., Nk, steps)
        """
        return t + torch.sum(t, keepdims=True, dim=-1) - torch.cumsum(t, dim=-1)

    def _reward_to_go(self, rewards):
        """Compute the reward-to-go at every timestep t.
        "Don't let the past destract you"
        Taking a step with the gradient pushes up the log-probabilities of each action in
        proportion to the sum of all rewards ever obtained. However, agents should only
        reinforce actions based on rewards obtained after they are taken.
        Check out: https://spinningup.openai.com/en/latest/spinningup/extra_pg_proof1.html

        Args:
            rewards (torch.Tensor): Tensor of shape (batch_size, steps), containing the
                rewards obtained at every step.

        Returns:
            reward_to_go (torch.Tensor): Tensor of shape (batch_size, steps).
        """
        return self._sum_to_go(rewards)

    def _reward_baseline(self, rewards, masks):
        """Compute the baseline as the average return at timestep t.

        The baseline is usually computed as the mean total return.
            `b = E[sum(r_1, r_2, ..., r_t)]`
        Subtracting the baseline from the total return has the effect of centering the
        return, giving positive values to good trajectories and negative values to bad
        trajectories. However, when using reward-to-go, subtracting the mean total return
        won't have the same effect. The most common choice of baseline is the value
        function V(s_t). An approximation of V(s_t) is computed as the mean reward-to-go.
            `b[i] = E[sum(r_i, r_(i+1), ..., r_T)]`

        Args:
            rewards (torch.Tensor): Tensor of shape (batch_size, steps), containing the
                rewards obtained at every step.
            masks (torch.Tensor): Boolean tensor of shape (batch_size, steps), that masks
                out the part of the trajectory after it has finished.

        Returns:
            baselines (torch.Tensor): Tensor of shape (steps,), giving the baseline term
                for every timestep.
        """
        # # When working with a batch of trajectories, only the active trajectories are
        # # considered for calculating the mean baseline. The reward-to-go sum of finished
        # # trajectories is 0.
        # baselines = torch.sum(self._reward_to_go(rewards), dim=0) / torch.maximum(
        #                 torch.sum(masks, dim=0), torch.Tensor([1]).to(self._policy_network.device))

        # # Additionally, if there is only 1 active trajectory in the batch, then the
        # # the baseline for that trajectory should be 0.
        # return (torch.sum(masks, dim=0) > 1) * baselines

        steps = rewards.shape[-1]
        b = torch.mean(torch.sum(rewards, dim=-1))
        lengths = torch.sum(masks, dim=-1, keepdim=True)
        mod_rewards = rewards - masks * (b/lengths).repeat(1, steps)
        return self._sum_to_go(mod_rewards)

#