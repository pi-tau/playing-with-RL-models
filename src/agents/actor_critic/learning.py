import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src import core


class ACLearner(core.Learner):
    """AC learner.
    This class implements the learning logic for an actor-critic agent.
    It updates the policy network of the agent by performing single update steps using a
    dataset of full episodes produced by the agent.
    This is an offline and on-policy learning algorithm.

    Attributes:
        policy_network (Network): The agent behavior policy to be optimized.
        value_network (Network): The critic value network used to evaluate environment
            states.
        config (dict): A dictionary with optimization parameters (learning rate, etc.).
        policy_optimizer (torch.optim): A torch optimizer object used to perform gradient
            descent updates on the policy network weights. The learner uses the Adam optimizer.
        policy_scheduler (torch.optim.lr_scheduler): A torch scheduler object used to
            schedule the value of the learning rate for the policy optimizer.
        value_optimizer (torch.optim): A torch optimizer object used to perform gradient
            descent updates on the value network weights. The learner uses the Adam optimizer.
        value_scheduler (torch.optim.lr_scheduler): A torch scheduler object used to
            schedule the value of the learning rate for the value optimizer.
        running_return (float): Keeping track of the running return from each episode
            during training.
        stdout (file): File object (stream) used for standard output of logging information.
    """

    def __init__(self, policy_network, value_network, config, stdout=sys.stdout):
        """Initialize an actor-critic learner object.

        Args:
            policy_network (Network): A network object used as a behavior policy.
            value_network (Network): A network object used to evaluate environment states.
            config (dict): A dictionary with configuration parameters containing:
                discount (float): Discount factor for future rewards.
                policy_lr (float): Learning rate parameter for the policy network.
                policy_lr_decay (float): Learning rate decay parameter for the policy network.
                policy_decay_steps (int): Every `decay_steps` decay the learning rate of
                    the policy network by `policy_lr_decay`.
                policy_reg (float): L2 regularization strength for the policy network.
                value_lr (float): Learning rate parameter for the value network.
                value_lr_decay (float): Learning rate decay parameter for the value network.
                value_decay_steps (int): Every `decay_steps` decay the learning rate of
                    the value network by `value_lr_decay`.
                value_reg (float): L2 regularization strength for the value network.
                batch_size (int): Batch size parameter for updating the value network.
                clip_grad (float): Parameter for gradient clipping by norm.
            stdout (file, optional): File object (stream) used for standard output of
                logging information. Default value is `sys.stdout`.
        """
        self.policy_network = policy_network
        self.value_network = value_network
        self.config = config
        self.stdout = stdout

        # Initialize the policy network optimizer.
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=config["policy_lr"],
            weight_decay=config["policy_reg"],
        )
        self.policy_scheduler = torch.optim.lr_scheduler.StepLR(
            self.policy_optimizer, step_size=config["policy_decay_steps"],
            gamma=config["policy_lr_decay"]
        )

        # Initialize the value network optimizer.
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(),
            lr=config["value_lr"],
            weight_decay=config["value_reg"],
        )
        self.value_scheduler = torch.optim.lr_scheduler.StepLR(
            self.value_optimizer, step_size=config["value_decay_steps"],
            gamma=config["value_lr_decay"]
        )

    def step(self, buffer, verbose=True):
        """Perform a single policy gradient update step.

        The gradient of the policy is given by the policy gradient theorem as:
            dJ = E[ d log Policy(a_t | s_t) * (R_t - b(s_t)) ],
        where b(s_t) is a state-dependent baseline.

        To get the gradients of the weights we will backpropagate a "pseudo-loss" given by
        integrating the above equation:
            J_pseudo = E[ log Policy(a_t | s_t) * (R_t - b(s_t)) ].

        To compute the return `R_t` we will use a one-step bootstrap and for the baseline
        we will use the value of the state `s_t`. Thus, we arrive at the "advantage" for
        the state-action pair `(s_t, a_t)`:
            A(s_t, a_t) = R_t - b(s_t)
            A(s_t, a_t) = r_t + V(s_(t+1)) - V(s_t).

        Compute an estimate of the pseudo-loss using samples drawn during interacting with
        the environment.
        Draw a batch of observed experiences from the buffer.

        To fit a value function to predict the value of a state `s_t` we will use the batch
        of experiences drawn from the buffer. The target value of a state `s_t` is the
        bootstrapped estimate:
            V(s_t) = r_t + V(s_(t+1)).
        NOTE: We could also use a monte carlo estimate for the value of a state `s_t`:
            V(s_t) = sum_t(r_t),
        however this estimate has very high variance and performs poorly.

        To fit the value network we will use a mean-squared error loss.
        NOTE: We could fit the value network performing multiple epochs of gradient descent
        updates. Performing only one epoch of updates is good enough as the policy network
        changes very slowly.

        Using the fitted value network compute the advantage `A(s_t, a_t)` for every state-
        action pair in the batch of experiences. Compute the "pseudo-loss" as the mean
        accross all sampled experiences of the batch.

        Args:
            buffer (core.Buffer): A buffer object used to store steps of experiences.
            verbose (bool, optional): If True, printout logging information.
                Default value is True.
        """
        device = self.policy_network.device
        stdout = self.stdout
        discount = self.config["discount"]
        batch_size = self.config["batch_size"]
        clip_grad = self.config["clip_grad"]
        eps = torch.finfo(torch.float32).eps

        # Fetch observation experiences from the buffer.
        prev_obs, actions, rewards, next_obs = buffer.draw(device=device)

        # Fit the value network.
        total_loss, total_grad_norm, i = 0.0, 0.0, 0
        for idxs in torch.randperm(len(prev_obs)).to(device).split(batch_size):
            # Compute the targets for the value network using one-step bootstrapping.
            prev_values = self.value_network(prev_obs[idxs]).squeeze(dim=-1)
            next_values = self.value_network(next_obs[idxs]).squeeze(dim=-1)
            targets = rewards[idxs] + discount * next_values

            # Compute the loss for the value network.
            value_loss = 0.5 * F.mse_loss(prev_values, targets, reduction="mean")

            # Perform backward pass for the value network.
            self.value_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad) for p in self.value_network.parameters()]))
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), clip_grad)
            self.value_optimizer.step()
            self.value_scheduler.step()

            total_loss += value_loss
            total_grad_norm += total_norm
            i += 1

        # Update the policy network.
        # Compute the advantages using the critic,
        prev_values = self.value_network(prev_obs).squeeze(dim=-1)
        next_values = self.value_network(next_obs).squeeze(dim=-1)
        advantages = rewards + discount * next_values - prev_values

        # Compute the loss for the policy gradient.
        logits = self.policy_network(prev_obs)
        nll = F.cross_entropy(logits, actions, reduction="none")
        weighted_nll = torch.mul(nll, advantages)
        loss = torch.mean(weighted_nll)

        # Perform backward pass for the policy network.
        self.policy_optimizer.zero_grad()
        loss.backward()
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad) for p in self.policy_network.parameters()]))
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), clip_grad)
        self.policy_optimizer.step()
        self.policy_scheduler.step()

        # Maybe log training information.
        if verbose:
            probs = F.softmax(logits, dim=-1)
            probs = torch.maximum(probs, torch.tensor(eps))
            avg_policy_ent = -torch.mean(torch.sum(probs*torch.log(probs), dim=-1))
            tqdm.write("#-- value network update --#", file=stdout)
            tqdm.write(f"  Avg Value Loss:  {total_loss / i:.5f}", file=stdout)
            tqdm.write(f"  Avg Grad norm:   {total_grad_norm / i:.5f}", file=stdout)
            tqdm.write("#-- policy network update --#", file=stdout)
            tqdm.write(f"  Pseudo loss:        {loss.item(): .5f}", file=stdout)
            tqdm.write(f"  Grad norm:          {total_norm: .5f}", file=stdout)
            tqdm.write(f"  Avg policy entropy: {avg_policy_ent: .3f}", file=stdout)

#