import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.distributions import Categorical


class A2CAgent:
    """Advantage Actor-Critic (A2C) agent.
    The updates for the policy network are computed using sample episodes
    generated from simulations. A bootstrapped estimate of the advantage is
    computed using a value network. A single policy update step is performed
    before the experiences are discarded.
    """

    def __init__(self, policy_network, value_network, config={}):
        """Init an A2C agent.

        Args:
            policy_network: torch.nn Module
            value_network: torch.nn Module
            config: dict, optional
                Dictionary with configuration parameters, containing:
                pi_lr: float, optional
                    Learning rate parameter for the policy network. Default: 3e-4
                vf_lr: float, optional
                    Learning rate parameter for the value network. Default: 3e-4
                discount: float, optional
                    Discount factor for future rewards. Default: 1.
                batch_size: int, optional
                    Batch size for iterating over the set of experiences. Default: 128.
                clip_grad: float, optional
                    Threshold for gradient norm clipping. Default: None.
                entropy_reg: float, optional
                    Entropy regularization factor. Default: 0.
        """
        # The networks should already be moved to device.
        self.policy_network = policy_network
        self.value_network = value_network

        # we will write the update stats to a dictionary and we will store that
        # dictionary in this list.
        self.train_history = []

        # Unpack the config parameters to configure the agent for training.
        pi_lr = config.get("pi_lr", 3e-4)
        vf_lr = config.get("vf_lr", 3e-4)
        self.discount = config.get("discount", 1.)
        self.batch_size = config.get("batch_size", 128)
        self.clip_grad = config.get("clip_grad", None)
        self.entropy_reg = config.get("entropy_reg", 0.)

        # Initialize the optimizers.
        self.policy_optim = torch.optim.Adam(self.policy_network.parameters(), lr=pi_lr)
        self.value_optim = torch.optim.Adam(self.value_network.parameters(), lr=vf_lr)

    @torch.no_grad()
    def policy(self, obs):
        self.policy_network.eval()
        return Categorical(logits=self.policy_network(obs))

    @torch.no_grad()
    def value(self, obs):
        self.value_network.eval()
        return self.value_network(obs).squeeze(dim=-1)

    def update(self, obs, acts, rewards, next_obs, done):
        """Update the agent policy and value networks using the provided experiences.

        Args:
            obs: torch.Tensor
                Tensor of shape (B, T, *), giving the observations produced by
                the agent during interacting with the environment.
            acts: torch.Tensor
                Tensor of shape (B, T), giving the actions selected by the agent.
            rewards: torch.Tensor
                Tensor of shape (B, T), giving the obtained rewards.
            next_obs: torch.Tensor
                Tensor of shape (B, T, *) giving the observations produced right
                after applying the selected actions.
            done: torch.Tensor
                Boolean tensor of shape (N,) indicating which of the
                observations are terminal states for the environment.
        """
        # Reshape the inputs for the neural networks.
        B, T = rewards.shape
        obs = obs.reshape(B*T, *obs.shape[2:])
        next_obs = next_obs.reshape(B*T, *next_obs.shape[2:])
        acts = acts.reshape(B*T)

        # Compute the returns and advantages using multi-step bootstrap.
        # The return for each state will be computed by summing all the rewards
        # along the current trajectory and only at the end we will bootstrap.
        values = self.value(obs).to(rewards.device).reshape(B, T)
        next_values = self.value(next_obs).to(rewards.device).reshape(B, T)
        next_values = torch.where(done, 0., next_values)
        returns = torch.zeros_like(rewards)
        returns[:, -1] = torch.where(done[:, -1], rewards[:, -1], values[:, -1])
        for t in range(T-2, -1, -1):
            returns[:, t] = rewards[:, t] + self.discount * returns[:, t+1] * ~done[:, t]
        adv = returns - values

        # Reshape the inputs for the neural networks.
        returns = returns.reshape(B*T)
        adv = adv.reshape(B*T)

        # Update the value and the policy networks.
        self.train_history.append({})
        self.update_value(obs, returns)
        self.update_policy(obs, acts, adv)

    def update_policy(self, obs, acts, adv):
        """Perform one gradient update step on the policy network.

        Args:
            obs: torch.Tensor
                Tensor of shape (N, *), giving the observations produced by the
                agent during rollout.
            acts: torch.Tensor
                Tensor of shape (N,), giving the actions selected by the agent.
            adv: torch.Tensor
                Tensor of shape (N,), giving the obtained advantages.
        """
        # Forward pass.
        self.policy_network.train()
        logits = self.policy_network(obs)
        logp = F.cross_entropy(logits, acts.to(logits.device), reduction="none")

        # Normalize the advantages and compute the pseudo-loss.
        eps = torch.finfo(torch.float32).eps
        adv = (adv - adv.mean()) / (adv.std() + eps)
        adv = adv.to(logp.device)
        pi_loss = torch.mean(logp * adv)

        # Add entropy regularization. Augment the loss with the mean entropy of
        # the policy calculated over the sampled observations.
        policy_entropy = Categorical(logits=logits).entropy()
        total_loss = pi_loss - self.entropy_reg * policy_entropy.mean()

        # Backward pass.
        self.policy_optim.zero_grad()
        total_loss.backward()
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad) for p in self.policy_network.parameters()]))
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.clip_grad)
        self.policy_optim.step()

        # Store the stats.
        self.train_history[-1].update({
            "Policy Loss"    : {"avg": pi_loss.item()},
            "Total_Loss"     : {"avg": total_loss.item()},
            "Policy Entropy" : {
                "avg": policy_entropy.mean().item(),
                "std": policy_entropy.std().item(),
            },
            "Policy Grad Norm": {"avg": total_norm.item()},
        })

    def update_value(self, obs, returns):
        """Update the value network to fit the value function of the current
        policy `V_pi`. This functions performs a single iteration over the
        set of experiences drawing mini-batches of examples and fits the value
        network using MSE loss.

        Args:
            obs: torch.Tensor
                Tensor of shape (N, *), giving the observations of the agent.
            returns: torch.Tensor
                Tensor of shape (N,), giving the obtained returns.
        """
        # Create a dataloader object for iterating through the examples.
        returns = returns.reshape(-1, 1) # match the output shape of the net (N, 1)
        dataset = data.TensorDataset(obs, returns)
        train_dataloader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Iterate over the collected experiences and update the value network.
        self.value_network.train()
        vf_losses, vf_norms = [], []
        for o, r in train_dataloader:
            # Forward pass.
            pred = self.value_network(o)
            vf_loss = F.mse_loss(pred, r.to(pred.device))
            # Backward pass.
            self.value_optim.zero_grad()
            vf_loss.backward()
            total_norm = torch.norm(torch.stack(
                [torch.norm(p.grad) for p in self.value_network.parameters()]))
            if self.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.clip_grad)
            self.value_optim.step()

            # Bookkeeping.
            vf_losses.append(vf_loss.item())
            vf_norms.append(total_norm.item())

        # Store the stats.
        self.train_history[-1].update({
            "Value Loss"      : {"avg": np.mean(vf_losses), "std": np.std(vf_losses)},
            "Value Grad Norm" : {"avg": np.mean(vf_norms), "std": np.std(vf_norms)},
        })


#