import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.distributions import Categorical

from src.agent import PGAgent


class PPOAgent(PGAgent):
    """Proximal policy optimization agent
    https://arxiv.org/abs/1707.06347

    The updates for the policy network are computed using sample episodes
    generated from simulations. Advantages are calculated based on GAE
    (see: https://arxiv.org/abs/1506.02438)
    Multiple policy update steps are performed before the experiences are
    discarded.
    """

    def __init__(self, policy_network, value_network, config={}):
        """Init a PPO agent.

        Args:
            policy_network: torch.nn Module
            value_network: torch.nn Module
                Value network used for computing the baseline.
            config: dict, optional
                Dictionary with configuration parameters, containing:
                policy_lr: float, optional
                    Learning rate parameter for the policy network. Default: 3e-4
                value_lr: float, optional
                    Learning rate parameter for the value network. Default: 3e-4
                discount: float, optional
                    Discount factor for future rewards. Default: 1.
                batch_size: int, optional
                    Batch size for iterating over the set of experiences. Default: 128.
                clip_grad: float, optional
                    Threshold for gradient norm clipping. Default: 1.
                entropy_reg: float, optional
                    Entropy regularization factor. Default: 0.

                # PPO-specific args
                pi_clip: float, optional
                    Clip ratio for clipping the policy objective. Default: 0.02
                vf_clip: float, optional
                    Clip value for clipping the value objective. Default: inf
                tgt_KL: float, optional
                    Maximum KL divergence for early stopping. Default: 0.2
                n_epochs: int, optional
                    Number of epochs of policy updates. Default: 5.
                lamb: float, optional
                    Advantage estimation discounting factor. Default: 0.95
        """
        super().__init__(policy_network, value_network, config)

        # PPO-specific args.
        self.pi_clip = config.get("pi_clip", 0.02)
        self.vf_clip = config.get("vf_clip", None)
        self.tgt_KL = config.get("tgt_KL", 0.2)
        self.n_epochs = config.get("n_epochs", 5)
        self.lamb = config.get("lamb", 0.95)

        # Initialize the policy and value network weights and biases.
        # Weights use orthogonal initialization and biases are zeroed.
        for module in self.policy_network.net:
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(module.weight, gain=0.01) # policy output layer

        for module in self.value_network.net:
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(module.weight, gain=1.) # value output layer

    def update(self, obs, acts, rewards, done):
        """Update the agent policy network and value network using the provided
        experiences.

        Args:
            obs: torch.Tensor
                Tensor of shape (N, T, *), giving the observations produced by
                the agent during multiple roll-outs.
            acts: torch.Tensor
                Tensor of shape (N, T), giving the actions selected by the agent.
            rewards: torch.Tensor
                Tensor of shape (N, T), giving the obtained rewards.
            done: torch.Tensor
                Boolean tensor of shape (N, T), indicating which of the
                observations are terminal states for the environment.
        """
        # Extend the training history with a dict of statistics.
        self.train_history.append({})
        N, T = rewards.shape

        # Bootstrap the last reward of unfinished episodes.
        values = self.value(obs).to(rewards.device) # uses torch.no_grad
        adv = torch.zeros_like(rewards)
        adv[:, -1] = torch.where(done[:, -1], rewards[:, -1], values[:, -1])

        # Compute the generalized advantages.
        for t in range(T-2, -1, -1): # O(T)  \_("/)_/
            delta = rewards[:, t] + (self.discount * values[:, t+1] - values[:, t]) * ~done[:, t]
            adv[:, t] = delta + self.lamb * self.discount * adv[:, t+1] * ~done[:, t]

        # Reshape the inputs for the neural networks.
        obs = obs.reshape(N*T, *obs.shape[2:])
        acts = acts.reshape(N*T)
        adv = adv.reshape(N*T)
        values = values.reshape(N*T)

        # Update value network.
        returns = adv + values
        self.update_value(obs, returns)

        # Update the policy network.
        self.update_policy(obs, acts, adv)

    def update_policy(self, obs, acts, adv):
        """Proximal Policy optimization (clip version).
        With early stopping based on KL divergence.

        Args:
            obs: torch.Tensor
                Tensor of shape (N, *), giving the observations produced by the
                agent during rollout.
            acts: torch.Tensor
                Tensor of shape (N,), giving the actions selected by the agent.
            adv: torch.Tensor
                Tensor of shape (N,), giving the obtained advantages.
        """
        # self.policy_network.train()
        eps = torch.finfo(torch.float32).eps
        device = self.policy_network.device

        # Compute the probs using the old parameters.
        logp_old = self.policy(obs).log_prob(acts.to(device)) # uses torch.no_grad

        # Update the policy multiple times.
        n_updates = 0
        total_pi_loss, total_loss, total_norm, j = 0., 0., 0., 0
        for _ in range(self.n_epochs):

            # For each epoch run through the entire set of experiences and
            # update the policy by sampling mini-batches at random.
            # https://github.com/DLR-RM/stable-baselines3/blob/e5deeed16efb57c34ccdcb14692439154d970527/stable_baselines3/ppo/ppo.py#L196
            # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/ppo2.py#L162
            dataset = data.TensorDataset(obs, acts, adv, logp_old)
            loader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            for o, a, ad, lp_old in loader:
                logits = self.policy_network(o)
                logp = -F.cross_entropy(logits, a.to(logits.device), reduction="none")
                ro = torch.exp(logp - lp_old.to(logp.device))

                # Normalize the advantages and compute the clipped loss.
                # Note that advantages are normalized at the mini-batch level.
                ad = (ad - ad.mean()) / (ad.std() + eps)
                ad = ad.to(logp.device)
                clip_adv = torch.clip(ro, 1-self.pi_clip, 1+self.pi_clip) * ad
                pi_loss = -torch.mean(torch.min(ro * ad, clip_adv))

                # Add entropy regularization. Augment the loss with the mean
                # entropy of the policy calculated over the sampled observations.
                avg_policy_ent = Categorical(logits=logits).entropy().mean(dim=-1)
                loss = pi_loss - self.entropy_reg * avg_policy_ent

                # Backward pass.
                self.policy_optim.zero_grad()
                loss.backward()
                grad_norm = torch.norm(
                    torch.stack([torch.norm(p.grad) for p in self.policy_network.parameters()]))
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.clip_grad)
                self.policy_optim.step()

                # Bookkeeping.
                n_updates += 1
                total_pi_loss += pi_loss.item() * o.shape[0]
                total_loss += loss.item() * o.shape[0]
                total_norm += grad_norm.item() * o.shape[0]
                j += o.shape[0]

            # Check for early stopping.
            logp = self.policy(obs).log_prob(acts.to(device))   # uses torch.no_grad
            KL = torch.mean(logp_old - logp)                    # KL(P,Q) = Sum(P log(Q/P)) = E_P[logQ-logP]
            if self.tgt_KL is not None and KL > 1.5 * self.tgt_KL:
                break

        # Store the stats.
        self.train_history[-1].update({
            "policy_loss"           : total_pi_loss / j,
            "policy_avg_loss"       : total_loss / j,
            "policy_avg_grad_norm"  : total_norm / j,
            "policy_entropy"        : avg_policy_ent.item(), # policy entropy after all updates
            "avg_KL_div"            : KL.item(),             # KL divergence after all updates
            "num_ppo_updates"       : n_updates,
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
        # self.value_network.train()
        returns = returns.reshape(-1, 1) # match the output shape of the net (B, 1)

        # Compute the values using the old parameters.
        values_old = self.value(obs) # uses torch.no_grad

        # Iterate over the collected experiences and update the value network.
        total_vf_loss, total_norm, j = 0., 0., 0
        for _ in range(self.n_epochs):

            # For each epoch run through the entire set of experiences and
            # update the policy by sampling mini-batches at random.
            dataset = data.TensorDataset(obs, returns, values_old)
            train_dataloader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            for o, r, v_old in train_dataloader:
                # Forward pass.
                # We are also clipping the value function loss following the
                # implementation of OpenAI.
                v_pred = self.value_network(o)
                v_clip = v_old + torch.clip(v_pred-v_old, -self.vf_clip, self.vf_clip)
                r = r.to(v_pred.device)
                vf_loss = torch.mean(torch.max((v_pred - r)**2, (v_clip - r)**2))

                # Backward pass.
                self.value_optim.zero_grad()
                vf_loss.backward()
                grad_norm = torch.norm(torch.stack(
                    [torch.norm(p.grad) for p in self.value_network.parameters()]))
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.clip_grad)
                self.value_optim.step()

                # Bookkeeping.
                total_vf_loss += vf_loss.item() * o.shape[0]
                total_norm += grad_norm.item() * o.shape[0]
                j += o.shape[0]

        # Store the stats.
        self.train_history[-1].update({
            "value_avg_loss"       : total_vf_loss / j,
            "value_avg_grad_norm"  : total_norm / j,
        })

#