import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.distributions import Categorical


class PPOAgent:
    """Proximal policy optimization agent
    https://arxiv.org/abs/1707.06347

    The updates for the policy network are computed using sample episodes
    generated from simulations. Advantages are calculated based on GAE.
    (see: https://arxiv.org/abs/1506.02438)
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
        # The networks should already be moved to device.
        self.policy_network = policy_network
        self.value_network = value_network

        # The training history is a list of dictionaries. At every update step
        # we will write the update stats to a dictionary and we will store that
        # dictionary in this list.
        self.train_history = []

        # Unpack the config parameters to configure the agent for training.
        pi_lr = config.get("pi_lr", 3e-4)
        vf_lr = config.get("vf_lr", 3e-4)
        self.discount = config.get("discount", 1.)
        self.batch_size = config.get("batch_size", 128)
        self.clip_grad = config.get("clip_grad", 1.)
        self.entropy_reg = config.get("entropy_reg", 0.)
        self.pi_clip = config.get("pi_clip", 0.02)
        self.vf_clip = config.get("vf_clip", None)
        self.tgt_KL = config.get("tgt_KL", 0.2)
        self.n_epochs = config.get("n_epochs", 5)
        self.lamb = config.get("lamb", 0.95)

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

    def update(self, obs, acts, rewards, _, done):
        """Proximal Policy optimization (clip version).
        With early stopping based on KL divergence.

        Args:
            obs: torch.Tensor
                Tensor of shape (B, T, *), giving the observations produced by
                the agent during multiple roll-outs.
            acts: torch.Tensor
                Tensor of shape (B, T), giving the actions selected by the agent.
            rewards: torch.Tensor
                Tensor of shape (B, T), giving the obtained rewards.
            done: torch.Tensor
                Boolean tensor of shape (B, T), indicating which of the
                observations are terminal states for the environment.
        """
        # Extend the training history with a dict of statistics.
        self.train_history.append({})
        B, T = rewards.shape

        # Reshape the observations to prepare them as input for the neural nets.
        obs = obs.reshape(B*T, *obs.shape[2:])

        # Bootstrap the last reward of unfinished episodes.
        values = self.value(obs).to(rewards.device) # uses torch.no_grad
        values = values.reshape(B, T)               # reshape back to rewards.shape
        adv = torch.zeros_like(rewards)
        adv[:, -1] = torch.where(done[:, -1], rewards[:, -1] - values[:, -1], 0.)

        # Compute the generalized advantages.
        for t in range(T-2, -1, -1): # O(T)  \_("/)_/
            delta = rewards[:, t] + self.discount * values[:, t+1] * ~done[:, t] - values[:, t]
            adv[:, t] = delta + self.lamb * self.discount * adv[:, t+1] * ~done[:, t]

        # Reshape the acts, advantages and values.
        acts = acts.reshape(B*T)
        adv = adv.reshape(B*T)
        values = values.reshape(B*T)
        returns = adv + values  # returns are estimated using TD(lambda)

        # Compute the probs and values using the old parameters.
        device = self.policy_network.device
        logp_old = self.policy(obs).log_prob(acts.to(device)) # uses torch.no_grad
        values_old = self.value(obs)                          # uses torch.no_grad

        # Iterate over the collected experiences and update the networks.
        self.n_updates = 0
        self.pi_losses, self.pi_norms, self.total_losses = [], [], []
        self.vf_losses, self.vf_norms = [], []
        for _ in range(self.n_epochs):
            self.policy_network.train()
            self.value_network.train()

            # For each epoch run through the entire set of experiences and
            # update the policy and value by sampling mini-batches at random.
            dataset = data.TensorDataset(obs, acts, adv, returns, logp_old, values_old)
            loader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for o, a, ad, r, lp_old, v_old in loader:
                self.update_policy(o, a, ad, lp_old)
                self.update_value(o, r, v_old)

            # Check for early stopping.
            pi = self.policy(obs)
            logp = pi.log_prob(acts.to(device)) # uses torch.no_grad
            KL = logp_old - logp                # KL(P,Q) = Sum(P log(Q/P)) = E_P[logQ-logP]
            if self.tgt_KL is not None and KL.mean() > 1.5 * self.tgt_KL:
                break

        # Store the stats.
        self.train_history[-1].update({
            # Policy network stats.
            "Policy Loss"    : {"avg": np.mean(self.pi_losses), "std": np.std(self.pi_losses)},
            "Total_Loss"     : {"avg": np.mean(self.total_losses), "std": np.std(self.total_losses)},
            "Policy Entropy" : {                            # policy entropy after all updates
                "avg": pi.entropy().mean().item(),
                "std": pi.entropy().std().item(),
            },
            "KL Divergence"  : {                            # KL divergence after all updates
                "avg": KL.mean().item(),
                "std": KL.std().item(),
            },
            "Policy Grad Norm": {"avg": np.mean(self.pi_norms), "std": np.std(self.pi_norms)},
            "Num PPO updates" : {"avg": self.n_updates},
            # Value network stats.
            "Value Loss"      : {"avg": np.mean(self.vf_losses), "std": np.std(self.vf_losses)},
            "Value Grad Norm" : {"avg": np.mean(self.vf_norms), "std": np.std(self.vf_norms)},
        })

    def update_policy(self, o, a, ad, lp_old):
        """Perform one gradient update step on the policy network.

        Args:
            o: torch.Tensor
                Tensor of shape (N, *), giving the mini-batch of observations.
            a: torch.Tensor
                Tensor of shape (N,), giving the mini-batch of actions.
            ad: torch.Tensor
                Tensor of shape (N,), giving the mini-batch of advantages.
            lp_old: torch.Tensor
                Tensor of shape (N,), giving the log probs of the actions
                calculated the old parameters of the policy network.
        """
        logits = self.policy_network(o)
        logp = -F.cross_entropy(logits, a.to(logits.device), reduction="none")
        rho = torch.exp(logp - lp_old.to(logp.device))

        # Normalize the advantages and compute the clipped loss.
        # Note that advantages are normalized at the mini-batch level.
        eps = torch.finfo(torch.float32).eps
        ad = (ad - ad.mean()) / (ad.std() + eps)
        ad = ad.to(logp.device)
        clip_adv = torch.clip(rho, 1-self.pi_clip, 1+self.pi_clip) * ad
        pi_loss = -torch.mean(torch.min(rho * ad, clip_adv))

        # Add entropy regularization. Augment the policy loss with the mean
        # entropy of the policy calculated over the sampled observations.
        policy_entropy = Categorical(logits=logits).entropy()
        loss = pi_loss - self.entropy_reg * policy_entropy.mean()

        # Backward pass.
        self.policy_optim.zero_grad()
        loss.backward()
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad) for p in self.policy_network.parameters()]))
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.clip_grad)
        self.policy_optim.step()

        # Bookkeeping.
        self.n_updates += 1
        self.pi_losses.append(pi_loss.item())
        self.pi_norms.append(total_norm.item())
        self.total_losses.append(loss.item())

    def update_value(self, o, r, v_old):
        """Perform one gradient update step on the value network.

        Args:
            o: torch.Tensor
                Tensor of shape (N, *), giving the mini-batch of observations.
            r: torch.Tensor
                Tensor of shape (N,), giving the mini-batch of rewards.
            v_old: torch.Tensor
                Tensor of shape (N,), giving the values of the states calculated
                using the the old parameters of the value network.
        """
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
        total_norm = torch.norm(torch.stack(
            [torch.norm(p.grad) for p in self.value_network.parameters()]))
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.clip_grad)
        self.value_optim.step()

        # Bookkeeping.
        self.vf_losses.append(vf_loss.item())
        self.vf_norms.append(total_norm.item())

#