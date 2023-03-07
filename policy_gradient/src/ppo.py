import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.distributions import Categorical

from src.agent import PGAgent


class PPOAgent(PGAgent):

    def update(self, obs, acts, rewards, done):
        """Update the agent policy network using the provided experiences.
        If the agent uses a value network, then it will also be updated.

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
        values = self.value(obs).to(rewards.device)
        adv = torch.zeros_like(rewards)
        adv[:, -1] = torch.where(done[:, -1], rewards[:, -1], values[:, -1])

        # Compute the generalized advantages.
        lamb = 0.9 # lambda for GAE
        for t in range(T-2, -1, -1): # O(T)  \_("/)_/
            delta = rewards[:, t] + (self.discount * values[:, t+1] - values[:, t]) * ~done[:, t]
            adv[:, t] = delta + lamb * self.discount * adv[:, t+1] * ~done[:, t]

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

        # Compute the probs using the old parameters.
        with torch.no_grad():
            logits = self.policy_network(obs)
            logp_old = -F.cross_entropy(logits, acts.to(logits.device), reduction="none")

        K = 5
        clip_epsi = 0.2
        max_KL = 0.01

        # Update the policy multiple times.
        for k in range(K):
            # Compute the surrogate loss.
            logits = self.policy_network(obs)
            logp = -F.cross_entropy(logits, acts.to(logits.device), reduction="none")
            ro = torch.exp(logp - logp_old)

            # Normalize the advantages and compute the clipped loss.
            eps = torch.finfo(torch.float32).eps
            adv = (adv - adv.mean()) / (adv.std() + eps)
            adv = adv.to(logp.device)
            clip_adv = torch.clip(ro, 1-clip_epsi, 1+clip_epsi) * adv
            policy_loss = -torch.mean(torch.min(ro * adv, clip_adv))

            # Add entropy regularization. Augment the loss with the mean entropy
            # of the policy calculated over the sampled observations.
            avg_policy_ent = Categorical(logits=logits).entropy().mean(dim=-1)
            total_loss = policy_loss - self.entropy_reg * avg_policy_ent

            # Backward pass.
            self.policy_optim.zero_grad()
            total_loss.backward()
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad) for p in self.policy_network.parameters()]))
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.clip_grad)
            self.policy_optim.step()

            # Check for early stopping.
            KL = torch.mean(logp_old - logp) # KL(P,Q) = Sum(P log(Q/P)) = E_P[logQ-logP]
            if KL > max_KL:
                break

        # Store the stats.
        self.train_history[-1].update({
            "policy_loss"       : policy_loss.item(),
            "total_loss"        : total_loss.item(),
            "policy_entropy"    : avg_policy_ent.item(),
            "policy_total_norm" : total_norm.item(),
            "avg_KL_div"        : KL.item(),
            "num_ppo_updates"   : k,
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
        returns = returns.reshape(-1, 1) # match the output shape of the net (B, 1)
        dataset = data.TensorDataset(obs, returns)
        train_dataloader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Iterate over the collected experiences and update the value network.
        # self.value_network.train()
        total_loss, total_grad_norm, j = 0., 0., 0
        for o, r in train_dataloader:
            # Forward pass.
            # OpenAI implementation fits the value network by clipping the value loss.
            # (https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L68-L75)
            # pred_old = 0
            # pred = self.value_network(o)
            # r = r.to(pred.device)
            # clip_pred = torch.clip(pred, pred_old-self.clip_epsi, pred_old+self.clip_epsi)
            # value_loss = torch.max(F.mse_loss(pred, r), F.mse_loss(clip_pred, r))

            pred = self.value_network(o)
            value_loss = F.mse_loss(pred, r.to(pred.device))

            # Backward pass.
            self.value_optim.zero_grad()
            value_loss.backward()
            total_norm = torch.norm(torch.stack(
                [torch.norm(p.grad) for p in self.value_network.parameters()]))
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.clip_grad)
            self.value_optim.step()

            # Bookkeeping.
            total_loss += value_loss.item()
            total_grad_norm += total_norm.item()
            j += 1

        # Store the stats.
        self.train_history[-1].update({
            "value_loss"        : total_loss / j,
            "value_total_norm"  : total_grad_norm / j,
        })

#