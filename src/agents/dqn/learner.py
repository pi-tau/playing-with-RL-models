import numpy as np
import torch
import torch.nn
from src.core import Learner


class DQNLearner(Learner):

    def __init__(self, Qnetwork, Q_regressions, target_update_every,
                 discount=0.9,
                 device=torch.device('cpu'), batch_size=128,
                 lr=1e-4, lr_decay=1.0, reg=0.0, clip_grad=None):
        self._Q_network = Qnetwork.to(device)
        self._Q_target_net = Qnetwork.copy().to(device)
        self.Q_regressions = Q_regressions
        self.target_update_every = target_update_every
        self.n = 0   # counts the number of steps since last Target net update
        self.discount = discount
        self.device = device
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(
            self._Q_network.parameters(),
            lr=lr,
            weight_decay=reg
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=lr_decay
        )
        self.clip_grad = clip_grad

    @property
    def Qnetwork(self):
        return self._Q_network.copy()

    def step(self, buffer):

        def huber_loss(inputs, target, delta=1.0):
            abs_diff = torch.abs(inputs - target)
            l1_loss = delta * (abs_diff - 0.5 * delta)
            l2_loss = 0.5 * (inputs - target) ** 2
            return torch.mean(torch.where(abs_diff < delta, l2_loss, l1_loss))

        for _ in range(self.Q_regressions):
            # Sampling
            batch = buffer.draw(self.batch_size)
            states0, actions, rewards, states1, done = [], [], [], [], []
            for i in range(len(batch)):
                experience = batch[i]
                states0.append(experience.current)
                actions.append(experience.action)
                rewards.append(experience.reward)
                states1.append(experience.next)
                done.append(experience.done)
            states0 = torch.FloatTensor(states0).to(self.device)
            states1 = torch.FloatTensor(states1).to(self.device)
            actions = torch.LongTensor(actions).to(self.device).reshape(-1, 1)
            rewards = torch.FloatTensor(rewards).to(self.device)
            done = np.array(done, dtype=bool)
            # Regression
            argmax_Q, _ = torch.max(self._Q_network(states1), dim=1)
            qvalues = torch.gather(self._Q_target_net(states0), 1, actions).squeeze()
            TD_target = rewards + self.discount * argmax_Q
            TD_target[done] = rewards[done]
            loss = huber_loss(qvalues, TD_target)
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self._Q_network.parameters(),
                    self.clip_grad
                )
            self.optimizer.step()
            self.scheduler.step()

        # Update Target network maybe
        self.n += 1
        if self.n >= self.target_update_every:
            self._Q_target_net = self.Qnetwork
            self._Q_target_net.eval()
            self.n = 0
