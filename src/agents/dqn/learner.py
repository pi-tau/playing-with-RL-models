import torch
from src.core import Learner


class DQNLearner(Learner):

    def __init__(self, Qnetwork, discount=0.9,
                 device=torch.device('cpu'), batch_size=128,
                 learning_rate=1e-4, lr_decay=1.0, reg=0.0, clip_grad=None):
        self._Qnetwork = Qnetwork.to(device)
        self._discount = discount
        self._device = device
        self._batch_size = batch_size
        self._optimizer = torch.optim.Adam(
            self._Qnetwork.parameters(),
            lr=learning_rate,
            weight_decay=reg
        )
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, step_size=1, gamma=lr_decay
        )
        self._clip_grad = clip_grad

    @property
    def Qnetwork(self):
        return self._Qnetwork.copy()

    def step(self, buffer, n_steps):
        target_network = self._Qnetwork.copy()
        target_network.eval()

        for _ in range(n_steps):
            # Sampling
            batch = buffer.draw(self._batch_size)
            states0, actions, rewards, states1 = [], [], [], []
            for i in range(len(batch)):
                experience = batch[i]
                states0.append(experience.current)
                actions.append(experience.action)
                rewards.append(experience.reward)
                states1.append(experience.next)
            states0 = torch.FloatTensor(states0).to(self._device)
            states1 = torch.FloatTensor(states1).to(self._device)
            actions = torch.LongTensor(actions).to(self._device).reshape(-1, 1)
            rewards = torch.FloatTensor(rewards).to(self._device)
            # Regression
            argmax_Q, _ = torch.max(target_network(states1), dim=1)
            qvalues = torch.gather(self._Qnetwork(states0), 1, actions).squeeze()
            TD_error = qvalues - (rewards + self._discount * argmax_Q)
            loss = 0.5 * torch.mean(TD_error ** 2)
            # Backward pass
            self._optimizer.zero_grad()
            loss.backward()
            if self._clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self._Qnetwork.parameters(),
                    self._clip_grad
                )
            self._optimizer.step()
            self._scheduler.step()
