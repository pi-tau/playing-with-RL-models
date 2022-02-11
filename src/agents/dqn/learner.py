import numpy as np
import torch
import torch.nn
import time
from src.core import Learner


class DQNLearner(Learner):

    def __init__(self, Qnetwork, Q_regressions, target_update_every,
                 discount=0.9,
                 device=torch.device('cpu'), batch_size=128,
                 lr=1e-4, lr_decay=1.0, reg=0.0, clip_grad=10):
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
            self.optimizer, step_size=1000, gamma=lr_decay
        )
        self.clip_grad = clip_grad
        self.total_updates = 0
        self.total_target_updates = 0

    @property
    def Qnetwork(self):
        return self._Q_network.copy()

    def step(self, buffer):

        def huber_loss(inputs, target, delta=1.0):
            abs_diff = torch.abs(inputs - target)
            l1_loss = delta * (abs_diff - 0.5 * delta)
            l2_loss = 0.5 * (inputs - target) ** 2
            return torch.mean(torch.where(abs_diff < delta, l2_loss, l1_loss))

        # tic = time.time()
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
            # Pytorch raises the following UserWarning if initializing
            # tensors directly from Python lists:
            # ===================================
            # UserWarning: Creating a tensor from a list of numpy.ndarrays
            # is extremely slow.
            # Please consider converting the list to a single numpy.ndarray
            # with numpy.array() before converting to a tensor
            states0 = np.array(states0, dtype=np.float32)
            states1 = np.array(states1, dtype=np.float32)
            actions = np.array(actions, dtype=np.int64)
            rewards = np.array(rewards, dtype=np.float32)
            done = np.array(done, dtype=bool)
            states0 = torch.from_numpy(states0).to(self.device)
            states1 = torch.from_numpy(states1).to(self.device)
            actions = torch.from_numpy(actions).to(self.device).reshape(-1, 1)
            rewards = torch.from_numpy(rewards).to(self.device)
            # Regression
            # _, argmax_Q = torch.max(self._Q_network(states1), dim=1, keepdim=True)
            # qvalues0 = torch.gather(self._Q_target_net(states0), 1, actions).squeeze()
            # qvalues1 = torch.gather(self._Q_target_net(states1), 1, argmax_Q).squeeze()
            qvalues0 = torch.gather(self._Q_network(states0), 1, actions).squeeze()
            with torch.no_grad():
                qvalues1, _ = torch.max(self._Q_target_net(states1), dim=1)
            TD_target = rewards + self.discount * qvalues1
            TD_target[done] = rewards[done]
            loss = huber_loss(qvalues0, TD_target)
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
            self.total_updates += 1
        # if self.n % 100 == 0:
            # print('Q values:', torch.mean(qvalues0).detach().cpu())
            # print('TD Target:', torch.mean(TD_target).detach().cpu())
            # print('Loss:', loss)
            # total_norm = 0.0
            # for p in self._Q_network.parameters():
                # param_norm = p.grad.data.norm(2)
                # total_norm += param_norm.item() ** 2
            # print(f'Total grad norm: {total_norm ** 0.5:.3f}')
            # toc = time.time()
            # print(f'Last Q-regression took {toc-tic} seconds.')
        # Update Target network maybe
        self.n += 1
        if self.n >= self.target_update_every:
            self._Q_target_net = self.Qnetwork.to(self.device)
            self._Q_target_net.eval()
            self.n = 0
            self.total_target_updates += 1


class DoubleDQNLearner(Learner):

    def __init__(self, Qnetwork, Q_regressions, target_update_every,
                 discount=0.9,
                 device=torch.device('cpu'), batch_size=128,
                 lr=1e-3, lr_decay=1.0, reg=0.0, clip_grad=10):
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
            self.optimizer, step_size=1000, gamma=lr_decay
        )
        self.clip_grad = clip_grad
        self.total_updates = 0
        self.total_target_updates = 0

    @property
    def Qnetwork(self):
        return self._Q_network.copy()

    def step(self, buffer):

        def huber_loss(inputs, target, delta=1.0):
            abs_diff = torch.abs(inputs - target)
            l1_loss = delta * (abs_diff - 0.5 * delta)
            l2_loss = 0.5 * (inputs - target) ** 2
            return torch.mean(torch.where(abs_diff < delta, l2_loss, l1_loss))

        # tic = time.time()
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
            # Pytorch raises the following UserWarning if initializing
            # tensors directly from Python lists:
            # ===================================
            # UserWarning: Creating a tensor from a list of numpy.ndarrays
            # is extremely slow.
            # Please consider converting the list to a single numpy.ndarray
            # with numpy.array() before converting to a tensor
            states0 = np.array(states0, dtype=np.float32)
            states1 = np.array(states1, dtype=np.float32)
            actions = np.array(actions, dtype=np.int64)
            rewards = np.array(rewards, dtype=np.float32)
            done = np.array(done, dtype=bool)
            states0 = torch.from_numpy(states0).to(self.device)
            states1 = torch.from_numpy(states1).to(self.device)
            actions = torch.from_numpy(actions).to(self.device).reshape(-1, 1)
            rewards = torch.from_numpy(rewards).to(self.device)
            # Regression
            with torch.no_grad():
                _, argmax_Q = torch.max(self._Q_network(states1), dim=1, keepdim=True)
                qvalues1 = torch.gather(self._Q_target_net(states1), 1, argmax_Q).squeeze()
            qvalues0 = torch.gather(self._Q_network(states0), 1, actions).squeeze()
            TD_target = rewards + self.discount * qvalues1
            TD_target[done] = rewards[done]
            loss = huber_loss(qvalues0, TD_target)
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
            self.total_updates += 1
        # Update Target network maybe
        self.n += 1
        if self.n >= self.target_update_every:
            self._Q_target_net = self.Qnetwork.to(self.device)
            self._Q_target_net.eval()
            self.n = 0
            self.total_target_updates += 1
