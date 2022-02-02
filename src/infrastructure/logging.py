import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

class NullLogger:

    def add_mean_Q(self, agent):
        pass

    def add_return(self, agent, G):
        pass

    def plot_mean_Q(self):
        pass

    def plot_mean_return(self):
        pass


class DQNAgentLogger:

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self._mean_Q_values = []
        self._returns = []

    def add_mean_Q(self, agent):
        qvalues = []
        net = agent._learner.Qnetwork
        device = net.device
        with torch.no_grad():
            for _ in range(10):
                batch = agent._buffer.draw(128)
                states = torch.FloatTensor([x.current for x in batch]).to(device)
                Qvals = net(states)
                qvalues.append(Qvals.cpu().numpy())
        self._mean_Q_values.append(np.mean(qvalues))
        self.plot_mean_Q()

    def add_return(self, agent, G):
        self._returns.append(G)
        self.plot_mean_return()

    def plot_mean_Q(self):
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(self._mean_Q_values)
        fig.savefig(os.path.join(self.output_dir, 'mean-Q.pdf'))
        plt.close(fig)

    def plot_mean_return(self):
        fig, ax = plt.subplots(figsize=(16,8))
        ax.plot(self._returns)
        fig.savefig(os.path.join(self.output_dir, 'returns.pdf'))
        plt.close(fig)
