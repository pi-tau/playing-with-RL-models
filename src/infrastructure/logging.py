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

    def add_episode_length(self, agent, L):
        pass

    def add_buffer_capacity(self, agent, capacity):
        pass


class DQNAgentLogger:

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self._mean_Q_values = []
        self._episode_returns = []
        self._episode_lengths = []
        self._experience_capacity = []

    def add_mean_Q(self, agent):
        qvalues = []
        net = agent._learner.Qnetwork
        device = net.device
        with torch.no_grad():
            for _ in range(10):
                batch = agent._buffer.draw(512)
                states = np.array([x.current for x in batch], dtype=np.float32)
                states = torch.from_numpy(states).to(device)
                Qvals = net(states)
                qvalues.append(Qvals.cpu().numpy())
        self._mean_Q_values.append(np.mean(qvalues))
        filename = os.path.join(self.output_dir, 'mean-Q.pdf')
        self.plot_line(filename, self._mean_Q_values)

    def add_return(self, agent, G):
        self._episode_returns.append(G)
        filename = os.path.join(self.output_dir, 'mean-return.pdf')
        self.plot_line(filename, self._episode_returns)

    def add_episode_length(self, agent, L):
        self._episode_lengths.append(L)
        filename = os.path.join(self.output_dir, 'episode-lengths.pdf')
        self.plot_line(filename, self._episode_lengths)

    def add_buffer_capacity(self, agent, capacity):
        self._experience_capacity.append(capacity)
        filename = os.path.join(self.output_dir, 'experience-capacity.pdf')
        self.plot_line(filename, self._experience_capacity)


    def plot_line(self, filename, *plot_args, **plot_kwargs):
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(*plot_args, **plot_kwargs)
        fig.savefig(filename)
        plt.close(fig)
