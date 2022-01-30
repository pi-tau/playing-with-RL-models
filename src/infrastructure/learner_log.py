import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl


class DQNLearnerLogger:

    def __init__(self, agent):
        self._agent = agent
        self._mean_Q_values = []

    def add_mean_Q(self):
        qvalues = []
        net = self._agent._learner.Qnetwork
        device = net.device()
        with torch.no_grad():
            for _ in range(10):
                batch = self._agent._buffer.draw(100)
                states = torch.FloatTensor([x.current for x in batch]).to(device)
                Qvals = net(states)
                qvalues.append(Qvals.cpu().numpy())
        self._mean_Q_values.append(np.mean(qvalues))

    def plot_mean_Q(self, filepath):
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(self._mean_Q_values)
        fig.savefig(filepath)
        plt.close(fig)
