import numpy as np
import random
from src.core import Buffer


class ReplayBuffer(Buffer):

    def __init__(self, capacity=1_000_000):
        self._data = []
        self._i = 0
        self.capacity = capacity

    def __len__(self):
        return len(self._data)

    def add_first(self, experience):
        raise NotImplementedError('add_first() makes no sense for ReplayBuffer')

    def add(self, experience):
        if len(self._data) < self.capacity:
            self._data.append(experience)
        else:
            self._data[self._i] = experience
        self._i = (self._i + 1) % self.capacity

    def draw(self, num_samples):
        indices = np.random.choice(len(self), size=num_samples, replace=False)
        return [self._data[i] for i in indices]

    def flush(self):
        self._i = 0
        self._data.clear()
