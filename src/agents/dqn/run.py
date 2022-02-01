import os
import sys
import time

import numpy as np
import torch

from src.agents.actors import DQNActor
from .agent import DQNAgent
from .learner import DQNLearner
from src.environment_loop import EnvironmentLoop
from src.envs.environment import Environment
from src.infrastructure.replay_buffer import ReplayBuffer
from src.networks.mlp import MLPNetwork
from src.infrastructure.util_funcs import fix_random_seeds, set_printoptions


# Create file to log output during training.
# log_dir = "logs"
# os.makedirs(log_dir, exist_ok=True)
# stdout = open(os.path.join(log_dir, "train_history.txt"), "w")
stdout = sys.stdout
seed = 0


# Fix the random seeds for NumPy and PyTorch, and set print options.
fix_random_seeds(seed)
set_printoptions(precision=5, sci_mode=False)


# Check if cuda is available.
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
print(f"Using device: {device}\n", file=stdout)


# Initialize the environment.
env = Environment(layout="testClassic", graphics=False)


q_network = MLPNetwork(env.shape()[0], [512, 1024], env.num_actions())
q_network.train()
q_network = q_network.to(device)

# Initialize a Deep Q-Learning agent
batch_size = 256
buffer = ReplayBuffer()
epsilon = 0.8

actor = DQNActor(q_network, buffer, epsilon)
learner = DQNLearner(q_network, batch_size=batch_size)
agent = DQNAgent(actor, learner, buffer, 1000, 500, 10)

# Initialize and run the agent-environment feedback loop.
iterations = 100
loop = EnvironmentLoop(agent, env, should_update=False)
tic = time.time()
loop.run(episodes=iterations*batch_size, steps=100)
toc = time.time()
print(f"Training on device {device} takes {toc-tic:.3f} seconds", file=stdout)

#