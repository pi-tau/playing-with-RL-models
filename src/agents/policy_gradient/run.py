import os
import sys
import time
sys.path.append("../../..")

import numpy as np
import torch

from src.agents.policy_gradient.agent import PGAgent
from src.environment_loop import EnvironmentLoop
from src.envs.environment import Environment
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
env = Environment(layout="testClassic", num_ghosts=0)


# Initialize a policy network, move it to device and prepare for training.
policy_network = MLPNetwork(env.observable_shape(), [256, 256], env.num_actions())
policy_network.train()
policy_network = policy_network.to(device)


# Initialize a policy gradient agent.
episodes = 32
agent = PGAgent(policy_network, discount=0.99, episodes=episodes,
    learning_rate=1e-3, clip_grad=5.0, stdout=stdout)


# Initialize and run the agent-environment feedback loop.
iterations = 100
loop = EnvironmentLoop(agent, env, should_update=True)
tic = time.time()
loop.run(episodes=iterations*episodes, steps=200, log_every=episodes, demo_every=10*episodes)
toc = time.time()
print(f"Training on device {device} takes {toc-tic:.3f} seconds", file=stdout)


# Close the file stream.
stdout.close()

#