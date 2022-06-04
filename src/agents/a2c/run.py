import sys
import time
sys.path.append("../../..")

import numpy as np
import torch

from src.agents.actor_critic.agent import ACAgent
from src.environment_loop import EnvironmentLoop
from src.envs.environment import Environment
from src.networks.mlp import MLPNetwork
from src.infrastructure.util_funcs import fix_random_seeds, set_printoptions


# Create file to log output during training.
stdout = open("train_history.txt", "w")
# stdout = sys.stdout
seed = 0


# Fix the random seeds for NumPy and PyTorch, and set print options.
fix_random_seeds(seed)
set_printoptions(precision=5, sci_mode=False)


# Check if cuda is available.
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
print(f"Using device: {device}\n", file=stdout)


# Initialize the environment.
env = Environment(layout="testClassic")


# Initialize policy and value networks, move them to device and prepare them for training.
policy_network = MLPNetwork(env.observable_shape(), [1024, 1024], env.num_actions())
policy_network.train()
policy_network = policy_network.to(device)

value_network = MLPNetwork(env.observable_shape(), [1024, 1024], 1)
value_network.train()
value_network = value_network.to(device)


# Initialize a policy gradient agent.
observations_per_step = 6400
batch_size = 32
agent = ACAgent(policy_network, value_network,
    observations_per_step=observations_per_step, batch_size=batch_size,
    discount=0.99, value_lr=1e-5, policy_lr=3e-5, clip_grad=10.0, stdout=stdout)


# Initialize and run the agent-environment feedback loop.
episodes = 500_000
loop = EnvironmentLoop(agent, env, should_update=True, stdout=stdout)
tic = time.time()
loop.run(episodes=episodes, steps=200, log_every=100, demo_every=10_000)
toc = time.time()
print(f"Training on device {device} takes {toc-tic:.3f} seconds", file=stdout)
agent.learner.policy_network.save("policy.bin")


# Close the file stream.
stdout.close()

#