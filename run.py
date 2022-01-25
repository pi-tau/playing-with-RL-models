import sys
import time

import torch

from src.agents.policy_gradient.agent import PGAgent
from src.environment_loop import EnvironmentLoop
from src.envs.environment import Environment
from src.infrastructure.episode_buffer import EpisodeBuffer
from src.networks.mlp import MLPNetwork
from src.infrastructure.util_funcs import fix_random_seeds, set_printoptions


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
env = Environment(layout="smallClassic")


# Initialize a policy network, move it to device and prepare for training.
policy_network = MLPNetwork(env.shape()[0], [1024, 1024], env.num_actions())
policy_network.train()
policy_network = policy_network.to(device)


# Initialize a policy gradient agent.
buffer = EpisodeBuffer()
agent = PGAgent(policy_network, buffer, learning_rate=1e-4, clip_grad=10.0)


# Initialize and run the agent-environment feedback loop.
loop = EnvironmentLoop(agent, env, should_update=False)
iterations = 100
episodes = 1024

for i in range(iterations):
    tic = time.time()
    print("running iteration:", i)
    loop.run(episodes, steps=500)
    agent.update()
    buffer.flush()
    toc = time.time()
    print(f"  one iteration using device {device} takes {toc-tic:.3f} seconds")

#