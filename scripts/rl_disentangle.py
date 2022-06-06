"""
python3 rl_disentangle.py -q 5 -b 1024 -i 5000 --steps 40 --ereg 0.01
"""

import argparse
import os
import sys
import time
sys.path.append("..")

import torch

from src.networks.mlp import MLPNetwork
from src.agents.vpg.agent import PGAgent
from src.envs.qubit_env import Environment
from src.environment_loop import EnvironmentLoop
from src.infrastructure.logging import plot_with_averaged_curves
from src.infrastructure.util_funcs import fix_random_seeds, set_printoptions


# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", dest="seed", type=int,
    help="random seed value", default=0)
parser.add_argument("-q", "--num_qubits", dest="num_qubits", type=int,
    help="Number of qubits in the quantum system", default=4)
parser.add_argument("--epsi", dest="epsi", type=float,
    help="Threshold for disentanglement", default=1e-3)
parser.add_argument("-b", "--batch_size", dest="batch_size", type=int,
    help="Number of episodes in a batch.", default=128)
parser.add_argument("--steps", dest="steps", type=int,
    help="Number of steps in an episode", default=10)
parser.add_argument("-i", "--num_iter", dest="num_iter", type=int,
    help="Number of iterations to run the training for", default=101)
parser.add_argument("--lr", dest="learning_rate", type=float,
    help="Learning rate", default=1e-4)
parser.add_argument("--lr_decay", dest="lr_decay", type=float,
    help="Step decay the learning rate at every iteration", default=1.0)
parser.add_argument("--reg", dest="reg", type=float,
    help="L2 regularization", default=0.0)
parser.add_argument("--ereg", dest="ereg", type=float,
    help="Entropy regularization temperature", default=0.0)
parser.add_argument("--clip_grad", dest="clip_grad", type=float,
    help="Clip the gradients by norm.", default=10.0)
parser.add_argument("--dropout", dest="dropout", type=float, default=0.0)
parser.add_argument("--log_every", dest="log_every", type=int, default=100)
parser.add_argument("--test_every", dest="test_every", type=int, default=1000)
parser.add_argument("--device", dest="device", type=str, default="cpu")
args = parser.parse_args()


# Fix the random seeds for NumPy and PyTorch, and set print options.
fix_random_seeds(args.seed)
set_printoptions(precision=5, sci_mode=False)


# Create file to log output during training.
log_dir = "../logs/qubit_sys/batch_{}_iters_{}_ereg_{}".format(
    args.batch_size, args.num_iter, args.ereg)
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "train.log")
stdout = open(log_file, "w")


# Create the environment.
env = Environment(args.num_qubits, epsi=args.epsi)

if args.device == "cuda":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
else:
    device = torch.device("cpu")

# Initialize the policy network and the agent.
input_shape = env.shape
hidden_dims = [4096, 4096, 512]
output_size = env.num_actions()
policy_network = MLPNetwork(input_shape, hidden_dims, output_size, args.dropout)
policy_network.train()
policy_network = policy_network.to(device)
agent = PGAgent(policy_network, discount=1., batch_size=args.batch_size,
    learning_rate=args.learning_rate, lr_decay=args.lr_decay, reg=args.reg,
    ereg=args.ereg, clip_grad=args.clip_grad, stdout=stdout)


# Initialize and run the Environment Loop.
tic = time.time()
loop = EnvironmentLoop(agent, env, stdout=stdout)
loop.run(args.num_iter * args.batch_size, args.steps,
    log_every=args.log_every * args.batch_size)
toc = time.time()
print(f"Training took {toc-tic:.3f} seconds.", file=stdout)


# Plot the results from the environment loop.
avg_every = args.log_every * args.batch_size
plot_with_averaged_curves(loop.run_history["returns"], avg_every=avg_every,
    label="Return", figname=os.path.join(log_dir, "returns.png"))
plot_with_averaged_curves(loop.run_history["nsteps"], avg_every=avg_every,
    label="nsteps", figname=os.path.join(log_dir, "nsteps.png"))

#