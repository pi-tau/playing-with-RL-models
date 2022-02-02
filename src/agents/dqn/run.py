# TODO
# =============================================================================
# * Run experiments (shell scripts)
# * Enhance features
# * Asynchronous actors
# -----------------------------------------------------------------------------
import argparse
import datetime
import os
import time
import torch

from src.agents.actors import DQNActor
from src.agents.dqn.agent import DQNAgent
from src.agents.dqn.learner import DQNLearner
from src.environment_loop import EnvironmentLoop
from src.envs.environment import Environment
from src.envs.atari import AtariEnvironment
from src.networks.mlp import MLPNetwork
from src.networks.cnn import ConvolutionalNet
from src.infrastructure.replay_buffer import ReplayBuffer
from src.infrastructure.util_funcs import fix_random_seeds, set_printoptions
from src.infrastructure.logging import DQNAgentLogger


_STARTUP_PLATE_ = \
    """
    ---------------------------------------------------------------------------
    Deep Q-Learning Agent started at {starttime} with:

    Device:                         {device}
    Game:                           {game}
    Skip Frames:                    {skip_frames}
    Stack Frames:                   {stack_frames}
    Number of Episodes:             {n_episodes}
    Total Experiences:              {n_experiences}
    Max Steps per Episode:          {max_steps}
    Batch Size:                     {batch_size}
    Learning Rate:                  {learning_rate}
    Policy Epsilon:                 {epsilon}
    Discount Factor:                {discount}
    Fit Q Network every:            {Q_update_every} experiences
    Update Target Q Network every:  {target_update_every} experiences
    Q Network Regressions:          {Q_regressions}
    Replay Buffer Capacity:         {capacity}
    Minimum Number of Experiences:  {min_experiences}
    Output Directory:               {output_dir}
    ===========================================================================

    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], type=str, default='cpu')
    parser.add_argument('--game', action='store', type=str, default='MsPacman-v4')
    parser.add_argument('--experiences', action='store', type=int, default=100_000)
    parser.add_argument('--max_steps', action='store', type=int, default=10_000)
    parser.add_argument('--batch_size', action='store', type=int, default=128)
    parser.add_argument('--lr', type=float, action='store', default=1e-3)
    parser.add_argument('--epsilon', type=float, action='store', default=1.0)
    parser.add_argument('--Q_update_every', action='store', type=int, default=1000)
    parser.add_argument('--Q_target_update_every', action='store', type=int, default=2000)
    parser.add_argument('--Q_regressions', action='store', type=int, default=10)
    parser.add_argument('--capacity', action='store', type=int, default=100_000)
    parser.add_argument('--min_experiences', action='store', type=int, default=10_000)
    parser.add_argument('--output_dir', action='store', type=str)
    args = parser.parse_args()

    # All hyperparameters
    DEVICE = torch.device(args.device)
    GAME = args.game
    SKIP_FRAMES = 4
    STACK_FRAMES = 4
    NUM_EPSIODES = int(args.experiences / args.max_steps)
    NUM_EXPERIENCES = args.experiences
    MAX_STEPS = args.max_steps
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    LR_DECAY = 1.0
    L2_REGULARIZATION = 1e-3
    CLIP_GRAD = True
    EPSILON = args.epsilon
    DISCOUNT = 0.9
    Q_UPDATE_EVERY = args.Q_update_every
    TARGET_UPDATE_EVERY = args.Q_target_update_every
    Q_REGRESSIONS = args.Q_regressions
    CAPACITY = args.capacity
    MIN_EXPERIENCES = args.min_experiences
    OUTPUT_DIR = args.output_dir

    # Fix the random seeds for NumPy and PyTorch, and set print options
    fix_random_seeds(0)
    set_printoptions(precision=5, sci_mode=False)

    # Initialize the environment
    if args.game == 'pacman':
        env = Environment(layout='testClassic', graphics=False)
        Q_network = MLPNetwork(env.shape()[0], [512, 1024], env.num_actions())
    else:
        env = AtariEnvironment(
            args.game,
            frameskip=SKIP_FRAMES,
            framestack=STACK_FRAMES
        )
        Q_network = ConvolutionalNet(
            input_shape=env.shape(),
            output_shape=env.num_actions()
        )
    obs = env.reset()
    print(obs.observation.shape)
    Q_network = Q_network.to(DEVICE)
    Q_network.train()

    # Initialize output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize Deep Q-Learning agent
    buffer = ReplayBuffer(capacity=CAPACITY)
    actor = DQNActor(Q_network, buffer, epsilon=EPSILON)
    learner = DQNLearner(Q_network, discount=DISCOUNT, device=DEVICE,
                         Q_regressions=Q_REGRESSIONS,
                         target_update_every=TARGET_UPDATE_EVERY,
                         batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                         lr_decay=LR_DECAY, reg=L2_REGULARIZATION,
                         clip_grad=CLIP_GRAD)
    agent = DQNAgent(actor, learner, buffer, min_experiences=MIN_EXPERIENCES,
                     Q_update_every=Q_UPDATE_EVERY,
                     logger=DQNAgentLogger(OUTPUT_DIR))

    print(_STARTUP_PLATE_.format(
        starttime=datetime.datetime.now(),
        device=DEVICE,
        game=GAME,
        skip_frames=SKIP_FRAMES,
        stack_frames=STACK_FRAMES,
        n_episodes=NUM_EPSIODES,
        n_experiences=NUM_EXPERIENCES,
        max_steps=MAX_STEPS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epsilon=EPSILON,
        discount=DISCOUNT,
        Q_update_every=Q_UPDATE_EVERY,
        target_update_every=TARGET_UPDATE_EVERY,
        Q_regressions=Q_REGRESSIONS,
        capacity=CAPACITY,
        min_experiences=MIN_EXPERIENCES,
        output_dir=OUTPUT_DIR
    ))
    # Initialize and run the agent-environment feedback loop.
    loop = EnvironmentLoop(agent, env, should_update=True)
    tic = time.time()
    loop.run(episodes=NUM_EPSIODES, steps=MAX_STEPS)
    toc = time.time()
    print(f'\nElapsed time: {toc - tic} sec.')
