import os
import sys
sys.path.append("..")

import gymnasium as gym
import numpy as np
from numpy.random import normal
import torch

from src.environment_loop import environment_loop
from src.networks import MLP
from src.a2c import A2CAgent
from src.ppo import PPOAgent
from src.vpg import VPGAgent


def demo_LunarLander(iter_number, agent):
    if iter_number % 100 != 0:
        return

    env = gym.wrappers.RecordEpisodeStatistics(
        gym.vector.SyncVectorEnv([
            lambda: gym.wrappers.RecordVideo(
                gym.make("LunarLander-v2", render_mode="rgb_array", enable_wind=True),
                video_folder=os.path.join(
                    "videos", "LunarLander_v2", type(agent).__name__, "random", f"iter_{iter_number}"),
                disable_logger=True,
                episode_trigger=lambda i: i == 0,
            ),
            lambda: gym.wrappers.RecordVideo(
                gym.make("LunarLander-v2", render_mode="rgb_array", enable_wind=True),
                video_folder=os.path.join(
                    "videos", "LunarLander_v2", type(agent).__name__, "greedy", f"iter_{iter_number}"),
                disable_logger=True,
                episode_trigger=lambda i: i == 0,
            ),
        ]),
    )

    o, _ = env.reset(seed=np.random.randint(1000))

    done_r, done_g = False, False
    while True:
        acts = agent.policy(torch.from_numpy(o))
        random_acts = acts.sample()[0]
        greedy_acts = torch.argmax(acts.probs, dim=-1)[1]
        acts = torch.stack((random_acts, greedy_acts))
        acts = acts.cpu().numpy()
        o, r, t, tr, info = env.step(acts)
        if (t | tr)[0]: done_r = True; info_r = info["episode"]
        if (t | tr)[1]: done_g = True; info_g = info["episode"]
        if (done_r and done_g): break

    agent.train_history[iter_number]["Return"].update({
        "avg": info_r["r"][0],
        "test_avg" : info_g["r"][1],
    })
    agent.train_history[iter_number]["Episode Length"].update({
        "avg": info_r["l"][0],
        "test_avg" : info_g["l"][1],
    })


def play_LunarLander(agent_type="ppo"):
    # Use cuda.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the random seeds.
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Create a vectorized gym environment. Initialize each of the sub-environments
    # with a randomized parametrization for a more robust training. A normal
    # distribution with the standard parametrization as mean is used. Sampled
    # values are further clipped to stay in the recommended parameter space.
    num_envs = 16 if agent_type != "vpg" else 1 # vpg works with only one env
    steps_limit = 500
    env = gym.wrappers.RecordEpisodeStatistics(
        gym.vector.SyncVectorEnv([
            lambda:
            # gym.wrappers.TransformReward(   # Rewards are scaled and clipped.
            # gym.wrappers.NormalizeReward(
                gym.make(
                    "LunarLander-v2",
                    gravity=np.clip(normal(-10.0, 1.0), -11.99, -0.01),
                    enable_wind=np.random.choice([True, False]),
                    wind_power=np.clip(normal(15.0, 1.0), 0.01, 19.99),
                    turbulence_power=np.clip(normal(1.5, 0.5), 0.01, 1.99),
                    max_episode_steps=steps_limit,
                )
            #), lambda r: np.clip(r, -10, 10))
            for _ in range(num_envs)
        ]),
    )

    # Create the RL agent.
    in_shape = env.single_observation_space.shape
    out_size = env.single_action_space.n
    policy_network = MLP(in_shape, [64, 64], out_size).to(device)
    value_network = MLP(in_shape, [64, 64], 1).to(device)

    if agent_type == "vpg":
        steps = steps_limit
        agent = VPGAgent(policy_network, value_network, config={
            "pi_lr"      : 3e-4,
            "vf_lr"      : 1e-3,
            "discount"   : 0.99,
            "clip_grad"  : 10.,
            "entropy_reg": 0.01,
        })
    elif agent_type == "a2c":
        steps = 64
        agent = A2CAgent(policy_network, value_network, config={
            "pi_lr"      : 3e-4,
            "vf_lr"      : 1e-3,
            "discount"   : 0.99,
            "clip_grad"  : 10.,
            "entropy_reg": 0.01,
        })
    elif agent_type == "ppo":
        steps = 256
        agent = PPOAgent(policy_network, value_network, config={
            "pi_lr"      : 3e-4,
            "vf_lr"      : 3e-4,
            "discount"   : 0.99,
            "batch_size" : 128,
            "clip_grad"  : 1.,
            "entropy_reg": 0.01,

            # PPO-specific
            "pi_clip" : 0.2,
            "vf_clip" : 100.,
            "tgt_KL"  : 0.02,
            "n_epochs": 3,
            "lamb"    : 0.95,
        })
    else:
        raise ValueError("unknown agent type")

    # Create log directory
    log_dir = os.path.join("logs", "LunarLander_v2", type(agent).__name__)
    os.makedirs(log_dir, exist_ok=True)

    # Run the environment loop
    num_iters = 2001
    environment_loop(seed, agent, env, num_iters, steps, log_dir, demo=demo_LunarLander)


if __name__ == "__main__":
    play_LunarLander(agent_type="ppo")

    # from gymnasium.utils.play import play
    # play(
    #     gym.make("LunarLander-v2", render_mode="rgb_array"),
    #     keys_to_action={"w":1, "a":2, "s":3, "d":4},
    # )
