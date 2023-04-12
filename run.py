import os
import pickle
import sys
sys.path.append("..")

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
import torch

from src.environment_loop import environment_loop
from src.networks import MLP
from src.ppo import PPOAgent
from src.vpg import VPGAgent


def demo_LunarLander(iter_number, agent):
    if iter_number % 100 != 0:
        return

    env = gym.wrappers.RecordEpisodeStatistics(
        gym.vector.SyncVectorEnv([
            lambda: gym.wrappers.RecordVideo(
                gym.make("LunarLander-v2", render_mode="rgb_array", enable_wind=True),
                video_folder=os.path.join("videos", "LunarLander_ppo", "random", f"iter_{iter_number}"),
                disable_logger=True,
                episode_trigger=lambda i: i == 0,
            ),
            lambda: gym.wrappers.RecordVideo(
                gym.make("LunarLander-v2", render_mode="rgb_array", enable_wind=True),
                video_folder=os.path.join("videos", "LunarLander_ppo", "greedy", f"iter_{iter_number}"),
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
        if (t | tr)[0]: done_r = True
        if (t | tr)[1]: done_g = True; info_g = info["episode"]
        if (done_r and done_g): break

    agent.train_history[iter_number]["Return"].update({
        "test_avg" : info_g["r"][1],
    })
    agent.train_history[iter_number]["Episode Length"].update({
        "test_avg" : info_g["l"][1],
    })


def pg_plays_LunarLander():
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
    num_envs = 16
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
    # agent = VPGAgent(policy_network, value_network, config={
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

    # Run the environment loop
    num_iters = 1001
    steps = 512
    log_dir = os.path.join("logs", "LunarLander_ppo")
    os.makedirs(log_dir, exist_ok=True)
    environment_loop(seed, agent, env, num_iters, steps, log_dir, demo=demo_LunarLander)

    # Generate plots.
    plt.style.use("ggplot")
    for k in agent.train_history[0].keys():
        fig, ax = plt.subplots()
        if "avg" in agent.train_history[0][k].keys():
            avg = np.array([agent.train_history[i][k]["avg"] for i in range(num_iters)])
            ax.plot(avg, label="Average")
        if "std" in agent.train_history[0][k].keys():
            std = np.array([agent.train_history[i][k]["std"] for i in range(num_iters)])
            ax.fill_between(np.arange(num_iters), avg-0.5*std, avg+0.5*std, color="k", alpha=0.25)
        if "run" in agent.train_history[0][k].keys():
            run = np.array([agent.train_history[i][k]["run"] for i in range(num_iters)])
            ax.plot(run, label="Running")
        if "test_avg" in agent.train_history[0][k].keys():
            test_avg = np.array([agent.train_history[i][k]["test_avg"]
                for i in range(num_iters) if "test_avg" in agent.train_history[i][k].keys()])
            xs = np.linspace(0, num_iters, len(test_avg))
            ax.plot(xs, test_avg, label="Test Average")
        if "test_std" in agent.train_history[0][k].keys():
            test_std = np.array([agent.train_history[i][k]["test_std"]
                for i in range(num_iters) if "test_std" in agent.train_history[i][k].keys()])
            ax.fill_between(xs, test_avg-0.5*test_std, test_avg+0.5*test_std, color="k", alpha=0.25)
        ax.legend(loc="upper left")
        ax.set_xlabel("Number of training iterations")
        ax.set_ylabel(k)
        fig.savefig(os.path.join(log_dir, k.replace(" ", "_")+".png"))


if __name__ == "__main__":
    pg_plays_LunarLander()

    # from gymnasium.utils.play import play
    # play(
    #     gym.make("LunarLander-v2", render_mode="rgb_array"),
    #     keys_to_action={"w":1, "a":2, "s":3, "d":4},
    # )
