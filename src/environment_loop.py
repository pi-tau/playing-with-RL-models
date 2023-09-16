import logging
import os
import pickle
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


def environment_loop(seed, agent, env, num_iters, steps, log_dir, log_every=1, demo=None):
    """Runs a number of agent-environment interaction loops.

    Args:
        seed: int
            Seed for random number generation.
        agent: Agent
            RL agent object implementing `policy` and `update` methods.
        env: gym.VectorEnv
            Vectorized environment conforming to the gym API.
        num_iters: int
            Number of agent-environment interaction loops.
        steps: int
            Number of time-steps the agent takes before it updates the policy.
        log_dir: str
            Path to a logging folder where useful information will be stored.
        log_every: int, optional
            Log the results every `log_every` iterations. Default: 1.
        demo: func(int, agent), optional
            Function that accepts an integer (the current iteration number) and
            an agent and produces a demo of the performance of the agent.
            Default: None.
    """
    logging.basicConfig(format="%(message)s", filemode="w", level=logging.INFO,
        filename=os.path.join(log_dir, "train.log"))
    tic = time.time()

    # Reset the environment and store the initial observations.
    # Note that during the interaction loops we will not be resetting the
    # environment. The vector environment will autoreset sub-environments after
    # they terminate or truncate.
    num_envs = env.num_envs
    o, _ = env.reset(seed=seed)

    run_ret, run_len = np.nan, np.nan
    for i in tqdm(range(num_iters)):
        # Allocate tensors for the rollout observations.
        obs = np.zeros(
            shape=(steps, num_envs, *env.single_observation_space.shape),
            dtype=np.float32,
        )
        next_obs = np.zeros_like(obs)
        actions = np.zeros(shape=(steps, num_envs), dtype=int)
        rewards = np.zeros(shape=(steps, num_envs), dtype=np.float32)
        done = np.zeros(shape=(steps, num_envs), dtype=bool)

        # Perform parallel step-by-step rollout along multiple trajectories.
        episode_returns, episode_lengths = [], []
        terminated = 0
        for s in range(steps):
            # Sample an action from the agent and step the environment.
            obs[s] = o
            acts = agent.policy(torch.from_numpy(o)).sample() # uses torch.no_grad()
            acts = acts.cpu().numpy()
            o, r, t, tr, infos = env.step(acts)

            actions[s] = acts
            rewards[s] = r
            next_obs[s] = o
            done[s] = (t | tr)

            # If some environment was truncated, then extract the final obs from
            # the returned info.
            if tr.any():
                for k in range(num_envs):
                    next_obs[s][k] = o[k] if not tr[k] else infos["final_observation"][k]
                # next_obs = np.where(tr, infos["final_observation"], o)

            # If any of the environments is done, then save the statistics.
            if done[s].any():
                episode_returns.extend([
                    infos["episode"]["r"][k] for k in range(num_envs) if (t | tr)[k]
                ])
                episode_lengths.extend([
                    infos["episode"]["l"][k] for k in range(num_envs) if (t | tr)[k]
                ])
                terminated += sum((1 for i in range(num_envs) if t[i]))

            # If there is only one env, then stop the loop when done.
            if num_envs == 1 and done[s].all():
                obs, actions, rewards, next_obs, done = \
                  obs[:s], actions[:s], rewards[:s], next_obs[:s], done[:s]
                break

        # Pass the experiences to the agent to update the policy. We have to
        # transpose `step` and `num_envs` dimensions and cast to torch tensors.
        agent.update(
            torch.from_numpy(obs.swapaxes(1, 0)),
            torch.from_numpy(actions.T),
            torch.from_numpy(rewards.T),
            torch.from_numpy(next_obs.swapaxes(1, 0)),
            torch.from_numpy(done.T),
        )

        # Bookkeeping.
        assert len(episode_returns) == len(episode_lengths), "lengths must match"
        total_ep = len(episode_returns)
        ratio_terminated = terminated / total_ep if total_ep > 0 else np.nan
        for r, l in zip(episode_returns, episode_lengths):
            run_ret = r if run_ret is np.nan else 0.99 * run_ret + 0.01 * r
            run_len = l if run_len is np.nan else 0.99 * run_len + 0.01 * l
        with warnings.catch_warnings():
            # We might finish the rollout without completing any episodes. In
            # this case we want to store NaN in the history. Taking the mean or
            # std of an empty slice throws a runtime warning and returns a NaN,
            # which is exactly what we want.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_r, avg_l = np.array(episode_returns).mean(), np.array(episode_lengths).mean()
            std_r, std_l = np.array(episode_returns).std(), np.array(episode_lengths).std()
        agent.train_history[i].update({
            "Return"           : {"avg" : avg_r, "std" : std_r, "run" : run_ret},
            "Episode Length"   : {"avg" : avg_l, "std" : std_l, "run" : run_len},
            "Ratio Terminated" : {"avg" : ratio_terminated},
        })

        # Demo.
        if demo is not None:
            demo(i, agent)

        # Log results.
        if i % log_every == 0:
            logging.info(f"\nIteration ({i+1} / {num_iters}):")
            for k, v in agent.train_history[i].items():
                logging.info(f"    {k}: {v}")

    # Time the entire agent-environment loop.
    toc = time.time()
    logging.info(f"\nTraining took {toc-tic:.3f} seconds in total.")

    # Close the environment, save the agent, and save the training history.
    env.close()
    torch.save(agent, os.path.join(log_dir, "agent.pt"))
    with open(os.path.join(log_dir, "train_history.pickle"), "wb") as f:
        pickle.dump(agent.train_history, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Generate plots from the training and save them in the log folder.
    generate_plots(log_dir, agent.train_history)


def generate_plots(log_dir, train_history):
    num_iters = len(train_history)
    plt.style.use("ggplot")
    train_history[0]["Return"]["test_avg"] = train_history[0]["Return"]["avg"]
    for k in train_history[0].keys():
        fig, ax = plt.subplots()
        if "avg" in train_history[0][k].keys():
            xs = np.array([i for i in range(num_iters) if "avg" in train_history[i][k].keys()])
            avg = np.array([train_history[i][k]["avg"] for i in xs])
            ax.plot(xs, avg, label="Average")
        if "std" in train_history[0][k].keys():
            std = np.array([train_history[i][k]["std"]
                for i in range(num_iters) if "std" in train_history[i][k].keys()])
            ax.fill_between(xs, avg-0.5*std, avg+0.5*std, color="k", alpha=0.25)
        if "run" in train_history[0][k].keys():
            run = np.array([train_history[i][k]["run"] for i in range(num_iters)])
            ax.plot(run, label="Running")
        if "test_avg" in train_history[0][k].keys():
            xs = np.array([i for i in range(num_iters) if "test_avg" in train_history[i][k].keys()])
            test_avg = np.array([train_history[i][k]["test_avg"] for i in xs])
            ax.plot(xs, test_avg, label="Test Average")
        if "test_std" in train_history[0][k].keys():
            test_std = np.array([train_history[i][k]["test_std"]
                for i in range(num_iters) if "test_std" in train_history[i][k].keys()])
            ax.fill_between(xs, test_avg-0.5*test_std, test_avg+0.5*test_std, color="k", alpha=0.25)
        ax.legend(loc="upper left")
        ax.set_xlabel("Number of training iterations")
        ax.set_ylabel(k)
        fig.savefig(os.path.join(log_dir, k.replace(" ", "_")+".png"))

#