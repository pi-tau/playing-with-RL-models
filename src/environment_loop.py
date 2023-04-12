import logging
import os
import pickle
import time
import warnings

import numpy as np
import torch
from tqdm import tqdm


def environment_loop(seed, agent, env, num_iters, steps, log_dir, log_every=1, demo=None):
    """Runs a number of agent-environment interaction loops.

    Args:
        seed: int
            Seed for random number generation.
        agent: RL agent object
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
        obs = np.zeros(shape=(steps, num_envs, *env.single_observation_space.shape), dtype=np.float32)
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
            done[s] = (t | tr)

            # If any of the environments is done, then save the statistics.
            if done[s].any():
                episode_returns.extend([
                    infos["episode"]["r"][k] for k in range(num_envs) if (t | tr)[k]
                ])
                episode_lengths.extend([
                    infos["episode"]["l"][k] for k in range(num_envs) if (t | tr)[k]
                ])

                terminated += sum((1 for i in range(num_envs) if t[i]))

        # Transpose `step` and `num_envs` dimensions and cast to torch tensors.
        obs = torch.from_numpy(obs).transpose(1, 0)
        actions = torch.from_numpy(actions).transpose(1, 0)
        rewards = torch.from_numpy(rewards).transpose(1, 0)
        done = torch.from_numpy(done).transpose(1, 0)

        # Pass the experiences to the agent to update the policy.
        agent.update(obs, actions, rewards, done)

        # Bookkeeping.
        assert len(episode_returns) == len(episode_lengths), "lengths must match"
        for r, l in zip(episode_returns, episode_lengths):
            run_ret = r if run_ret is np.nan else 0.99 * run_ret + 0.01 * r
            run_len = l if run_len is np.nan else 0.99 * run_len + 0.01 * l
        with warnings.catch_warnings():
            # We might finish the rollout without completing any episodes. Then
            # taking the mean or std of an empty slice throws a runtime warning.
            # As a result we would get a NaN, which is exactly what we want.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_r, avg_l = np.array(episode_returns).mean(), np.array(episode_lengths).mean()
            std_r, std_l = np.array(episode_returns).std(), np.array(episode_lengths).std()

        # NOTE: These should not be added to the agent training history.
        # Instead we should probably keep another training history from the
        # environment loop, that tracks statistics on a per-episode basis
        # instead of per-iteration basis.
        agent.train_history[i].update({
            "run_return": run_ret,
            "avg_return": avg_r,
            "std_return": std_r,
            "run_length": run_len,
            "avg_length": avg_l,
            "std_length": std_l,
            "terminated": terminated,
            "total_ep"  : len(episode_lengths),
        })

        # Demo.
        if demo is not None:
            demo(i, agent)

        # Log results.
        if i % log_every == 0:
            logging.info(f"\nIteration ({i+1} / {num_iters}):")
            for k, v in agent.train_history[i].items():
                logging.info(f"    {k}: {v:.5f}")

    # Time the entire agent-environment loop.
    toc = time.time()
    logging.info(f"\nTraining took {toc-tic:.3f} seconds in total.")

    # Close the environment, save the agent, and save the training history.
    env.close()
    torch.save(agent, os.path.join(log_dir, "agent.pt"))
    with open(os.path.join(log_dir, "train_history.pickle"), "wb") as f:
        pickle.dump(agent.train_history, f, protocol=pickle.HIGHEST_PROTOCOL)

#