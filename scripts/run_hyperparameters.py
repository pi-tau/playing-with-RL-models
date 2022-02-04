import os
import multiprocessing
import subprocess
from itertools import product

arguments = {
'device': 'cuda',
'game': 'pacman',
'episodes': 3000,
'max_steps': 300,
'batch_size': 512,
'lr': 5e-4,
'initial_eps': 0.3,
'final_eps': 0.05,
'eps_decay_range': 100_000,
'Q_update_every': 10,
'Q_target_update_every': 100,
'Q_regressions': 1,
'capacity': 500_000,
'min_experiences': 50_000,
'output_dir': 'dqn-berkley-pacman'
}

program = 'python -m src.agents.dqn.run'


def main():
    # batch_size = [128, 1024]
    initial_eps = [0.3, 0.6]
    # eps_decay_range = [10_000, 100_000]
    Q_target_update_every = [100, 1000]

    keys = ['initial_eps', 'Q_target_update_every']
    k = 1
    for vals in product(initial_eps, Q_target_update_every):
        d = dict(zip(keys, vals))
        d['output_dir'] = f'dqn-berkley-pacman-{k}'
        args = arguments.copy()
        args.update(d)
        args = [f'{k}={v}' for k, v in d]
        subprocess.run([program] + args)
        k += 1
