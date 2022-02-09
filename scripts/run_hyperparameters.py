import os
import multiprocessing
from itertools import product


arguments = {
    'device': 'cpu',
    'game': 'pacman.testClassic',
    'episodes': 10000,
    'max_steps': 300,
    'batch_size': 1024,
    'lr': 5e-4,
    'initial_eps': 0.3,
    'final_eps': 0.05,
    'eps_decay_range': 100_000,
    'Q_update_every': 10,
    'Q_target_update_every': 100,
    'Q_regressions': 1,
    'capacity': 1_000_000,
    'min_experiences': 100_000,
    'output_dir': 'dqn-berkley-pacman'
}

program = 'python -m src.agents.dqn.run'


def run_shell(command):
    os.system(command)


def main():
    params = dict(
        game = ['pacman.testClassic',
                'pacman.smallClassic',
                'pacman.trickyClassic',
                'pacman.originalClassic'],
        # batch_size = [128, 1024],
        lr = [1e-4, 1e-3],
        # initial_eps = [0.3, 0.8],
        # eps_decay_range = [10_000, 100_000],
        # Q_update_every = [4, 10],
        Q_target_update_every = [100, 1000],
    )
    k = 1
    processes = []
    for vals in product(*params.values()):
        d = dict(zip(params.keys(), vals))
        d['output_dir'] = f'dqn-berkley-pacman-{k}'
        args = arguments.copy()
        args.update(d)
        args = [f'--{k}={v}' for k, v in args.items()]
        line = ' '.join([program] + args)
        print(line)
        p = multiprocessing.Process(target=run_shell, args=(line,))
        p.start()
        processes.append(p)
        k += 1

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
