# REINFORCEMENT LEARNING

This is my attempt at providing clean, "abstraction-free" implementations of
various gradient based reinforcement learning algorithms. I have somewhat tried
to adopt the "single-file" implementation strategy for each of the algorithms in
order to make it easier for anyone who wants to read the code.

The code does not aim to be flexible for different parameter configurations or
optimized for solving hard problems and running on multiple GPUs. It is rather a
simplified single-process, single-file implementation exposing all the relevant
details and removing all the confusing abstractions. Maybe it could be used as a
reference if you want to roll out your own implementations.

Implementations of the following algorithms can be found here:
* Vanilla policy gradient -
[code](https://github.com/pi-tau/playing-with-RL-models/blob/main/src/vpg.py),
[docs](https://github.com/pi-tau/playing-with-RL-models/blob/main/docs/vpg.md)
* Advantage Actor-Critic -
[code](https://github.com/pi-tau/playing-with-RL-models/blob/main/src/a2c.py)
[docs](https://github.com/pi-tau/playing-with-RL-models/blob/main/docs/a2c.md)
* Proximal policy optimization -
[code](https://github.com/pi-tau/playing-with-RL-models/blob/main/src/ppo.py),
[docs](https://github.com/pi-tau/playing-with-RL-models/blob/main/docs/ppo.md)

If you want to read more about policy gradient algorithms, then checkout a
[blog post](https://pi-tau.github.io/posts/actor-critic/) that I wrote.