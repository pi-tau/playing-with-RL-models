# RL with policy gradient methods

This is my attempt at providing clean, "abstraction-free" implementations of
various policy gradient based reinforcement learning algorithms. I have somewhat
tried to adopt the "single-file" implementation strategy for each of the
algorithms in order to make it easier for anyone who wants to read the code.

The code does not aim to be flexible for different parameter configurations or
optimized for solving hard problems and running on multiple GPUs. It is rather a
simplified single-process, single-file implementation exposing all the relevant
details and removing all the confusing abstractions.

The experiences -- observations, actions, rewards, done, are provided as tensors
of shape `(N, T)`. Here `N` is the number of (parallel) trajectories and `T` is
the number of time-steps. Note that each trajectory could contain multiple
episodes that are concatenated one after another.
An example tensor of observations is:
```python
    obs = [
        [ s11,  s12,  s13,  s14,  s15,| s71,  s72,  s73]
        [ s21,  s22,  s23,  s24,  s25,  s26,  s27,  s28]
        [ s31,  s32,  s33,| s51,  s52,  s53,| s81,  s82]
        [ s41,  s42,  s43,  s44,| s61,  s62,  s63,  s64]
    ]
```

We have 4 trajectories each consisting of 8 time-steps. The corresponding `done`
tensor provides information whether each of the observations is a terminal state
for the environment:
```python
    done = [
        [   0,    0,    0,    0, True,    0,    0,    0]
        [   0,    0,    0,    0,    0,    0,    0,    0]
        [   0,    0, True,    0,    0, True,    0,    0]
        [   0,    0,    0, True,    0,    0,    0,    0]
    ]
```