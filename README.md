# RL with policy gradient methods

This is my attempt at providing clean, "abstraction-free" implementations of
various policy gradient based reinforcement learning algorithms. I have somewhat
tried to adopt the "single-file" implementation strategy for each of the
algorithms in order to make it easier for anyone who wants to read the code.

The code does not aim to be flexible for different parameter configurations or
optimized for solving hard problems and running on multiple GPUs. It is rather a
simplified single-process, single-file implementation exposing all the relevant
details and removing all the confusing abstractions. Maybe it could be used as a
reference if you want to roll out your own implementations

### Vanilla policy gradient
The agent expects the experiences to be provided in the form
`(observations, actions, rewards, done)`, each is a tensor of shape `(N, T)`.
Here `N` is the number of (parallel) trajectories and `T` is the number of
time-steps. Note that each trajectory could contain multiple episodes that are
concatenated one after another. An example tensor of observations is:
```python3
    obs = [
        [ o_11,  o_12,  o_13,  o_14,  o_15,| o_71,  o_72,  o_73]
        [ o_21,  o_22,  o_23,  o_24,  o_25,  o_26,  o_27,  o_28]
        [ o_31,  o_32,  o_33,| o_51,  o_52,  o_53,| o_81,  o_82]
        [ o_41,  o_42,  o_43,  o_44,| o_61,  o_62,  o_63,  o_64]
    ]
```

We have 4 trajectories each consisting of 8 time-steps. The corresponding `done`
tensor provides information whether each of the observations is a terminal state
for the environment, i.e. whether the environment was terminated or truncated at
that step:
```python3
    done = [
        [   0,    0,    0,    0, True,    0,    0,    0]
        [   0,    0,    0,    0,    0,    0,    0,    0]
        [   0,    0, True,    0,    0, True,    0,    0]
        [   0,    0,    0, True,    0,    0,    0,    0]
    ]
```

Returns for each state are calculated as the cumulative discounted reward starting
from the given state until the end of the episode. If a sub-environment is not
terminated nor truncated at the end of a trajectory then we have two options:

* mask the unfinished episode, since we cannot accurately compute the returns
    for any of the states
* bootstrap the the return of the last visited state using the value network
    (only works if a value network is provided)
    $R = r_0 + \gamma r_1 + \gamma^2 r_2 + ... + \gamma^k r_k + \gamma^{k+1} V(s_{k+1})$

The objective that we are trying to maximize is the expected return starting from
the initial state:

$J(\theta) = \mathbb{E} \bigg[ \sum_{t=0} r_t \bigg]$

From the policy gradient theorem we now that the gradient of the objective is
given by:

$\displaystyle \nabla_\theta J(\theta) \propto \sum_s \mu(s) \sum_a \nabla_\theta \pi_\theta(a|s) Q(s, a)$

$\displaystyle \nabla_\theta J(\theta) \propto \sum_s \mu(s) \sum_a \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s) Q(s,a)$,

where $\mu$ is the on-policy distribution of the states under $\pi_\theta$.

Writing the gradient as an expectation, we can now estimate it by collecting
samples:

$\nabla_\theta J(\theta) \propto E_{s \sim \mu, a \sim \pi} \bigg[ \nabla_\theta \log \pi_\theta Q(s,a) \bigg]$

$\displaystyle \nabla_\theta J(\theta) \propto E_{s_t \sim \mu, a_t \sim \pi} \bigg[ \nabla_\theta \log \pi_\theta \sum_{t'=t}^{T} r_{t'} \bigg]$

Note that calculating the gradient this way is only useful if the samples were
collected using the latest policy parameters. If the policy parameters change
then the collected samples are no longer an estimate of the expectation. Thus,
in theory, we are allowed to update the policy only once before we throw away
the data.

Given that, people usually estimate the gradient with a single
trajectory (i.e. `N=1`). Using multiple-trajectories however, would reduce the
variance of the estimate and would lead to better updates. If the environment is
easily parallelizable then setting `N>1` is probably a better choice. One issue
that might arise with larger `N` though is that the size of the observations is
too large to fit on the GPU at once. In this case you need to iterate through
the data using mini-batches, but remember that your are allowed to update only
once! Thus, when iterating the gradients are only accumulated, and only one
optimization step is performed at the end.
```python3
optim.zero_grad()
for o, a, r in minibatch_loader(zip(obs, acts, returns)):
    # move o, a and r to device
    r = (r - r.mean()) / r.std()  # normalize on the mini-batch level
    logits = pi(o)
    logp = F.cross_entropy(logits, a, reduction="none")
    loss = (logp * r).mean()
    loss.backward()

# maybe do gradient clipping here
optim.step()
```

One simple trick done in practice is to center sampled returns and normalize
them to have an std of 1 before performing the backward pass. This usually boosts
performance as it stabilizes the training of the neural network.

```python3
returns = (returns - returns.mean()) / returns.std()
```

This modification makes use of a constant baseline at all time-steps for all
trajectories and effectively rescales the learning rate by a factor of $1 / \sigma$.


