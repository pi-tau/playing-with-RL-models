# VANILLA POLICY GRADIENT
The objective that we are trying to maximize is the expected return starting
from the initial state:

```math
J(\theta) = \mathbb{E}_{s_t \sim \mu_\theta, \space a_t \sim \pi_\theta}
\bigg[ \sum_{t=0} r_t(a_t, s_t) \bigg]
```

From the policy gradient theorem we now that the gradient of the objective is
given by:

```math
\begin{align}
\displaystyle \nabla_\theta J(\theta)
& \propto \sum_s \mu_\theta(s) \sum_a \nabla_\theta \pi_\theta(a|s) Q(s, a) \\
& \propto \sum_s \mu_\theta(s) \sum_a \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s) Q(s,a),
\end{align}
```

where $\mu_\theta$ is the on-policy distribution of the states under $\pi_\theta$.
Note that the next state $s_{t+1}$ is drawn from the distribution $p(s_t, a_t)$,
and the action $a_t$ is drawn from $\pi_theta$, this means that the distribution
from which states are drawn actually depends on $\theta$.

Writing the gradient as an expectation, we can now estimate it by collecting
samples:

```math
\begin{align}
\displaystyle \nabla_\theta J(\theta)
& \propto E_{s \sim \mu_\theta, \space a \sim \pi_\theta}
\bigg[ \nabla_\theta \log \pi_\theta Q(s,a) \bigg] \\

& \propto E_{s_t \sim \mu_\theta, \space a_t \sim \pi_\theta}
\bigg[ \nabla_\theta \log \pi_\theta \sum_{t'=t}^{T} r_{t'} \bigg].
\end{align}
```

Thus, in order to calculate the gradient for updating the policy parameters we
need to perform a full episode rollout, i.e., until a terminal state is reached,
and then calculate the returns for each of the states of the episode. One simple
trick done in practice is to center sampled returns and normalize them to have
an std of 1. before performing the backward pass. This usually boosts performance
as it stabilizes the training of the neural network.

```python3
returns = (returns - returns.mean()) / returns.std()
```

This modification makes use of a constant baseline at all time-steps for all
trajectories and effectively rescales the learning rate by a factor of $1 / \sigma$.

Note that calculating the gradient this way is only useful if the samples were
collected using the latest policy parameters. If the policy parameters change
then the collected samples are no longer an estimate of the expectation. Thus,
in theory, we are allowed to update the policy only once before we throw away
the data. Because of this inefficient use of the data people usually perform a
single episode rollout before estimating the gradient and updating the model
parameters.

One issue that might arise with longer episodes is that the size of the
observations is too large to fit on the GPU at once. In this case you need to
iterate through the data using mini-batches, but remember that your are allowed
to update only once! Thus, when iterating the gradients are only accumulated,
and only one optimization step is performed at the end.
```python3
optim.zero_grad()
for o, a, r in minibatch_loader(zip(obs, acts, returns)):
    # move o, a and r to device
    r = (r - r.mean()) / r.std()  # normalize on the mini-batch level
    logits = pi(obs)
    logp = F.cross_entropy(logits, a, reduction="none")
    loss = (logp * r).mean()
    loss.backward()

# maybe do gradient clipping here
optim.step()
```