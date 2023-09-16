# VANILLA POLICY GRADIENT
The objective that we are trying to maximize is the expected return starting
from any state drawn from $\mu_\theta$:

```math
J(\theta) = \mathbb{E}_{s_t \sim \mu_\theta, \space a_t \sim \pi_\theta}
\bigg[ \sum_{t=0} r_t(a_t, s_t) \bigg].
```

Here $\mu_\theta$ is the on-policy distribution of the states under $\pi_\theta$.
Note that the next state $s_{t+1}$ is drawn from the distribution $p(s_t, a_t)$,
and the action $a_t$ is drawn from $\pi_\theta$, this means that the distribution
from which states are drawn actually depends on $\theta$.

The formula for updating the policy network using the policy gradient theorem is:
```math
\displaystyle \nabla_\theta J(\theta) = E_{s \sim \mu_\theta, \space a \sim \pi_\theta}
\bigg[ \nabla_\theta \log \pi_\theta R(s,a) \bigg],
```
where $R(s,a)$ is the return obtained for state $s$ if we select action $a$ and
would continue to follow the policy $\pi_\theta$.

In vanilla policy gradient (vpg) the return is calculated as a Monte-Carlo
estimate. We perform a full episode rollout and we collect the states, actions,
and rewards ($s_t$, $a_t$, $r_{t+1}$) at each step until we reach a terminal
state. Once we've reached a terminal state we estimate the return for each of
the visited states as:

```math
R_t = \sum_{i=t}^T r_{i+1}
```

Finally we compute the gradient as:

```math
\nabla_\theta J(\theta) = \frac{1}{T} \sum_{t=1}^T \nabla \log \pi_\theta (a_t | s_t) R_t.
```

Each of the states was visited using the current policy and the state
distribution function of the environment, so for each state-action pair we can
compute an estimate of the gradient. Then, we average over the samples, much
like a mini-batch update in supervised learning.

Note that calculating the gradient this way is only useful if the samples were
collected using the latest policy parameters. If the policy parameters change
then the collected samples are no longer an estimate of the expectation. Thus,
in theory, we are allowed to update the policy only once before we throw away
the data. Because of this inefficient use of the data people usually perform a
single episode rollout before estimating the gradient and updating the model
parameters.

Compute the discounted cumulative reward-to-go at every time-step `t`.

When computing the return we will discount future rewards by a discount factor.
Multiplying the rewards by a discount factor can be interpreted as encouraging
the agent to focus more on the rewards that are closer in time. This can also be
thought of as a means for reducing variance, because there is more variance
possible when considering rewards that are further into the future. The
cumulative return at time-step `t` is computed as the sum of all future rewards
starting from the current time-step. The discounted cumulative return for all
states of the episode can be computed as a matrix multiplication between the
rewards vector and a special toeplitz matrix.
```math
\text{toeplitz} =
\begin{matrix}
1       & 0       & 0       & 0       & \cdots     & 0       & 0       & 0 \\
g       & 1       & 0       & 0       & \cdots     & 0       & 0       & 0 \\
g^2     & g       & 1       & 0       & \cdots     & 0       & 0       & 0 \\
g^3     & g^2     & g       & 1       & \cdots     & 0       & 0       & 0 \\
\cdots \\
g^(n-2) & g^(n-3) & g^(n-4) & g^(n-5) & \cdots     & g       & 1       & 0 \\
g^(n-1) & g^(n-2) & g^(n-3) & g^(n-4) & \cdots     & g^2     & g       & 1
\end{matrix}
```

```python3
returns = rewards @ toeplitz
```

The single-sample Monte-Carlo estimate of the returns might have a lot of
variance. A good approach for reducing the variance of the estimation is to add
a baseline to the estimate of the return. We can baseline the returns using the
value function by approximating it using a second neural network which is
trained concurrently with the policy. The formula for the gradient would become:

```math
\nabla_\theta J(\theta) = \frac{1}{T} \sum_{t=1}^T \nabla \log \pi_\theta (a_t | s_t) \Big( R_t - V_\phi(s_t) \Big).
```
where $(R_t - V_\phi(s_t))$ is the *advantage*, describing how much better it is
to take action $a_t$ compared to the other actions.

One simple trick done in practice is to center final returns and normalize
them to have an std of 1.0 before performing the backward pass. This usually
boosts performance as it stabilizes the training of the neural network.

```python3
returns = (returns - returns.mean()) / returns.std()
```

This modification makes use of a constant baseline at all time-steps and
effectively rescales the learning rate by a factor of $1 / \sigma$. Note that
even though we might baseline the returns we are still normalizing after that.
This effectively means that we are using a different baseline, but this will
not change the expectation of the gradient. It will simply improve the training
of our network.

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
