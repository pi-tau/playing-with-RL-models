# ADVANTAGE ACTOR-CRITIC

Actor-Critic algorithms update the policy network using the policy gradient
formula:

```math
\displaystyle \nabla_\theta J(\theta) = E_{s \sim \mu_\theta, \space a \sim \pi_\theta}
\bigg[ \nabla_\theta \log \pi_\theta \big( R(s,a) - V_\phi(s) \big) \bigg],
```

However, instead of using a Monte-Carlo estimate, we compute the return $R$ using
bootstrapping:

```math
R(s_t, a_t) = r_{t+1} + V(s_{t+1})
```

Combining this bootstrapped estimate of the returns with a baseline we get the
update formula for the advantage actor-critic.

With this setup we actually don’t have to run episodes until the end. In fact we
can update the policy (and the value network) at every single step. But that is
not a very good idea because we will be estimating the gradient using a single
sample. Instead, what we could do is:

* either run multiple environments in parallel in order to obtain multiple
samples at every step,
* or rollout the episode for several steps and only then perform the update.

We will actually do both.

Note that we are rolling out the policy for multiple steps, but we are computing
the return using a single-step bootstrap. What we could do instead is use an
n-step bootstrap estimation: the return for each state will be computed by
summing all the rewards along the current trajectory and only at the end we will
bootstrap:

```math
R(s_t, a_t) = r_{t+1} + r_{t+2} + \cdots + r_{T} + V(s_{T})
```
