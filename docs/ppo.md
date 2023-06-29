# PROXIMAL POLICY OPTIMIZATION
Proximal policy optimization was introduced in the paper
*"Proximal policy optimization algorithms"* by Schulman et. al.
([here](https://arxiv.org/abs/1707.06347))

The modification introduced by PPO to the vanilla PG algorithm seems like it is
very simple and easy to implement, but there are a lot of other crucial details
that also need to be taken into consideration. Checkout the brilliant
[blog post](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
*"The 37 details of proximal policy optimization"* for a thorough discussion on
all the details.


## THE ALGORITHM
The problem that the authors are trying to solve is to come up with an algorithm
that allows us to take the biggest possible update step on the policy parameters
before throwing out the collected rollout data. The approach taken in this paper
is to allow for multiple update steps that, when combined, would approximate
this maximum possible update. Note that after we update the policy parameters
once, $\pi_\theta = \pi_{\theta_{old}} + \nabla \theta_{old}$, every other update
would actually be using off-policy data. To correct for this data miss-match we
can use importance sampling:

```math
\begin{align}
\displaystyle J(\theta)
& = \mathbb{E}_{s_t \sim \mu_\theta, \space a_t \sim \pi_\theta}
    \bigg[ \sum_{t=0} r_t(a_t, s_t) \bigg] \\
& = \mathbb{E}_{s_t \sim \mu_\theta, \space a_t \sim \pi_\theta}
    \bigg[
        \frac{\pi_{\theta_{old}}(a|s)}{\pi_{\theta_{old}}(a|s)} \sum_{t=0} r_t(a_t, s_t)
    \bigg] \\
& = \mathbb{E}_{s_t \sim \mu_\theta, \space a_t \sim \pi_{\theta_{old}}}
    \bigg[
        \frac{\pi_{\theta_{old}}(a|s)}{\pi_\theta(a|s)} \sum_{t=0} r_t(a_t, s_t)
    \bigg]
\end{align}
```

It seems like we could compute the objective to update the new policy weights
using the data collected with the old policy weights, as long as we correct with
the importance sampling weight:

```math
\displaystyle \rho(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}.
```

Thus, we could perform multiple update steps using the collected rollout data.
However, note that, in order to compute the correct gradient estimate, the
actions have to be sampled under $\pi_{\theta_{old}}, but the states have to be
sampled under $\mu_\theta$. Unfortunately our data was sampled under
$\mu_{\theta_{old}}$.

How bad is that?

It turns out that if $\pi_\theta$ does not deviate *too much* from
$\pi_{\theta_{old}}$, then using the old data sampled from $\mu_{\theta_{old}}$
is actually ok. The difference between the objectives calculated using $\mu_\theta$
and $\mu_{\theta_{old}}$ is bounded, and thus, optimizing one would also optimize
the other. In simple words, it is ok to optimize the objective with the old data
as long as $\pi_\theta$ is close to $\pi_{\theta_{old}}$. A proof of this claim
can be found in Appendix A of the [paper](https://arxiv.org/abs/1502.05477)
*"Trust Region Policy Optimization"* by Schulman et. al.

There are two different proximal policy algorithms each using a different
heuristic to try to ensure that $\pi_\theta$ is close to $\pi_{\theta_{old}}$:

* **PPO-Penalty** - constraints the *KL divergence* between the two distributions
by adding it as a penalty to the objective:

```math
J(\theta) =
\mathbb{E}_{s_t \sim \mu_\theta, \space a_t \sim \pi_{\theta_{old}}}
\bigg[
    \rho(\theta) A(s,a) - \beta KL(\pi_\theta(\cdot|s), \pi_{\theta_{old}}(\cdot|s))
\bigg]
```

* **PPO-CLIP** - clips the objective function if $\pi_\theta$ deviates too much
from $\pi_{\theta_{old}}$:

```math
J(\theta) =
\mathbb{E}_{s_t \sim \mu_\theta, \space a_t \sim \pi_{\theta_{old}}}
\bigg[
    \min \big(
        \rho(\theta) A(s,a), \space \text{clip}(\rho(\theta), 1-\epsilon, 1+\epsilon) A(s,a)
    \big)
\bigg]
```

The algorithm implemented here is **PPO-CLIP** augmented with a check for
early stopping. If the mean *KL divergence* between $\pi_\theta$ and
$\pi_{\theta_{old}}$ grows beyond a given threshold, then we stop taking
gradient steps and we collect new rollout data.


## FIXED-LENGTH SEGMENTS
In Algorithm 1. of the PPO paper the authors state that they use "fixed-length
trajectory segments" for training the agent. What this means is that there are
two phases to training:
* Phase 1 - Rollout phase. The agent performs a fixed-length $T$ step rollout
collecting (state, action, reward) triples.
* Phase 2 - Learning phase. The agent performs multiple ppo updates to the
policy weights using the collected data.
Once the learning phase is over the agent starts a new rollout phase but
continues to step the environment from where it left off, i.e. the environment
is not reset at the beginning of the rollout phase. This allows PPO to learn in
long-horizon tasks where episodes could be extremely long (think 100K steps). In
addition, rollout is performed on $N$ environments in parallel, allowing for a
more efficient data collection.

After the rollout phase is over the experiences are stored in the following
$N \times T$ tensors: `observations`, `actions`, `rewards`, `done`. The `done`
tensor provides information whether each of the observations is a terminal state
for the environment, i.e. whether the environment was terminated or truncated at
that step. Note that each fixed-length segment could contain multiple episodes
that are concatenated one after another. An example tensor of observations is:
```python3
    obs = [
        [ o_11,  o_12,  o_13,  o_14,  o_15,| o_71,  o_72,  o_73]
        [ o_21,  o_22,  o_23,  o_24,  o_25,  o_26,  o_27,  o_28]
        [ o_31,  o_32,  o_33,| o_51,  o_52,  o_53,| o_81,  o_82]
        [ o_41,  o_42,  o_43,  o_44,| o_61,  o_62,  o_63,  o_64]
    ]
```

We have 4 trajectories each consisting of 8 time-steps. The corresponding `done`
tensor indicates which of the observations are from terminal states:
```python3
    done = [
        [   0,    0,    0,    0, True,    0,    0,    0]
        [   0,    0,    0,    0,    0,    0,    0,    0]
        [   0,    0, True,    0,    0, True,    0,    0]
        [   0,    0,    0, True,    0,    0,    0,    0]
    ]
```

One more issue that needs to be addressed when using fixed-length segments is
the computation of the advantage (or return). Vanilla policy gradient estimates
the return using a Monte-Carlo estimate, which requires a full-episode rollout
giving all of the rewards from the given episode. However, in order to compute
the advantage $A(s,a)$ for a (state, action) pair from the fixed-length segment
we need an advantage estimator that does not look beyond timestep $T$. That means
we need to bootstrap the estimation at timestep $T$ by using, for example, an
approximation of the value function:

```math
R(s_t) = r_t + r_{t+1} + \cdots + r_{T-2} + V(s_{T-1}).
```

The estimator used is a truncated version of the generalized advantage estimator
introduced in the [paper](https://arxiv.org/abs/1506.02438)
*"High-Dimensional Continuous Control Using Generalized Advantage Estimation"*
by Schulman et. el.:

```math
A(s_t, a_t) = \delta_t + (\gamma \lambda) \delta_{t+1} +
\cdots + (\gamma \lambda)^{T-t+1} \delta_{T-1},
```

where $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$. Here $V_\phi(s)$
is a neural network approximating the value function.


## WEIGHT UPDATES
During the learning phase we optimize the objective for $K$ epochs by performing
mini-batch updates of the policy weights with batch size $B$ using the Adam
optimizer.

At every iteration, before computing the loss, the advantages are normalized on
the mini-batch level to have zero mean and unit variance. This usually boosts
performance as it stabilizes the training of the neural network. This
modification makes use of a constant baseline for all (state, action) pairs in
the batch and effectively rescales the learning rate by a factor of $1 / \sigma$.

```python3
adv = (adv - adv.mean()) / adv.std()
```

The loss is further augmented with an entropy regularization term calculated
over the mini-batch. Trying to maximize the entropy has the effect of pushing
the policy distribution to be more random, preventing it from becoming a delta
function and thus increasing exploration during training.

```python3
entropy_loss = distributions.Categorical(pi(obs)).entropy()
```

Finally, at the end of every epoch we check the *KL divergence* between the
newest and the original policy and stop the learning phase if the threshold is
reached.

```python3
logp_old = pi(obs) # compute the log prob with the old weights
for _ in range(K):
    for o, a, ad, lp_old in minibatch_loader(zip(obs, acts, adv, logp_old)):
        adv = (adv - adv.mean()) / adv.std()  # normalize on the mini-batch level
        logits = pi(o)
        logp = -F.cross_entropy(logits, a, reduction="none")
        rho = (logp - lp_old).exp()
        loss = min(rho * ad, clip(rho, 1-eps, 1+eps) * ad)

        entropy_loss = distributions.Categorical(pi(obs)).entropy()
        loss = -loss.mean() - c * entropy_loss # c ~ [0.01, 0.1]

        optim.zero_grad()
        loss.backward()
        optim.step()

    logp = pi(obs) # compute the log prob with the newest weights
    KL = (logp - logp_old).mean()
    if KL > threshold:
        break
```

In addition to optimizing the policy we also need to optimize the weights of
the value network. The value network is optimized by minimizing the mean squared
error between the value net predictions and the calculated returns. Just like
the clipped objective for the policy network, we also clip the value loss before
updating the parameters:

```math
V_{CLIP} = clip(V_\phi, V_{\phi_{old}}-\epsilon, V_{\phi_{old}}-\epsilon)

L^V = max[(V_\phi - V_{tgt})^2, (V_{CLIP} - V_{tgt})^2].
```

And again we perform $K$ epochs of mini-batch updates with batch size $B$ using
the Adam optimizer.
