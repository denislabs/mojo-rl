"""A2C-specific GPU kernels.

## A2C GPU
- a2c_gae_kernel:            GAE advantages + returns (per-env backward accumulation)
- a2c_softmax_sample_kernel: Softmax action sampling + log-prob for parallel envs
"""

from std.gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from std.math import exp, log


# =============================================================================
# A2C GAE (Generalized Advantage Estimation)
# =============================================================================


@always_inline
fn a2c_gae_kernel[
    dtype: DType,
    N_ENVS: Int,
    ROLLOUT: Int,
](
    advantages: LayoutTensor[
        dtype, Layout.row_major(ROLLOUT, N_ENVS), MutAnyOrigin
    ],
    returns: LayoutTensor[
        dtype, Layout.row_major(ROLLOUT, N_ENVS), MutAnyOrigin
    ],
    rewards: LayoutTensor[
        dtype, Layout.row_major(ROLLOUT, N_ENVS), MutAnyOrigin
    ],
    dones: LayoutTensor[dtype, Layout.row_major(ROLLOUT, N_ENVS), MutAnyOrigin],
    values: LayoutTensor[
        dtype, Layout.row_major(ROLLOUT + 1, N_ENVS), MutAnyOrigin
    ],
    gamma: Scalar[dtype],
    gae_lambda: Scalar[dtype],
):
    """Compute GAE advantages and discounted returns for A2C.

    Backward accumulation per environment:
        δ_t = r_t + γ * V(s_{t+1}) * (1-done_t) - V(s_t)
        A_t = δ_t + γ * λ * A_{t+1} * (1-done_t)
        R_t = A_t + V(s_t)  [= V(s_t) + δ_t + γλA_{t+1}(1-done)]

    One thread per environment (handles all ROLLOUT steps serially backwards).

    Args:
        advantages: Output advantages [ROLLOUT, N_ENVS].
        returns:    Output discounted returns [ROLLOUT, N_ENVS].
        rewards:    Step rewards [ROLLOUT, N_ENVS].
        dones:      Done flags [ROLLOUT, N_ENVS].
        values:     Value estimates [ROLLOUT+1, N_ENVS] (index ROLLOUT = bootstrap value).
        gamma:      Discount factor.
        gae_lambda: GAE lambda parameter.
    """
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= N_ENVS:
        return

    var one = Scalar[dtype](1.0)
    var gae: values.element_type = 0.0

    # Backward accumulation over time steps
    for t_rev in range(ROLLOUT):
        var t = ROLLOUT - 1 - t_rev
        var done_t = dones[t, env]
        var mask = one - done_t

        # TD residual
        var delta = (
            rewards[t, env] + gamma * values[t + 1, env] * mask - values[t, env]
        )

        # GAE: A_t = δ_t + γλ * A_{t+1} * (1-done)
        gae = delta + gamma * gae_lambda * gae * mask

        advantages[t, env] = gae
        returns[t, env] = gae + values[t, env]


# =============================================================================
# A2C Softmax Action Sampling
# =============================================================================


@always_inline
fn a2c_softmax_sample_kernel[
    dtype: DType where dtype.is_floating_point(),
    N_ENVS: Int,
    N_ACTIONS: Int,
](
    actions: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    log_probs: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    logits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS, N_ACTIONS), MutAnyOrigin
    ],
    seed: Scalar[DType.uint32],
):
    """Sample discrete actions from softmax distribution and compute log-probabilities.

    For each environment:
        probs = softmax(logits)
        action ~ Categorical(probs)
        log_prob = log(probs[action])

    Uses numerically stable softmax: subtract max(logits) before exp.
    Sampling via inverse-CDF (linear scan over cumulative probabilities).

    One thread per environment.

    Args:
        actions:    Output sampled actions [N_ENVS].
        log_probs:  Output log-probabilities [N_ENVS].
        logits:     Input logits from actor [N_ENVS, N_ACTIONS].
        seed:       RNG seed for action sampling.
    """
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= N_ENVS:
        return

    # Step 1: find max logit for numerical stability
    var max_logit = logits[env, 0]
    for k in range(1, N_ACTIONS):
        var l = logits[env, k]
        if l > max_logit:
            max_logit = l

    # Step 2: compute exp(logit - max) and sum
    var sum_exp: logits.element_type = 0.0
    for k in range(N_ACTIONS):
        sum_exp += exp(logits[env, k] - max_logit)

    # Step 3: sample from Categorical via inverse CDF
    from std.random.philox import Random as PhiloxRandom
    var rng = PhiloxRandom(
        seed=UInt64(seed) * UInt64(N_ENVS) + UInt64(env), offset=0
    )
    var rand_vals = rng.step_uniform()
    var u = Scalar[dtype](rand_vals[0])

    var cum: logits.element_type = 0.0
    var sampled_action = N_ACTIONS - 1  # fallback
    for k in range(N_ACTIONS):
        var prob = exp(logits[env, k] - max_logit) / sum_exp
        cum += prob
        if u <= cum:
            sampled_action = k
            break

    actions[env] = Scalar[dtype](sampled_action)

    # Step 4: compute log_prob = log(prob[sampled_action])
    var logit_a = logits[env, sampled_action]
    # log(softmax(a)) = logit_a - log(sum_exp) - max_logit (which cancels)
    log_probs[env] = logit_a - max_logit - log(sum_exp)
