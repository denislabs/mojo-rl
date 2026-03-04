"""GPU kernels for continuous control RL algorithms (DDPG, TD3, SAC, A2C).

## TD Target Computation
- td_target_continuous_kernel: DDPG target — r + γ * Q_t(s',a') * (1-done)
- td_target_min_twin_kernel:   TD3/SAC target — r + γ * min(Q1,Q2) * (1-done) [- α*log_π for SAC]

## Actor Gradient Extraction
- actor_grad_from_critic_kernel: Extract ∂Q/∂a from critic's input gradient

## Noise
- add_gaussian_noise_kernel: Add clipped Gaussian noise to actions (TD3 exploration)

## SAC Reparameterization
- sac_reparameterize_kernel: a = tanh(μ + exp(logσ)*ε), log_prob with Jacobian correction

## A2C GPU
- a2c_gae_kernel:            GAE advantages + returns (per-env backward accumulation)
- a2c_softmax_sample_kernel: Softmax action sampling + log-prob for parallel envs
"""

from std.gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from std.math import exp, log, tanh, sqrt, max, min, pi, cos

from .random import gaussian_noise_gpu, random_uniform


# =============================================================================
# TD Target Computation
# =============================================================================


@always_inline
fn td_target_continuous_kernel[
    dtype: DType,
    BATCH: Int,
](
    td_targets: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    next_q_values: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    dones: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    gamma: Scalar[dtype],
):
    """Compute DDPG TD targets: y = r + γ * Q_target(s', a') * (1 - done).

    One thread per batch sample.

    Args:
        td_targets:    Output TD targets [BATCH].
        rewards:       Sampled rewards [BATCH].
        next_q_values: Q-values from target critic at next states [BATCH].
        dones:         Done flags [BATCH] (1.0 = terminal).
        gamma:         Discount factor.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return

    var one = Scalar[dtype](1.0)
    td_targets[i] = rewards[i] + gamma * next_q_values[i] * (one - dones[i])


@always_inline
fn td_target_min_twin_kernel[
    dtype: DType,
    BATCH: Int,
    use_entropy: Bool,
](
    td_targets: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    rewards: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    q1_values: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    q2_values: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    dones: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    log_probs: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    gamma: Scalar[dtype],
    alpha: Scalar[dtype],
):
    """Compute TD3/SAC TD targets with min(Q1,Q2) twin critics.

    TD3: y = r + γ * min(Q1_t, Q2_t)(s', a'_smoothed) * (1-done)
    SAC: y = r + γ * (min(Q1_t, Q2_t)(s', a') - α * log_π(a'|s')) * (1-done)

    Set use_entropy=False for TD3, use_entropy=True for SAC.
    For TD3, log_probs and alpha are unused (set alpha=0 or use_entropy=False).

    Args:
        td_targets: Output TD targets [BATCH].
        rewards:    Sampled rewards [BATCH].
        q1_values:  Q-values from target critic 1 at next states [BATCH].
        q2_values:  Q-values from target critic 2 at next states [BATCH].
        dones:      Done flags [BATCH] (1.0 = terminal).
        log_probs:  Log-probabilities of next actions (SAC only) [BATCH].
        gamma:      Discount factor.
        alpha:      Entropy coefficient (SAC) or 0.0 (TD3).
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return

    var one = Scalar[dtype](1.0)
    var q_min = q1_values[i] if q1_values[i] < q2_values[i] else q2_values[i]

    @parameter
    if use_entropy:
        # SAC: entropy bonus in target
        td_targets[i] = rewards[i] + gamma * (q_min - alpha * log_probs[i]) * (
            one - dones[i]
        )
    else:
        # TD3: no entropy
        td_targets[i] = rewards[i] + gamma * q_min * (one - dones[i])


# =============================================================================
# Actor Gradient Extraction
# =============================================================================


@always_inline
fn actor_grad_from_critic_kernel[
    dtype: DType,
    BATCH: Int,
    OBS_DIM: Int,
    ACTION_DIM: Int,
](
    d_actor_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    d_critic_in: LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS_DIM + ACTION_DIM), MutAnyOrigin
    ],
):
    """Extract actor output gradients from critic input gradients (∂Q/∂a).

    After running critic backward with gradient [-1/BATCH, ...] (gradient ascent),
    the critic's input gradient d_critic_in has shape [BATCH, OBS_DIM + ACTION_DIM].
    The action portion (columns OBS_DIM:OBS_DIM+ACTION_DIM) is the policy gradient.

    One thread per (batch, action_dim) element.

    Args:
        d_actor_out: Output actor gradient [BATCH, ACTION_DIM].
        d_critic_in: Critic input gradient [BATCH, OBS_DIM + ACTION_DIM].
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= BATCH * ACTION_DIM:
        return

    var b = tid // ACTION_DIM
    var a = tid % ACTION_DIM
    d_actor_out[b, a] = d_critic_in[b, OBS_DIM + a]


# =============================================================================
# Gaussian Noise for Exploration
# =============================================================================


@always_inline
fn add_gaussian_noise_kernel[
    dtype: DType,
    BATCH: Int,
    ACTION_DIM: Int,
](
    noisy_actions: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    actions: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    rng_states: LayoutTensor[
        DType.uint32, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    noise_std: Scalar[dtype],
    noise_clip: Scalar[dtype],
    action_min: Scalar[dtype],
    action_max: Scalar[dtype],
):
    """Add clipped Gaussian exploration noise to actions (TD3-style).

    Each element gets independent noise from N(0, noise_std²), clipped to
    [-noise_clip, noise_clip], then the result is clipped to [action_min, action_max].

    One thread per (batch, action_dim) element.

    Args:
        noisy_actions: Output noisy actions [BATCH, ACTION_DIM].
        actions:       Clean actions from actor [BATCH, ACTION_DIM].
        rng_states:    Per-element RNG states (updated in-place) [BATCH, ACTION_DIM].
        noise_std:     Noise standard deviation.
        noise_clip:    Maximum absolute noise value.
        action_min:    Minimum action value (e.g. -action_scale).
        action_max:    Maximum action value (e.g. +action_scale).
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= BATCH * ACTION_DIM:
        return

    var b = tid // ACTION_DIM
    var a = tid % ACTION_DIM

    var rng = rebind[UInt32](rng_states[b, a])
    var noise_result = gaussian_noise_gpu[dtype](rng)
    var z = noise_result[0]
    var new_rng = noise_result[1]

    # Update RNG state
    rng_states[b, a] = new_rng

    # Scale and clip noise
    var noise = z * noise_std
    if noise < -noise_clip:
        noise = -noise_clip
    if noise > noise_clip:
        noise = noise_clip

    # Apply noise and clip to action range
    var noisy = actions[b, a] + noise
    if noisy < action_min:
        noisy = action_min
    if noisy > action_max:
        noisy = action_max

    noisy_actions[b, a] = noisy


# =============================================================================
# SAC Reparameterization
# =============================================================================


@always_inline
fn sac_reparameterize_kernel[
    dtype: DType,
    BATCH: Int,
    ACTION_DIM: Int,
](
    actions: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    log_probs: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    mean: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    log_std: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    rng_states: LayoutTensor[
        DType.uint32, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    log_std_min: Scalar[dtype],
    log_std_max: Scalar[dtype],
):
    """SAC reparameterization trick with tanh squashing and Jacobian log-prob correction.

    Computes:
        ε ~ N(0, 1)
        z = μ + exp(log_σ) * ε         (reparameterization)
        a = tanh(z)                     (squashed action in (-1, 1))
        log π(a|s) = Σ_j [ -0.5*ε_j² - 0.5*log(2π) - log_σ_j
                            - log(1 - tanh²(z_j)) ]  (change-of-variables)

    One thread per (batch, action_dim) element computes its contribution.
    Log-prob accumulation uses atomic-style per-sample summation (thread b=0..BATCH-1,
    a=0..ACTION_DIM-1 all try to add to log_probs[b]; this is fine for small ACTION_DIM
    since ACTION_DIM threads write to the same log_probs[b] — use serialized sum approach:
    each thread writes to a 2D scratch, then sum is done in second pass).

    NOTE: This kernel uses a two-pass approach:
      Pass 1 (BATCH * ACTION_DIM threads): compute z, a, per-dim log_prob contribution
             and store to a temporary scratch array scratch[BATCH, ACTION_DIM].
      Pass 2 (BATCH threads): sum scratch[b, :] into log_probs[b].

    For simplicity with a single-kernel design, each thread b processes ALL action dims
    serially. This is fine since ACTION_DIM is typically small (≤ 17 for MuJoCo).

    One thread per batch sample.

    Args:
        actions:    Output squashed actions in (-1, 1) [BATCH, ACTION_DIM].
        log_probs:  Output log-probabilities (summed over action dims) [BATCH].
        mean:       Actor output mean [BATCH, ACTION_DIM].
        log_std:    Actor output log_std (before clamping) [BATCH, ACTION_DIM].
        rng_states: Per-(batch, action) RNG states (updated in-place) [BATCH, ACTION_DIM].
        log_std_min: Minimum log_std clamp (e.g. -20).
        log_std_max: Maximum log_std clamp (e.g. 2).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var half_log_2pi = Scalar[dtype](0.9189385332046727)  # 0.5 * log(2π)
    var one = Scalar[dtype](1.0)
    var two = Scalar[dtype](2.0)

    var lp = Scalar[dtype](0.0)

    for a in range(ACTION_DIM):
        # Clamp log_std
        var ls = log_std[b, a]
        if ls < log_std_min:
            ls = log_std_min
        if ls > log_std_max:
            ls = log_std_max

        var std_val = exp(ls)

        # Sample ε ~ N(0, 1) using Box-Muller
        var rng = rebind[UInt32](rng_states[b, a])
        var noise_result = gaussian_noise_gpu[dtype](rng)
        var eps = noise_result[0]
        var new_rng = noise_result[1]
        rng_states[b, a] = new_rng

        # Reparameterize: z = μ + σ * ε
        var z = mean[b, a] + std_val * eps

        # Squash: a = tanh(z)
        var act = tanh(z)
        actions[b, a] = act

        # Log-prob: -0.5*(ε²) - 0.5*log(2π) - log_σ - log(1 - tanh²(z))
        # Numerically stable: log(1 - tanh²(z)) = log(4) + 2*z - 2*log(exp(2z)+1)
        # But simpler and sufficient: use clipped 1 - act²
        var one_minus_tanh2 = one - act * act
        # Clamp to avoid log(0)
        if one_minus_tanh2 < Scalar[dtype](1e-6):
            one_minus_tanh2 = Scalar[dtype](1e-6)

        lp += (
            -Scalar[dtype](0.5) * eps * eps
            - half_log_2pi
            - ls
            - log(one_minus_tanh2)
        )

    log_probs[b] = lp


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
    var gae = Scalar[dtype](0.0)

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
    dtype: DType,
    N_ENVS: Int,
    N_ACTIONS: Int,
](
    actions: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    log_probs: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    logits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS, N_ACTIONS), MutAnyOrigin
    ],
    rng_states: LayoutTensor[
        DType.uint32, Layout.row_major(N_ENVS), MutAnyOrigin
    ],
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
        rng_states: Per-env RNG states (updated in-place) [N_ENVS].
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
    var sum_exp = Scalar[dtype](0.0)
    for k in range(N_ACTIONS):
        sum_exp += exp(logits[env, k] - max_logit)

    # Step 3: sample from Categorical via inverse CDF
    var rng = rebind[UInt32](rng_states[env])
    var u_result = random_uniform[dtype](rng)
    var u = u_result[0]
    rng_states[env] = u_result[1]

    var cum = Scalar[dtype](0.0)
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


# =============================================================================
# Batch Obs-Action Concatenation (DDPG/TD3/SAC critic input construction)
# =============================================================================


@always_inline
fn concat_obs_action_kernel[
    dtype: DType,
    BATCH: Int,
    OBS_DIM: Int,
    ACTION_DIM: Int,
](
    dst: LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS_DIM + ACTION_DIM), MutAnyOrigin
    ],
    obs: LayoutTensor[dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin],
    act: LayoutTensor[dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin],
):
    """Concatenate [BATCH, OBS_DIM] and [BATCH, ACTION_DIM] → [BATCH, OBS+ACT].

    Used by DDPG/TD3/SAC to build critic inputs (obs ‖ action) on GPU.
    One thread per output element.

    Args:
        dst: Output tensor [BATCH, OBS_DIM + ACTION_DIM].
        obs: Observations [BATCH, OBS_DIM].
        act: Actions [BATCH, ACTION_DIM].
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total = BATCH * (OBS_DIM + ACTION_DIM)
    if tid >= total:
        return
    var b = tid // (OBS_DIM + ACTION_DIM)
    var c = tid % (OBS_DIM + ACTION_DIM)
    if c < OBS_DIM:
        dst[b, c] = obs[b, c]
    else:
        dst[b, c] = act[b, c - OBS_DIM]


# =============================================================================
# DDPG Action Kernels (scale + optional exploration noise)
# =============================================================================


@always_inline
fn scale_clip_actions_kernel[
    dtype: DType,
    BATCH: Int,
    ACTION_DIM: Int,
](
    actions_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    raw_actions: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    action_scale: Scalar[dtype],
):
    """Scale tanh actor output by action_scale and clip to [-scale, scale].

    Used for greedy (deterministic) action selection on GPU.
    One thread per (batch, action) element.

    Args:
        actions_out: Scaled and clipped output [BATCH, ACTION_DIM].
        raw_actions: Tanh actor output in [-1, 1] [BATCH, ACTION_DIM].
        action_scale: Scaling factor (action range = [-scale, scale]).
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= BATCH * ACTION_DIM:
        return
    var b = tid // ACTION_DIM
    var a = tid % ACTION_DIM
    var val = raw_actions[b, a] * action_scale
    if val > action_scale:
        val = action_scale
    elif val < -action_scale:
        val = -action_scale
    actions_out[b, a] = val


@always_inline
fn ddpg_exploration_kernel[
    dtype: DType,
    BATCH: Int,
    ACTION_DIM: Int,
](
    actions_out: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    raw_actions: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    rng_states: LayoutTensor[
        DType.uint32, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    noise_std: Scalar[dtype],
    action_scale: Scalar[dtype],
):
    """Scale actor output and add Gaussian exploration noise (DDPG-style).

    action = clip(raw * action_scale + noise_std * action_scale * N(0,1),
                  -action_scale, +action_scale)

    One thread per (batch, action) element.

    Args:
        actions_out: Output noisy actions [BATCH, ACTION_DIM].
        raw_actions: Tanh actor output in [-1, 1] [BATCH, ACTION_DIM].
        rng_states:  Per-element RNG states (updated in-place) [BATCH, ACTION_DIM].
        noise_std:   Noise std relative to action scale (e.g. 0.1).
        action_scale: Action range bound (output clipped to [-scale, scale]).
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= BATCH * ACTION_DIM:
        return
    var b = tid // ACTION_DIM
    var a = tid % ACTION_DIM
    var rng = rebind[UInt32](rng_states[b, a])
    var noise_result = gaussian_noise_gpu[dtype](rng)
    var z = noise_result[0]
    rng_states[b, a] = noise_result[1]
    var val = raw_actions[b, a] * action_scale + noise_std * action_scale * z
    if val > action_scale:
        val = action_scale
    elif val < -action_scale:
        val = -action_scale
    actions_out[b, a] = val


# =============================================================================
# TD Critic MSE Gradient
# =============================================================================


@always_inline
fn td_mse_grad_kernel[
    dtype: DType,
    BATCH: Int,
](
    q_grad: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
    q: LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin],
    targets: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
):
    """MSE loss backward for scalar TD critic: q_grad[i,0] = 2*(q-target)/BATCH.

    One thread per batch element.

    Args:
        q_grad:  Output gradient [BATCH, 1] (written).
        q:       Critic output Q-values [BATCH, 1].
        targets: TD targets [BATCH].
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH:
        return
    q_grad[i, 0] = (
        Scalar[dtype](2.0) * (q[i, 0] - targets[i]) / Scalar[dtype](BATCH)
    )
