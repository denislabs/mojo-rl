"""MPPI (Model Predictive Path Integral) Planner for TDMPC2.

MPPI plans in latent space over a horizon H by:
  1. Sampling num_samples candidate action sequences from a Gaussian distribution
  2. Rolling out the world model for each candidate
  3. Computing returns (reward + terminal value via min-Q bootstrap)
  4. Updating the action distribution using softmax-weighted elite candidates
  5. Selecting the first action of the best sequence (with optional noise)

Reference: Hansen et al., 2023 — TD-MPC2
"""

from std.math import exp, sqrt, cos, log
from std.random import random_float64

from layout import Layout, LayoutTensor

from nn.constants import dtype
from .world_model import WorldModel, decode_value_batch_scalar


fn plan[
    OBS_DIM: Int,
    ACTION_DIM: Int,
    LATENT_DIM: Int,
    MLP_DIM: Int,
    NUM_BINS: Int,
    NUM_Q: Int,
    SIMPLEX_DIM: Int,
    V_MIN: Float64,
    V_MAX: Float64,
    HORIZON: Int,
    NUM_SAMPLES: Int,
    NUM_PI_TRAJS: Int,
    NUM_ITERATIONS: Int,
](
    z0: InlineArray[Scalar[dtype], LATENT_DIM],
    mut wm: WorldModel[
        OBS_DIM,
        ACTION_DIM,
        LATENT_DIM,
        MLP_DIM,
        NUM_BINS,
        NUM_Q,
        SIMPLEX_DIM,
        V_MIN,
        V_MAX,
    ],
    gamma: Float64,
    temperature: Float64,
    action_scale: Float64 = 1.0,
    deterministic: Bool = False,
    t0: Bool = True,
    mut prev_mean: List[Float64] = List[Float64](),
) -> InlineArray[Scalar[dtype], ACTION_DIM]:
    """MPPI planning in latent space.

    Args:
        z0: Initial latent state [LATENT_DIM].
        wm: World model for rollouts.
        gamma: Discount factor.
        temperature: MPPI softmax temperature.
        action_scale: Action scaling factor (default 1.0 = [-1, 1]).
        deterministic: If True, add no exploration noise (eval mode).
        t0: If True, this is the first timestep of an episode (no warm-start).
        prev_mean: Previous plan's mean [HORIZON * ACTION_DIM] for warm-start.
            Updated in-place with the new plan's mean on return.

    Returns:
        Selected action [ACTION_DIM] in [-action_scale, action_scale].
    """
    # -------------------------------------------------------------------------
    # MPPI parameters
    # -------------------------------------------------------------------------
    comptime TOTAL_SAMPLES = NUM_SAMPLES + NUM_PI_TRAJS
    comptime STD_MIN: Float64 = 0.05
    comptime STD_MAX: Float64 = 2.0

    # -------------------------------------------------------------------------
    # Initialize action distribution
    # Warm-start: shift previous plan's mean forward by 1 step if not t0.
    # mean: [HORIZON, ACTION_DIM]
    # std:  [HORIZON, ACTION_DIM] = 0.5
    # -------------------------------------------------------------------------
    var mean = List[Float64](capacity=HORIZON * ACTION_DIM)
    var std = List[Float64](capacity=HORIZON * ACTION_DIM)
    if not t0 and len(prev_mean) == HORIZON * ACTION_DIM:
        # Shift prev_mean[1:] into mean[:-1], last step = 0
        for t in range(HORIZON - 1):
            for a in range(ACTION_DIM):
                mean.append(prev_mean[(t + 1) * ACTION_DIM + a])
        # Last horizon step: zero (no info from previous plan)
        for _ in range(ACTION_DIM):
            mean.append(0.0)
    else:
        for _ in range(HORIZON * ACTION_DIM):
            mean.append(0.0)
    for _ in range(HORIZON * ACTION_DIM):
        std.append(0.5)

    # -------------------------------------------------------------------------
    # Storage for candidate trajectories
    # -------------------------------------------------------------------------
    # actions[s, t, a] = action for sample s at step t, dimension a
    var actions = List[Float64](capacity=TOTAL_SAMPLES * HORIZON * ACTION_DIM)
    for _ in range(TOTAL_SAMPLES * HORIZON * ACTION_DIM):
        actions.append(0.0)

    # returns[s] = discounted return for sample s
    var returns = List[Float64](capacity=TOTAL_SAMPLES)
    for _ in range(TOTAL_SAMPLES):
        returns.append(0.0)

    # -------------------------------------------------------------------------
    # Main MPPI iterations
    # -------------------------------------------------------------------------
    # Declare weights outside loop so it's accessible for action selection
    var weights = List[Float64](capacity=TOTAL_SAMPLES)

    for _iter in range(NUM_ITERATIONS):
        # Step 1: Sample NUM_PI_TRAJS trajectories from the learned policy
        for s in range(NUM_PI_TRAJS):
            var z_curr = InlineArray[Scalar[dtype], LATENT_DIM](
                uninitialized=True
            )
            for i in range(LATENT_DIM):
                z_curr[i] = z0[i]

            for t in range(HORIZON):
                # Get policy mean (deterministic action from learned policy)
                var z_in = InlineArray[Scalar[dtype], LATENT_DIM](
                    uninitialized=True
                )
                for i in range(LATENT_DIM):
                    z_in[i] = z_curr[i]
                var pi_mean = InlineArray[Scalar[dtype], ACTION_DIM](
                    uninitialized=True
                )
                var pi_log_std = InlineArray[Scalar[dtype], ACTION_DIM](
                    uninitialized=True
                )
                var z_in_v = LayoutTensor[
                    dtype, Layout.row_major(1, LATENT_DIM), MutAnyOrigin
                ](z_in.unsafe_ptr())
                var pi_mean_v = LayoutTensor[
                    dtype, Layout.row_major(1, ACTION_DIM), MutAnyOrigin
                ](pi_mean.unsafe_ptr())
                var pi_log_std_v = LayoutTensor[
                    dtype, Layout.row_major(1, ACTION_DIM), MutAnyOrigin
                ](pi_log_std.unsafe_ptr())
                wm.policy_forward[1](z_in_v, pi_mean_v, pi_log_std_v)

                var base = s * HORIZON * ACTION_DIM + t * ACTION_DIM
                for a in range(ACTION_DIM):
                    # Sample from policy with some noise
                    var noise = _gaussian_sample() * 0.1
                    var act = Float64(pi_mean[a]) + noise
                    # Clamp to [-1, 1]
                    act = _clamp(act, -1.0, 1.0)
                    actions[base + a] = act

                # Advance latent state
                var za = InlineArray[Scalar[dtype], LATENT_DIM + ACTION_DIM](
                    uninitialized=True
                )
                for i in range(LATENT_DIM):
                    za[i] = z_curr[i]
                for a in range(ACTION_DIM):
                    za[LATENT_DIM + a] = Scalar[dtype](actions[base + a])
                var za_v = LayoutTensor[
                    dtype,
                    Layout.row_major(1, LATENT_DIM + ACTION_DIM),
                    MutAnyOrigin,
                ](za.unsafe_ptr())
                var z_curr_v = LayoutTensor[
                    dtype, Layout.row_major(1, LATENT_DIM), MutAnyOrigin
                ](z_curr.unsafe_ptr())
                wm.dynamics_forward[1](za_v, z_curr_v)

        # Step 2: Sample NUM_SAMPLES trajectories from the MPPI distribution
        for s in range(NUM_PI_TRAJS, TOTAL_SAMPLES):
            for t in range(HORIZON):
                for a in range(ACTION_DIM):
                    var mu = mean[t * ACTION_DIM + a]
                    var sigma = std[t * ACTION_DIM + a]
                    var noise = _gaussian_sample()
                    var act = mu + sigma * noise
                    act = _clamp(act, -1.0, 1.0)
                    var base = s * HORIZON * ACTION_DIM + t * ACTION_DIM
                    actions[base + a] = act

        # Step 3: Evaluate all trajectories
        for s in range(TOTAL_SAMPLES):
            var z_curr = InlineArray[Scalar[dtype], LATENT_DIM](
                uninitialized=True
            )
            for i in range(LATENT_DIM):
                z_curr[i] = z0[i]

            var G: Float64 = 0.0
            var discount: Float64 = 1.0

            for t in range(HORIZON):
                var base = s * HORIZON * ACTION_DIM + t * ACTION_DIM

                # Build z_a for this step
                var za = InlineArray[Scalar[dtype], LATENT_DIM + ACTION_DIM](
                    uninitialized=True
                )
                for i in range(LATENT_DIM):
                    za[i] = z_curr[i]
                for a in range(ACTION_DIM):
                    za[LATENT_DIM + a] = Scalar[dtype](actions[base + a])

                # Predict reward
                var rew_logits = InlineArray[Scalar[dtype], NUM_BINS](
                    uninitialized=True
                )
                var za_v2 = LayoutTensor[
                    dtype,
                    Layout.row_major(1, LATENT_DIM + ACTION_DIM),
                    MutAnyOrigin,
                ](za.unsafe_ptr())
                var rew_logits_v = LayoutTensor[
                    dtype, Layout.row_major(1, NUM_BINS), MutAnyOrigin
                ](rew_logits.unsafe_ptr())
                wm.reward_forward[1](za_v2, rew_logits_v)
                var rew_logits_f32 = InlineArray[Float32, NUM_BINS](
                    uninitialized=True
                )
                for i in range(NUM_BINS):
                    rew_logits_f32[i] = Float32(rew_logits[i])
                var reward_val = Float64(
                    decode_value_batch_scalar[NUM_BINS](rew_logits_f32, wm.bins)
                )
                G += discount * reward_val
                discount *= gamma

                # Advance latent state
                var za_v3 = LayoutTensor[
                    dtype,
                    Layout.row_major(1, LATENT_DIM + ACTION_DIM),
                    MutAnyOrigin,
                ](za.unsafe_ptr())
                var z_curr_v2 = LayoutTensor[
                    dtype, Layout.row_major(1, LATENT_DIM), MutAnyOrigin
                ](z_curr.unsafe_ptr())
                wm.dynamics_forward[1](za_v3, z_curr_v2)

            # Bootstrap terminal value: min_Q(z_H, π(z_H))
            var pi_mean = InlineArray[Scalar[dtype], ACTION_DIM](
                uninitialized=True
            )
            var pi_log_std = InlineArray[Scalar[dtype], ACTION_DIM](
                uninitialized=True
            )
            var z_curr_pv = LayoutTensor[
                dtype, Layout.row_major(1, LATENT_DIM), MutAnyOrigin
            ](z_curr.unsafe_ptr())
            var pi_mean_pv = LayoutTensor[
                dtype, Layout.row_major(1, ACTION_DIM), MutAnyOrigin
            ](pi_mean.unsafe_ptr())
            var pi_log_std_pv = LayoutTensor[
                dtype, Layout.row_major(1, ACTION_DIM), MutAnyOrigin
            ](pi_log_std.unsafe_ptr())
            wm.policy_forward[1](z_curr_pv, pi_mean_pv, pi_log_std_pv)

            var za_terminal = InlineArray[
                Scalar[dtype], LATENT_DIM + ACTION_DIM
            ](uninitialized=True)
            for i in range(LATENT_DIM):
                za_terminal[i] = z_curr[i]
            for a in range(ACTION_DIM):
                # Clamp actions for terminal value estimation
                var act = Float64(pi_mean[a])
                act = _clamp(act, -1.0, 1.0)
                za_terminal[LATENT_DIM + a] = Scalar[dtype](act)

            var terminal_values = InlineArray[Scalar[dtype], 1](
                uninitialized=True
            )
            var za_term_v = LayoutTensor[
                dtype,
                Layout.row_major(1, LATENT_DIM + ACTION_DIM),
                MutAnyOrigin,
            ](za_terminal.unsafe_ptr())
            var term_val_v = LayoutTensor[
                dtype, Layout.row_major(1, 1), MutAnyOrigin
            ](terminal_values.unsafe_ptr())
            wm.q_min_forward[1](za_term_v, term_val_v, True)  # use targets
            G += discount * Float64(terminal_values[0])

            returns[s] = G

        # Step 4: Softmax weighting of elites
        # Find max return for numerical stability
        var max_return = returns[0]
        for s in range(1, TOTAL_SAMPLES):
            if returns[s] > max_return:
                max_return = returns[s]

        # Compute weights: w_s = exp(temperature * (G_s - max_G))
        weights = List[Float64](capacity=TOTAL_SAMPLES)
        var sum_w: Float64 = 0.0
        for s in range(TOTAL_SAMPLES):
            var w = exp(temperature * (returns[s] - max_return))
            weights.append(w)
            sum_w += w

        # Normalize
        if sum_w < 1e-10:
            sum_w = 1e-10
        for s in range(TOTAL_SAMPLES):
            weights[s] = weights[s] / sum_w

        # Step 5: Update mean and std
        for t in range(HORIZON):
            for a in range(ACTION_DIM):
                var new_mean: Float64 = 0.0
                for s in range(TOTAL_SAMPLES):
                    var base = s * HORIZON * ACTION_DIM + t * ACTION_DIM
                    new_mean += weights[s] * actions[base + a]
                mean[t * ACTION_DIM + a] = new_mean

                var new_var: Float64 = 0.0
                for s in range(TOTAL_SAMPLES):
                    var base = s * HORIZON * ACTION_DIM + t * ACTION_DIM
                    var diff = actions[base + a] - new_mean
                    new_var += weights[s] * diff * diff

                var new_std = sqrt(new_var + 1e-8)
                new_std = _clamp(new_std, STD_MIN, STD_MAX)
                std[t * ACTION_DIM + a] = new_std

    # -------------------------------------------------------------------------
    # Store final mean for warm-starting the next timestep
    # -------------------------------------------------------------------------
    prev_mean = List[Float64](capacity=HORIZON * ACTION_DIM)
    for i in range(HORIZON * ACTION_DIM):
        prev_mean.append(mean[i])

    # -------------------------------------------------------------------------
    # Action Selection: weighted random sampling (multinomial) over scores
    # Reference: rand_idx = torch.multinomial(score.squeeze(1).cpu(), 1)
    # -------------------------------------------------------------------------
    var selected_s = _weighted_sample(weights, TOTAL_SAMPLES)

    var result = InlineArray[Scalar[dtype], ACTION_DIM](uninitialized=True)
    for a in range(ACTION_DIM):
        var act = actions[selected_s * HORIZON * ACTION_DIM + a]
        # Add exploration noise scaled by current std (not fixed noise)
        if not deterministic:
            act += _gaussian_sample() * std[a]
        act = _clamp(act * action_scale, -action_scale, action_scale)
        result[a] = Scalar[dtype](act)

    return result^


@always_inline
fn _weighted_sample(weights: List[Float64], n: Int) -> Int:
    """Multinomial sampling: draw one index proportional to weights.

    Equivalent to torch.multinomial(weights, 1). Weights must be
    non-negative and sum to ~1 (already normalized by MPPI softmax).
    """
    var u = random_float64()
    var cumsum: Float64 = 0.0
    for i in range(n):
        cumsum += weights[i]
        if u <= cumsum:
            return i
    # Fallback (rounding): return last
    return n - 1


@always_inline
fn _gaussian_sample() -> Float64:
    """Box-Muller transform to generate a standard normal sample."""
    var u1 = random_float64()
    var u2 = random_float64()
    # Avoid log(0)
    if u1 < 1e-10:
        u1 = 1e-10
    var z = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2)
    return z


@always_inline
fn _clamp(x: Float64, lo: Float64, hi: Float64) -> Float64:
    """Clamp x to [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
