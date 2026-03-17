"""MPPI (Model Predictive Path Integral) Planner for TDMPC2.

MPPI plans in latent space over a horizon H by:
  1. Sampling num_samples candidate action sequences from a Gaussian distribution
  2. Rolling out the world model for each candidate
  3. Computing returns (reward + terminal value via min-Q bootstrap)
  4. Updating the action distribution using softmax-weighted elite candidates
  5. Selecting the first action of the best sequence (with optional noise)

Provides both CPU (`plan()`) and GPU-batched (`plan_gpu()`) implementations.

Reference: Hansen et al., 2023 — TD-MPC2
"""

from std.math import exp, sqrt, cos, log
from std.random import random_float64
from std.sys import has_nvidia_gpu_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream, HostBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.model import Model
from mojo_rl.nn.optimizer import Optimizer
from mojo_rl.nn.training import Network
from .world_model import WorldModel, decode_value_batch_scalar
from .state import MPPIGPUBuffers, BatchedMPPIGPUBuffers
from .kernels import (
    mppi_broadcast_z0_kernel,
    mppi_sample_actions_kernel,
    mppi_accumulate_reward_kernel,
    mppi_add_terminal_value_kernel,
    mppi_copy_z_kernel,
    mppi_broadcast_z0_batched_kernel,
    mppi_sample_actions_batched_kernel,
    mppi_softmax_weights_kernel,
    mppi_weighted_mean_std_kernel,
    tdmpc2_build_za_kernel,
    tdmpc2_apply_tanh_build_za_deterministic_kernel,
    tdmpc2_q_decode_kernel,
    tdmpc2_decode_and_min_kernel,
    tdmpc2_min5_q_values_kernel,
    tdmpc2_zero_kernel,
    q5_concat_params_kernel,
    q5_replicate_input_kernel,
    q5_grouped_matmul_bias_kernel,
    q5_grouped_ln_mish_kernel,
    q5_decode_min_kernel,
    # Fused kernels
    mppi_broadcast_z0_zero_returns_batched_kernel,
    mppi_sample_actions_build_za_batched_kernel,
    mppi_accum_reward_copy_z_kernel,
)


fn plan[
    OBS_DIM: Int,
    ACTION_DIM: Int,
    LATENT_DIM: Int,
    MLP_DIM: Int,
    ENC_DIM: Int,
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
        ENC_DIM,
        NUM_BINS,
        NUM_Q,
        SIMPLEX_DIM,
        V_MIN,
        V_MAX,
    ],
    gamma: Float64,
    temperature: Float64,
    mut prev_mean: List[Float64],
    action_scale: Float64 = 1.0,
    deterministic: Bool = False,
    t0: Bool = True,
) -> InlineArray[Scalar[dtype], ACTION_DIM]:
    """MPPI planning in latent space.

    Args:
        z0: Initial latent state [LATENT_DIM].
        wm: World model for rollouts.
        gamma: Discount factor.
        temperature: MPPI softmax temperature.
        prev_mean: Previous plan's mean [HORIZON * ACTION_DIM] for warm-start.
        action_scale: Action scaling factor (default 1.0 = [-1, 1]).
        deterministic: If True, add no exploration noise (eval mode).
        t0: If True, this is the first timestep of an episode (no warm-start).

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


# =============================================================================
# GPU-Batched MPPI Planning
# =============================================================================


fn plan_gpu[
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
    # Network types for GPU forward passes
    DynModel: Model,
    DynOpt: Optimizer,
    RewModel: Model,
    RewOpt: Optimizer,
    PolModel: Model,
    PolOpt: Optimizer,
    QModel: Model,
    QOpt: Optimizer,
](
    ctx: DeviceContext,
    # z0 already encoded on GPU [1, LATENT_DIM]
    z0_tensor: LayoutTensor[
        dtype, Layout.row_major(1, LATENT_DIM), MutAnyOrigin
    ],
    # Network params on GPU (from TDMPC2GPUState)
    dyn_params: LayoutTensor[
        dtype, Layout.row_major(DynModel.PARAM_SIZE), MutAnyOrigin
    ],
    rew_params: LayoutTensor[
        dtype, Layout.row_major(RewModel.PARAM_SIZE), MutAnyOrigin
    ],
    pol_params: LayoutTensor[
        dtype, Layout.row_major(PolModel.PARAM_SIZE), MutAnyOrigin
    ],
    qt_param_ptrs: InlineArray[
        UnsafePointer[Scalar[dtype], MutAnyOrigin], NUM_Q
    ],
    bins_tensor: LayoutTensor[dtype, Layout.row_major(NUM_BINS), MutAnyOrigin],
    # MPPI GPU buffers
    mut mb: MPPIGPUBuffers[
        DynModel,  # EncModel placeholder (not used but needed for type)
        DynOpt,  # EncOpt placeholder
        DynModel,
        DynOpt,
        RewModel,
        RewOpt,
        PolModel,
        PolOpt,
        QModel,
        QOpt,
        ACTION_DIM,
        LATENT_DIM,
        NUM_BINS,
        NUM_SAMPLES,
        NUM_PI_TRAJS,
        HORIZON,
    ],
    # Hyperparams
    gamma: Float64,
    temperature: Float64,
    mut prev_mean: List[Float64],
    action_scale: Float64 = 1.0,
    deterministic: Bool = False,
    t0: Bool = True,
    rng_base_seed: UInt32 = 42,
) raises -> InlineArray[Scalar[dtype], ACTION_DIM]:
    """GPU-batched MPPI planning in latent space.

    All TOTAL_SAMPLES (NUM_SAMPLES + NUM_PI_TRAJS) trajectory rollouts
    are batched through the world model on GPU in parallel.
    Only the lightweight distribution update (softmax weights, mean/std)
    syncs back to CPU per iteration.

    Args:
        ctx: GPU device context.
        z0_tensor: Encoded observation [1, LATENT_DIM] on GPU.
        dyn_params: Dynamics network params on GPU.
        rew_params: Reward network params on GPU.
        pol_params: Policy network params on GPU.
        qt_param_ptrs: Target Q-network param pointers (NUM_Q).
        bins_tensor: Distribution bin centers [NUM_BINS] on GPU.
        mb: MPPI GPU buffers (workspace + data).
        gamma: Discount factor.
        temperature: MPPI softmax temperature.
        prev_mean: Previous plan mean [H * ACT] for warm-start (updated in-place).
        action_scale: Action scaling factor (default 1.0).
        deterministic: If True, no exploration noise (eval mode).
        t0: If True, first timestep of episode (no warm-start).
        rng_base_seed: Base seed for RNG (should vary per call).

    Returns:
        Selected action [ACTION_DIM] in [-action_scale, action_scale].
    """
    # ─── Compile-time constants ────────────────────────────────────────────
    comptime TOTAL_SAMPLES = NUM_SAMPLES + NUM_PI_TRAJS
    comptime ZA_DIM = LATENT_DIM + ACTION_DIM
    comptime POL_OUT = PolModel.OUT_DIM
    comptime STD_MIN: Float64 = 0.05
    comptime STD_MAX: Float64 = 2.0
    comptime MPPI_BLOCKS = (TOTAL_SAMPLES + TPB - 1) // TPB
    comptime RETURNS_SIZE = TOTAL_SAMPLES
    comptime RETURNS_BLOCKS = (RETURNS_SIZE + TPB - 1) // TPB

    comptime DynNet = Network[DynModel, DynOpt]
    comptime RewNet = Network[RewModel, RewOpt]
    comptime PolNet = Network[PolModel, PolOpt]
    comptime QNet = Network[QModel, QOpt]

    # ─── LayoutTensor views over MPPI GPU buffers ──────────────────────────
    var z_tensor = LayoutTensor[
        dtype, Layout.row_major(TOTAL_SAMPLES, LATENT_DIM), MutAnyOrigin
    ](mb.z_buf.unsafe_ptr())
    var z_next_tensor = LayoutTensor[
        dtype, Layout.row_major(TOTAL_SAMPLES, LATENT_DIM), MutAnyOrigin
    ](mb.z_next_buf.unsafe_ptr())
    var za_tensor = LayoutTensor[
        dtype, Layout.row_major(TOTAL_SAMPLES, ZA_DIM), MutAnyOrigin
    ](mb.za_buf.unsafe_ptr())
    var act_step_tensor = LayoutTensor[
        dtype, Layout.row_major(TOTAL_SAMPLES, ACTION_DIM), MutAnyOrigin
    ](mb.act_step_buf.unsafe_ptr())
    var all_actions_tensor = LayoutTensor[
        dtype,
        Layout.row_major(TOTAL_SAMPLES * HORIZON * ACTION_DIM),
        MutAnyOrigin,
    ](mb.all_actions_buf.unsafe_ptr())
    var rew_logits_tensor = LayoutTensor[
        dtype, Layout.row_major(TOTAL_SAMPLES, NUM_BINS), MutAnyOrigin
    ](mb.rew_logits_buf.unsafe_ptr())
    var q_logits_tensor = LayoutTensor[
        dtype, Layout.row_major(TOTAL_SAMPLES, NUM_BINS), MutAnyOrigin
    ](mb.q_logits_buf.unsafe_ptr())
    var returns_tensor = LayoutTensor[
        dtype, Layout.row_major(TOTAL_SAMPLES), MutAnyOrigin
    ](mb.returns_buf.unsafe_ptr())
    var q_min_tensor = LayoutTensor[
        dtype, Layout.row_major(TOTAL_SAMPLES), MutAnyOrigin
    ](mb.q_min_buf.unsafe_ptr())
    var pi_out_tensor = LayoutTensor[
        dtype, Layout.row_major(TOTAL_SAMPLES, POL_OUT), MutAnyOrigin
    ](mb.pi_out_buf.unsafe_ptr())
    var mean_tensor = LayoutTensor[
        dtype, Layout.row_major(HORIZON * ACTION_DIM), MutAnyOrigin
    ](mb.mean_buf.unsafe_ptr())
    var std_tensor = LayoutTensor[
        dtype, Layout.row_major(HORIZON * ACTION_DIM), MutAnyOrigin
    ](mb.std_buf.unsafe_ptr())

    # Model-typed tensor views (same underlying buffers, but with Model.IN_DIM
    # / Model.OUT_DIM dimensions so forward_gpu_no_cache type-checks)
    var pol_out_tensor = LayoutTensor[
        dtype,
        Layout.row_major(TOTAL_SAMPLES, PolModel.OUT_DIM),
        MutAnyOrigin,
    ](mb.pi_out_buf.unsafe_ptr())
    var pol_in_tensor = LayoutTensor[
        dtype,
        Layout.row_major(TOTAL_SAMPLES, PolModel.IN_DIM),
        MutAnyOrigin,
    ](mb.z_buf.unsafe_ptr())
    var dyn_in_tensor = LayoutTensor[
        dtype,
        Layout.row_major(TOTAL_SAMPLES, DynModel.IN_DIM),
        MutAnyOrigin,
    ](mb.za_buf.unsafe_ptr())
    var dyn_out_tensor = LayoutTensor[
        dtype,
        Layout.row_major(TOTAL_SAMPLES, DynModel.OUT_DIM),
        MutAnyOrigin,
    ](mb.z_next_buf.unsafe_ptr())
    var rew_in_tensor = LayoutTensor[
        dtype,
        Layout.row_major(TOTAL_SAMPLES, RewModel.IN_DIM),
        MutAnyOrigin,
    ](mb.za_buf.unsafe_ptr())
    var rew_out_tensor = LayoutTensor[
        dtype,
        Layout.row_major(TOTAL_SAMPLES, RewModel.OUT_DIM),
        MutAnyOrigin,
    ](mb.rew_logits_buf.unsafe_ptr())
    var q_in_tensor = LayoutTensor[
        dtype,
        Layout.row_major(TOTAL_SAMPLES, QModel.IN_DIM),
        MutAnyOrigin,
    ](mb.za_buf.unsafe_ptr())
    var q_out_tensor = LayoutTensor[
        dtype,
        Layout.row_major(TOTAL_SAMPLES, QModel.OUT_DIM),
        MutAnyOrigin,
    ](mb.q_logits_buf.unsafe_ptr())

    # ─── Kernel wrappers (compile-time parameterized) ──────────────────────
    comptime broadcast_z0 = mppi_broadcast_z0_kernel[
        dtype, TOTAL_SAMPLES, LATENT_DIM
    ]
    comptime sample_actions = mppi_sample_actions_kernel[
        dtype, TOTAL_SAMPLES, NUM_PI_TRAJS, ACTION_DIM, HORIZON, POL_OUT
    ]
    comptime accum_reward = mppi_accumulate_reward_kernel[
        dtype, TOTAL_SAMPLES, NUM_BINS
    ]
    comptime add_terminal = mppi_add_terminal_value_kernel[dtype, TOTAL_SAMPLES]
    comptime copy_z = mppi_copy_z_kernel[dtype, TOTAL_SAMPLES, LATENT_DIM]
    comptime build_za = tdmpc2_build_za_kernel[
        dtype, TOTAL_SAMPLES, LATENT_DIM, ACTION_DIM
    ]
    comptime tanh_build_za = tdmpc2_apply_tanh_build_za_deterministic_kernel[
        dtype, TOTAL_SAMPLES, ACTION_DIM, LATENT_DIM, POL_OUT
    ]
    comptime q_decode = tdmpc2_q_decode_kernel[dtype, TOTAL_SAMPLES, NUM_BINS]
    comptime decode_min = tdmpc2_decode_and_min_kernel[
        dtype, TOTAL_SAMPLES, NUM_BINS
    ]
    comptime zero_returns = tdmpc2_zero_kernel[dtype, RETURNS_SIZE]

    # ─── Initialize mean/std on CPU, upload to GPU ─────────────────────────
    # Warm-start: shift previous plan's mean forward by 1 step
    for i in range(HORIZON * ACTION_DIM):
        mb.mean_host[i] = Scalar[dtype](0.0)
        mb.std_host[i] = Scalar[dtype](0.5)

    if not t0 and len(prev_mean) == HORIZON * ACTION_DIM:
        for t in range(HORIZON - 1):
            for a in range(ACTION_DIM):
                mb.mean_host[t * ACTION_DIM + a] = Scalar[dtype](
                    prev_mean[(t + 1) * ACTION_DIM + a]
                )
        # Last step stays 0

    ctx.enqueue_copy(mb.mean_buf, mb.mean_host)
    ctx.enqueue_copy(mb.std_buf, mb.std_host)

    # ─── Softmax weights storage (CPU-side) ────────────────────────────────
    var weights = List[Float64](capacity=TOTAL_SAMPLES)

    # ─── Main MPPI iterations ──────────────────────────────────────────────
    for mppi_iter in range(NUM_ITERATIONS):
        var rng_seed = rng_base_seed + UInt32(
            mppi_iter * TOTAL_SAMPLES * HORIZON * ACTION_DIM * 2
        )

        # 1. Broadcast z0 to all samples
        ctx.enqueue_function[broadcast_z0, broadcast_z0](
            z0_tensor,
            z_tensor,
            grid_dim=(MPPI_BLOCKS,),
            block_dim=(TPB,),
        )

        # 2. Zero returns
        ctx.enqueue_function[zero_returns, zero_returns](
            returns_tensor,
            grid_dim=(RETURNS_BLOCKS,),
            block_dim=(TPB,),
        )

        # 3. Horizon rollout (H sequential steps, each fully batched)
        var discount = Scalar[dtype](1.0)
        for t in range(HORIZON):
            var step_seed = rng_seed + UInt32(
                t * TOTAL_SAMPLES * ACTION_DIM + 1
            )

            # 3a. Policy forward on all samples (for policy trajectories)
            PolModel.forward_gpu_no_cache[TOTAL_SAMPLES](
                ctx,
                pol_out_tensor,
                pol_in_tensor,
                pol_params,
                mb.pol_ws_buf,
            )

            # 3b. Sample actions (policy + MPPI distribution)
            ctx.enqueue_function[sample_actions, sample_actions](
                pi_out_tensor,
                mean_tensor,
                std_tensor,
                act_step_tensor,
                all_actions_tensor,
                t,
                Scalar[DType.uint32](step_seed),
                grid_dim=(MPPI_BLOCKS,),
                block_dim=(TPB,),
            )

            # 3c. Build za = [z, action]
            ctx.enqueue_function[build_za, build_za](
                z_tensor,
                act_step_tensor,
                za_tensor,
                grid_dim=(MPPI_BLOCKS,),
                block_dim=(TPB,),
            )

            # 3d. Reward forward
            RewModel.forward_gpu_no_cache[TOTAL_SAMPLES](
                ctx,
                rew_out_tensor,
                rew_in_tensor,
                rew_params,
                mb.rew_ws_buf,
            )

            # 3e. Accumulate discounted reward
            ctx.enqueue_function[accum_reward, accum_reward](
                rew_logits_tensor,
                bins_tensor,
                returns_tensor,
                discount,
                grid_dim=(MPPI_BLOCKS,),
                block_dim=(TPB,),
            )
            discount = discount * Scalar[dtype](gamma)

            # 3f. Dynamics forward (advance z)
            DynModel.forward_gpu_no_cache[TOTAL_SAMPLES](
                ctx,
                dyn_out_tensor,
                dyn_in_tensor,
                dyn_params,
                mb.dyn_ws_buf,
            )

            # 3g. Copy z_next → z for next step
            ctx.enqueue_function[copy_z, copy_z](
                z_tensor,
                z_next_tensor,
                grid_dim=(MPPI_BLOCKS,),
                block_dim=(TPB,),
            )

        # 4. Terminal value: policy → tanh → build za → Q-min
        PolModel.forward_gpu_no_cache[TOTAL_SAMPLES](
            ctx,
            pol_out_tensor,
            pol_in_tensor,
            pol_params,
            mb.pol_ws_buf,
        )

        # Fused tanh(policy_mean) + build za
        ctx.enqueue_function[tanh_build_za, tanh_build_za](
            pi_out_tensor,
            act_step_tensor,  # reused as terminal actions
            z_tensor,
            za_tensor,
            grid_dim=(MPPI_BLOCKS,),
            block_dim=(TPB,),
        )

        # Q1: decode → q_min (initialize)
        var qt1_params = LayoutTensor[
            dtype, Layout.row_major(QModel.PARAM_SIZE), MutAnyOrigin
        ](qt_param_ptrs[0])
        QModel.forward_gpu_no_cache[TOTAL_SAMPLES](
            ctx,
            q_out_tensor,
            q_in_tensor,
            qt1_params,
            mb.q_ws_buf,
        )
        ctx.enqueue_function[q_decode, q_decode](
            q_logits_tensor,
            bins_tensor,
            q_min_tensor,
            grid_dim=(MPPI_BLOCKS,),
            block_dim=(TPB,),
        )

        # Q2..Q5: decode + min update
        for qi in range(1, NUM_Q):
            var qt_params = LayoutTensor[
                dtype, Layout.row_major(QModel.PARAM_SIZE), MutAnyOrigin
            ](qt_param_ptrs[qi])
            QModel.forward_gpu_no_cache[TOTAL_SAMPLES](
                ctx,
                q_out_tensor,
                q_in_tensor,
                qt_params,
                mb.q_ws_buf,
            )
            ctx.enqueue_function[decode_min, decode_min](
                q_logits_tensor,
                bins_tensor,
                q_min_tensor,
                grid_dim=(MPPI_BLOCKS,),
                block_dim=(TPB,),
            )

        # 4b. Add terminal value to returns
        ctx.enqueue_function[add_terminal, add_terminal](
            q_min_tensor,
            returns_tensor,
            discount,
            grid_dim=(MPPI_BLOCKS,),
            block_dim=(TPB,),
        )

        # 5. Download returns + actions to CPU for distribution update
        ctx.enqueue_copy(mb.returns_host, mb.returns_buf)
        ctx.enqueue_copy(mb.all_actions_host, mb.all_actions_buf)
        ctx.synchronize()

        # 6. CPU: softmax weights
        var max_return = Float64(mb.returns_host[0])
        for s in range(1, TOTAL_SAMPLES):
            var v = Float64(mb.returns_host[s])
            if v > max_return:
                max_return = v

        weights = List[Float64](capacity=TOTAL_SAMPLES)
        var sum_w: Float64 = 0.0
        for s in range(TOTAL_SAMPLES):
            var w = exp(
                temperature * (Float64(mb.returns_host[s]) - max_return)
            )
            weights.append(w)
            sum_w += w

        if sum_w < 1e-10:
            sum_w = 1e-10
        for s in range(TOTAL_SAMPLES):
            weights[s] = weights[s] / sum_w

        # 7. CPU: update mean and std
        for t in range(HORIZON):
            for a in range(ACTION_DIM):
                var new_mean: Float64 = 0.0
                for s in range(TOTAL_SAMPLES):
                    var base = s * HORIZON * ACTION_DIM + t * ACTION_DIM
                    new_mean += weights[s] * Float64(
                        mb.all_actions_host[base + a]
                    )
                mb.mean_host[t * ACTION_DIM + a] = Scalar[dtype](new_mean)

                var new_var: Float64 = 0.0
                for s in range(TOTAL_SAMPLES):
                    var base = s * HORIZON * ACTION_DIM + t * ACTION_DIM
                    var diff = Float64(mb.all_actions_host[base + a]) - new_mean
                    new_var += weights[s] * diff * diff

                var new_std = sqrt(new_var + 1e-8)
                new_std = _clamp(new_std, STD_MIN, STD_MAX)
                mb.std_host[t * ACTION_DIM + a] = Scalar[dtype](new_std)

        # 8. Upload updated mean/std to GPU for next iteration
        ctx.enqueue_copy(mb.mean_buf, mb.mean_host)
        ctx.enqueue_copy(mb.std_buf, mb.std_host)

    # ─── Store final mean for warm-starting next timestep ──────────────────
    prev_mean = List[Float64](capacity=HORIZON * ACTION_DIM)
    for i in range(HORIZON * ACTION_DIM):
        prev_mean.append(Float64(mb.mean_host[i]))

    # ─── Action selection: weighted random sampling ────────────────────────
    var selected_s = _weighted_sample(weights, TOTAL_SAMPLES)

    var result = InlineArray[Scalar[dtype], ACTION_DIM](uninitialized=True)
    for a in range(ACTION_DIM):
        var act = Float64(
            mb.all_actions_host[selected_s * HORIZON * ACTION_DIM + a]
        )
        # Add exploration noise scaled by current std
        if not deterministic:
            act += _gaussian_sample() * Float64(mb.std_host[a])
        act = _clamp(act * action_scale, -action_scale, action_scale)
        result[a] = Scalar[dtype](act)

    return result^


# =============================================================================
# Batched GPU MPPI — plans all N_ENVS simultaneously in one GPU call
# =============================================================================


fn plan_gpu_batched[
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
    # Network types for GPU forward passes
    DynModel: Model,
    DynOpt: Optimizer,
    RewModel: Model,
    RewOpt: Optimizer,
    PolModel: Model,
    PolOpt: Optimizer,
    QModel: Model,
    QOpt: Optimizer,
    N_ENVS: Int,
](
    ctx: DeviceContext,
    # z0 for all envs on GPU [N_ENVS, LATENT_DIM]
    z0_tensor: LayoutTensor[
        dtype, Layout.row_major(N_ENVS, LATENT_DIM), MutAnyOrigin
    ],
    # Network params on GPU
    dyn_params: LayoutTensor[
        dtype, Layout.row_major(DynModel.PARAM_SIZE), MutAnyOrigin
    ],
    rew_params: LayoutTensor[
        dtype, Layout.row_major(RewModel.PARAM_SIZE), MutAnyOrigin
    ],
    pol_params: LayoutTensor[
        dtype, Layout.row_major(PolModel.PARAM_SIZE), MutAnyOrigin
    ],
    qt_param_ptrs: InlineArray[
        UnsafePointer[Scalar[dtype], MutAnyOrigin], NUM_Q
    ],
    bins_tensor: LayoutTensor[dtype, Layout.row_major(NUM_BINS), MutAnyOrigin],
    # Batched MPPI GPU buffers
    mut mb: BatchedMPPIGPUBuffers[
        DynModel,  # EncModel placeholder
        DynOpt,  # EncOpt placeholder
        DynModel,
        DynOpt,
        RewModel,
        RewOpt,
        PolModel,
        PolOpt,
        QModel,
        QOpt,
        ACTION_DIM,
        LATENT_DIM,
        NUM_BINS,
        NUM_SAMPLES,
        NUM_PI_TRAJS,
        HORIZON,
        N_ENVS,
        MLP_DIM,
        NUM_Q,
    ],
    # Hyperparams
    gamma: Float64,
    temperature: Float64,
    # Per-env warm-start state (mutated in-place)
    mut env_prev_means: List[List[Float64]],
    mut env_t0_flags: List[Bool],
    # Host buffer to write selected actions [N_ENVS * ACTION_DIM]
    act_host: HostBuffer[dtype],
    action_scale: Float64 = 1.0,
    deterministic: Bool = False,
    rng_base_seed: UInt32 = 42,
) raises:
    """GPU-batched MPPI planning for all envs simultaneously.

    Plans for all N_ENVS environments in a single GPU call by batching
    N_ENVS * TOTAL_SAMPLES trajectories. Reduces GPU-CPU syncs from
    N_ENVS * NUM_ITERATIONS to just NUM_ITERATIONS.

    Selected actions are written to act_host [N_ENVS * ACTION_DIM].
    """
    # ─── Compile-time constants ────────────────────────────────────────────
    comptime TOTAL_SAMPLES = NUM_SAMPLES + NUM_PI_TRAJS
    comptime BATCH_TOTAL = N_ENVS * TOTAL_SAMPLES
    comptime ZA_DIM = LATENT_DIM + ACTION_DIM
    comptime POL_OUT = PolModel.OUT_DIM
    comptime STD_MIN: Float64 = 0.05
    comptime STD_MAX: Float64 = 2.0
    comptime MPPI_BLOCKS = (BATCH_TOTAL + TPB - 1) // TPB

    # ─── LayoutTensor views over batched MPPI GPU buffers ──────────────────
    var z_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH_TOTAL, LATENT_DIM), MutAnyOrigin
    ](mb.z_buf.unsafe_ptr())
    var z_next_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH_TOTAL, LATENT_DIM), MutAnyOrigin
    ](mb.z_next_buf.unsafe_ptr())
    var za_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH_TOTAL, ZA_DIM), MutAnyOrigin
    ](mb.za_buf.unsafe_ptr())
    var act_step_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH_TOTAL, ACTION_DIM), MutAnyOrigin
    ](mb.act_step_buf.unsafe_ptr())
    var all_actions_tensor = LayoutTensor[
        dtype,
        Layout.row_major(BATCH_TOTAL * HORIZON * ACTION_DIM),
        MutAnyOrigin,
    ](mb.all_actions_buf.unsafe_ptr())
    var rew_logits_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH_TOTAL, NUM_BINS), MutAnyOrigin
    ](mb.rew_logits_buf.unsafe_ptr())
    var q_logits_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH_TOTAL, NUM_BINS), MutAnyOrigin
    ](mb.q_logits_buf.unsafe_ptr())
    var returns_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH_TOTAL), MutAnyOrigin
    ](mb.returns_buf.unsafe_ptr())
    var q_min_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH_TOTAL), MutAnyOrigin
    ](mb.q_min_buf.unsafe_ptr())
    var pi_out_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH_TOTAL, POL_OUT), MutAnyOrigin
    ](mb.pi_out_buf.unsafe_ptr())
    var mean_tensor = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * HORIZON * ACTION_DIM), MutAnyOrigin
    ](mb.mean_buf.unsafe_ptr())
    var std_tensor = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * HORIZON * ACTION_DIM), MutAnyOrigin
    ](mb.std_buf.unsafe_ptr())

    # Model-typed tensor views
    var pol_out_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH_TOTAL, PolModel.OUT_DIM), MutAnyOrigin
    ](mb.pi_out_buf.unsafe_ptr())
    var pol_in_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH_TOTAL, PolModel.IN_DIM), MutAnyOrigin
    ](mb.z_buf.unsafe_ptr())
    var dyn_in_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH_TOTAL, DynModel.IN_DIM), MutAnyOrigin
    ](mb.za_buf.unsafe_ptr())
    var dyn_out_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH_TOTAL, DynModel.OUT_DIM), MutAnyOrigin
    ](mb.z_next_buf.unsafe_ptr())
    var rew_in_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH_TOTAL, RewModel.IN_DIM), MutAnyOrigin
    ](mb.za_buf.unsafe_ptr())
    var rew_out_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH_TOTAL, RewModel.OUT_DIM), MutAnyOrigin
    ](mb.rew_logits_buf.unsafe_ptr())
    var q_in_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH_TOTAL, QModel.IN_DIM), MutAnyOrigin
    ](mb.za_buf.unsafe_ptr())
    var q_out_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH_TOTAL, QModel.OUT_DIM), MutAnyOrigin
    ](mb.q_logits_buf.unsafe_ptr())

    # ─── Kernel aliases ────────────────────────────────────────────────────
    # Fused: broadcast_z0 + zero_returns → single kernel
    comptime broadcast_z0_zero = mppi_broadcast_z0_zero_returns_batched_kernel[
        dtype, BATCH_TOTAL, N_ENVS, TOTAL_SAMPLES, LATENT_DIM
    ]
    # Fused: sample_actions + build_za → single kernel
    comptime sample_build_za = mppi_sample_actions_build_za_batched_kernel[
        dtype,
        BATCH_TOTAL,
        N_ENVS,
        TOTAL_SAMPLES,
        NUM_PI_TRAJS,
        ACTION_DIM,
        LATENT_DIM,
        HORIZON,
        POL_OUT,
    ]
    # Fused: accum_reward + copy_z → single kernel
    comptime accum_copy = mppi_accum_reward_copy_z_kernel[
        dtype, BATCH_TOTAL, NUM_BINS, LATENT_DIM
    ]
    # Unfused (still needed standalone for terminal value step)
    comptime accum_reward = mppi_accumulate_reward_kernel[
        dtype, BATCH_TOTAL, NUM_BINS
    ]
    comptime add_terminal = mppi_add_terminal_value_kernel[dtype, BATCH_TOTAL]
    comptime tanh_build_za = tdmpc2_apply_tanh_build_za_deterministic_kernel[
        dtype, BATCH_TOTAL, ACTION_DIM, LATENT_DIM, POL_OUT
    ]
    comptime q_decode = tdmpc2_q_decode_kernel[dtype, BATCH_TOTAL, NUM_BINS]
    comptime decode_min = tdmpc2_decode_and_min_kernel[
        dtype, BATCH_TOTAL, NUM_BINS
    ]
    comptime softmax_weights = mppi_softmax_weights_kernel[
        dtype, N_ENVS, TOTAL_SAMPLES, TPB
    ]
    comptime weighted_mean_std = mppi_weighted_mean_std_kernel[
        dtype, N_ENVS, TOTAL_SAMPLES, HORIZON, ACTION_DIM
    ]
    comptime MEAN_STD_TOTAL = N_ENVS * HORIZON * ACTION_DIM
    comptime MEAN_STD_BLOCKS = (MEAN_STD_TOTAL + TPB - 1) // TPB

    # ─── Tensor views for weights on GPU ───────────────────────────────────
    var weights_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH_TOTAL), MutAnyOrigin
    ](mb.weights_buf.unsafe_ptr())

    # ─── Initialize per-env mean/std on CPU, upload to GPU ─────────────────
    for env_idx in range(N_ENVS):
        var base = env_idx * HORIZON * ACTION_DIM
        for i in range(HORIZON * ACTION_DIM):
            mb.mean_host[base + i] = Scalar[dtype](0.0)
            mb.std_host[base + i] = Scalar[dtype](0.5)

        # Warm-start: shift previous plan's mean forward by 1 step
        if (
            not env_t0_flags[env_idx]
            and len(env_prev_means[env_idx]) == HORIZON * ACTION_DIM
        ):
            for t in range(HORIZON - 1):
                for a in range(ACTION_DIM):
                    mb.mean_host[base + t * ACTION_DIM + a] = Scalar[dtype](
                        env_prev_means[env_idx][(t + 1) * ACTION_DIM + a]
                    )

    ctx.enqueue_copy(mb.mean_buf, mb.mean_host)
    ctx.enqueue_copy(mb.std_buf, mb.std_host)

    # ─── Create streams for parallel rew+dyn (NVIDIA only) ─────────────
    comptime USE_STREAMS = has_nvidia_gpu_accelerator()
    var s1 = ctx.create_stream()  # reward
    var s2 = ctx.create_stream()  # dynamics

    # ─── Hoist Q5 concat_params before loop (target params are read-only) ──
    # Q model = Sequential[NormedLinear[ZA,MLP], NormedLinear[MLP,MLP], Linear[MLP,BINS]]
    comptime Q_PS = QModel.PARAM_SIZE
    comptime Q5_BT = NUM_Q * BATCH_TOTAL
    comptime Q5_BLOCKS = (Q5_BT + TPB - 1) // TPB
    comptime concat_params = q5_concat_params_kernel[dtype, Q_PS]
    var q5_params = LayoutTensor[
        dtype, Layout.row_major(NUM_Q * Q_PS), MutAnyOrigin
    ](mb.q5_params_buf.unsafe_ptr())
    var qt1_p = LayoutTensor[
        dtype, Layout.row_major(Q_PS), MutAnyOrigin
    ](qt_param_ptrs[0])
    var qt2_p = LayoutTensor[
        dtype, Layout.row_major(Q_PS), MutAnyOrigin
    ](qt_param_ptrs[1])
    var qt3_p = LayoutTensor[
        dtype, Layout.row_major(Q_PS), MutAnyOrigin
    ](qt_param_ptrs[2])
    var qt4_p = LayoutTensor[
        dtype, Layout.row_major(Q_PS), MutAnyOrigin
    ](qt_param_ptrs[3])
    var qt5_p = LayoutTensor[
        dtype, Layout.row_major(Q_PS), MutAnyOrigin
    ](qt_param_ptrs[4])
    ctx.enqueue_function[concat_params, concat_params](
        q5_params,
        qt1_p,
        qt2_p,
        qt3_p,
        qt4_p,
        qt5_p,
        grid_dim=((Q_PS + TPB - 1) // TPB,),
        block_dim=(TPB,),
    )

    # ─── Main MPPI iterations ────────────────────────────────────────────
    var temp_scalar = Scalar[dtype](temperature)
    for mppi_iter in range(NUM_ITERATIONS):
        var rng_seed = rng_base_seed + UInt32(
            mppi_iter * BATCH_TOTAL * HORIZON * ACTION_DIM * 2
        )

        # 1. Fused: broadcast per-env z0 + zero returns (1 kernel, was 2)
        ctx.enqueue_function[broadcast_z0_zero, broadcast_z0_zero](
            z0_tensor,
            z_tensor,
            returns_tensor,
            grid_dim=(MPPI_BLOCKS,),
            block_dim=(TPB,),
        )

        # 3. Horizon rollout (H sequential steps, all envs batched)
        var discount = Scalar[dtype](1.0)
        for t in range(HORIZON):
            var step_seed = rng_seed + UInt32(t * BATCH_TOTAL * ACTION_DIM + 1)

            # 3a. Policy forward
            PolModel.forward_gpu_no_cache[BATCH_TOTAL](
                ctx,
                pol_out_tensor,
                pol_in_tensor,
                pol_params,
                mb.pol_ws_buf,
            )

            # 3b. Fused: sample actions + build za (1 kernel, was 2)
            ctx.enqueue_function[sample_build_za, sample_build_za](
                pi_out_tensor,
                mean_tensor,
                std_tensor,
                z_tensor,
                za_tensor,
                all_actions_tensor,
                t,
                Scalar[DType.uint32](step_seed),
                grid_dim=(MPPI_BLOCKS,),
                block_dim=(TPB,),
            )

            # 3d-g. Reward + Dynamics
            comptime if USE_STREAMS:
                # Stream 1: Reward forward + accumulate
                RewModel.forward_gpu_no_cache_on_stream[BATCH_TOTAL](
                    ctx,
                    s1,
                    rew_out_tensor,
                    rew_in_tensor,
                    rew_params,
                    mb.rew_ws_buf,
                )
                var compiled_accum = ctx.compile_function[
                    accum_reward, accum_reward
                ]()
                s1.enqueue_function(
                    compiled_accum,
                    rew_logits_tensor,
                    bins_tensor,
                    returns_tensor,
                    discount,
                    grid_dim=(MPPI_BLOCKS,),
                    block_dim=(TPB,),
                )
                # Stream 2: Dynamics forward + copy z
                DynModel.forward_gpu_no_cache_on_stream[BATCH_TOTAL](
                    ctx,
                    s2,
                    dyn_out_tensor,
                    dyn_in_tensor,
                    dyn_params,
                    mb.dyn_ws_buf,
                )
                comptime copy_z = mppi_copy_z_kernel[
                    dtype, BATCH_TOTAL, LATENT_DIM
                ]
                var compiled_copy_z = ctx.compile_function[copy_z, copy_z]()
                s2.enqueue_function(
                    compiled_copy_z,
                    z_tensor,
                    z_next_tensor,
                    grid_dim=(MPPI_BLOCKS,),
                    block_dim=(TPB,),
                )
                # Wait for both to finish before next horizon step
                s1.synchronize()
                s2.synchronize()
            else:
                # Reward forward
                RewModel.forward_gpu_no_cache[BATCH_TOTAL](
                    ctx,
                    rew_out_tensor,
                    rew_in_tensor,
                    rew_params,
                    mb.rew_ws_buf,
                )
                # Dynamics forward
                DynModel.forward_gpu_no_cache[BATCH_TOTAL](
                    ctx,
                    dyn_out_tensor,
                    dyn_in_tensor,
                    dyn_params,
                    mb.dyn_ws_buf,
                )
                # Fused: accum reward + copy z (1 kernel, was 2)
                ctx.enqueue_function[accum_copy, accum_copy](
                    rew_logits_tensor,
                    bins_tensor,
                    returns_tensor,
                    discount,
                    z_tensor,
                    z_next_tensor,
                    grid_dim=(MPPI_BLOCKS,),
                    block_dim=(TPB,),
                )
            discount = discount * Scalar[dtype](gamma)

        # 4. Terminal value: policy → tanh → build za
        PolModel.forward_gpu_no_cache[BATCH_TOTAL](
            ctx,
            pol_out_tensor,
            pol_in_tensor,
            pol_params,
            mb.pol_ws_buf,
        )
        ctx.enqueue_function[tanh_build_za, tanh_build_za](
            pi_out_tensor,
            act_step_tensor,
            z_tensor,
            za_tensor,
            grid_dim=(MPPI_BLOCKS,),
            block_dim=(TPB,),
        )

        # Q1..Q5: batched grouped forward + decode + min
        # (concat_params hoisted before the loop — target params don't change)

        # ── Compile-time layer param offsets ─────────────────────────
        comptime L0_SIZE = ZA_DIM * MLP_DIM + 3 * MLP_DIM
        comptime L1_SIZE = MLP_DIM * MLP_DIM + 3 * MLP_DIM
        comptime L0_BASE = 0
        comptime L1_BASE = L0_SIZE
        comptime L2_BASE = L0_SIZE + L1_SIZE
        comptime L0_W = L0_BASE
        comptime L0_B = L0_BASE + ZA_DIM * MLP_DIM
        comptime L0_GAMMA = L0_BASE + ZA_DIM * MLP_DIM + MLP_DIM
        comptime L0_BETA = L0_BASE + ZA_DIM * MLP_DIM + 2 * MLP_DIM
        comptime L1_W = L1_BASE
        comptime L1_B = L1_BASE + MLP_DIM * MLP_DIM
        comptime L1_GAMMA = L1_BASE + MLP_DIM * MLP_DIM + MLP_DIM
        comptime L1_BETA = L1_BASE + MLP_DIM * MLP_DIM + 2 * MLP_DIM
        comptime L2_W = L2_BASE
        comptime L2_B = L2_BASE + MLP_DIM * NUM_BINS

        # ── Grouped kernel aliases ───────────────────────────────────
        comptime TILE = 16
        comptime MM_GRID_0 = (
            (MLP_DIM + TILE - 1) // TILE,
            (BATCH_TOTAL + TILE - 1) // TILE,
            NUM_Q,
        )
        comptime MM_GRID_2 = (
            (NUM_BINS + TILE - 1) // TILE,
            (BATCH_TOTAL + TILE - 1) // TILE,
            NUM_Q,
        )

        comptime replicate_za = q5_replicate_input_kernel[
            dtype, BATCH_TOTAL, ZA_DIM, NUM_Q
        ]
        comptime gmatmul_0 = q5_grouped_matmul_bias_kernel[
            dtype, NUM_Q, BATCH_TOTAL, ZA_DIM, MLP_DIM, Q_PS, L0_W, L0_B
        ]
        comptime gln_mish_0 = q5_grouped_ln_mish_kernel[
            dtype, NUM_Q, BATCH_TOTAL, MLP_DIM, Q_PS, L0_GAMMA, L0_BETA
        ]
        comptime gmatmul_1 = q5_grouped_matmul_bias_kernel[
            dtype, NUM_Q, BATCH_TOTAL, MLP_DIM, MLP_DIM, Q_PS, L1_W, L1_B
        ]
        comptime gln_mish_1 = q5_grouped_ln_mish_kernel[
            dtype, NUM_Q, BATCH_TOTAL, MLP_DIM, Q_PS, L1_GAMMA, L1_BETA
        ]
        comptime gmatmul_2 = q5_grouped_matmul_bias_kernel[
            dtype, NUM_Q, BATCH_TOTAL, MLP_DIM, NUM_BINS, Q_PS, L2_W, L2_B
        ]
        comptime gdecode_min = q5_decode_min_kernel[
            dtype, NUM_Q, BATCH_TOTAL, NUM_BINS
        ]

        # ── Tensor views over batched Q buffers ──────────────────────
        var q5_a_za = LayoutTensor[
            dtype, Layout.row_major(Q5_BT, ZA_DIM), MutAnyOrigin
        ](mb.q5_buf_a.unsafe_ptr())
        var q5_a_mlp = LayoutTensor[
            dtype, Layout.row_major(Q5_BT, MLP_DIM), MutAnyOrigin
        ](mb.q5_buf_a.unsafe_ptr())
        var q5_b_mlp = LayoutTensor[
            dtype, Layout.row_major(Q5_BT, MLP_DIM), MutAnyOrigin
        ](mb.q5_buf_b.unsafe_ptr())
        var q5_b_bins = LayoutTensor[
            dtype, Layout.row_major(Q5_BT, NUM_BINS), MutAnyOrigin
        ](mb.q5_buf_b.unsafe_ptr())

        # ── 1. Replicate za input 5x ──────────────────────────────
        ctx.enqueue_function[replicate_za, replicate_za](
            q5_a_za,
            q_in_tensor,
            grid_dim=(Q5_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 3. Layer 0: grouped matmul+bias → grouped LN+Mish ──────
        #    input: q5_a_za [5*BT, ZA] → matmul → q5_b_mlp [5*BT, MLP]
        #    LN+Mish: q5_b_mlp → q5_a_mlp [5*BT, MLP]
        var eps_scalar = Scalar[dtype](1e-5)
        ctx.enqueue_function[gmatmul_0, gmatmul_0](
            q5_b_mlp,
            q5_a_za,
            q5_params,
            grid_dim=MM_GRID_0,
            block_dim=(TILE, TILE),
        )
        ctx.enqueue_function[gln_mish_0, gln_mish_0](
            q5_a_mlp,
            q5_b_mlp,
            q5_params,
            eps_scalar,
            grid_dim=(Q5_BT,),
            block_dim=(1,),
        )

        # ── 4. Layer 1: grouped matmul+bias → grouped LN+Mish ──────
        #    input: q5_a_mlp [5*BT, MLP] → matmul → q5_b_mlp [5*BT, MLP]
        #    LN+Mish: q5_b_mlp → q5_a_mlp [5*BT, MLP]
        ctx.enqueue_function[gmatmul_1, gmatmul_1](
            q5_b_mlp,
            q5_a_mlp,
            q5_params,
            grid_dim=MM_GRID_0,
            block_dim=(TILE, TILE),
        )
        ctx.enqueue_function[gln_mish_1, gln_mish_1](
            q5_a_mlp,
            q5_b_mlp,
            q5_params,
            eps_scalar,
            grid_dim=(Q5_BT,),
            block_dim=(1,),
        )

        # ── 5. Layer 2: grouped matmul+bias (no activation) ─────────
        #    input: q5_a_mlp [5*BT, MLP] → matmul → q5_b_bins [5*BT, BINS]
        ctx.enqueue_function[gmatmul_2, gmatmul_2](
            q5_b_bins,
            q5_a_mlp,
            q5_params,
            grid_dim=MM_GRID_2,
            block_dim=(TILE, TILE),
        )

        # ── 6. Decode 5 logits + min ────────────────────────────────
        ctx.enqueue_function[gdecode_min, gdecode_min](
            q5_b_bins,
            bins_tensor,
            q_min_tensor,
            grid_dim=(MPPI_BLOCKS,),
            block_dim=(TPB,),
        )

        # 4b. Add terminal value to returns
        ctx.enqueue_function[add_terminal, add_terminal](
            q_min_tensor,
            returns_tensor,
            discount,
            grid_dim=(MPPI_BLOCKS,),
            block_dim=(TPB,),
        )

        # 5. Softmax weights + 6. Weighted mean/std
        ctx.enqueue_function[softmax_weights, softmax_weights](
            returns_tensor,
            weights_tensor,
            temp_scalar,
            grid_dim=(N_ENVS,),
            block_dim=(TPB,),
        )
        ctx.enqueue_function[weighted_mean_std, weighted_mean_std](
            weights_tensor,
            all_actions_tensor,
            mean_tensor,
            std_tensor,
            grid_dim=(MEAN_STD_BLOCKS,),
            block_dim=(TPB,),
        )
        # No sync needed — GPU executes kernels in order on the same queue.
        # Next iteration's sample_build_za will naturally wait for
        # weighted_mean_std to finish writing mean/std.

    # ─── Download results ─────────────────────────────────────────────────
    ctx.enqueue_copy(mb.all_actions_host, mb.all_actions_buf)
    ctx.enqueue_copy(mb.weights_host, mb.weights_buf)
    ctx.enqueue_copy(mb.mean_host, mb.mean_buf)
    ctx.enqueue_copy(mb.std_host, mb.std_buf)
    ctx.synchronize()

    # ─── Store final means + action selection per env ──────────────────────
    for env_idx in range(N_ENVS):
        var mean_base = env_idx * HORIZON * ACTION_DIM
        var env_act_off = env_idx * TOTAL_SAMPLES * HORIZON * ACTION_DIM
        var w_off = env_idx * TOTAL_SAMPLES

        # Store final mean for warm-starting next timestep
        env_prev_means[env_idx] = List[Float64](capacity=HORIZON * ACTION_DIM)
        for i in range(HORIZON * ACTION_DIM):
            env_prev_means[env_idx].append(Float64(mb.mean_host[mean_base + i]))
        env_t0_flags[env_idx] = False

        # Weighted random sample for this env using final weights
        var env_w = List[Float64](capacity=TOTAL_SAMPLES)
        for s in range(TOTAL_SAMPLES):
            env_w.append(Float64(mb.weights_host[w_off + s]))
        var selected_s = _weighted_sample(env_w, TOTAL_SAMPLES)

        for a in range(ACTION_DIM):
            var act = Float64(
                mb.all_actions_host[
                    env_act_off + selected_s * HORIZON * ACTION_DIM + a
                ]
            )
            if not deterministic:
                act += _gaussian_sample() * Float64(mb.std_host[mean_base + a])
            act = _clamp(act * action_scale, -action_scale, action_scale)
            act_host[env_idx * ACTION_DIM + a] = Scalar[dtype](act)
