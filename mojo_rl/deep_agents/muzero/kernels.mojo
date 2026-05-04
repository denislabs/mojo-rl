"""MuZero GPU Kernels — Elementwise operations for GPU training.

Kernels for:
  - Hidden state min-max scaling
  - Cross-entropy gradient (softmax - target) for policy/value/reward
  - Gradient clipping (global norm)
  - Dynamics input assembly (hidden || one_hot_action)
  - Dynamics output extraction (hidden state + reward logits)
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor
from std.math import exp, sqrt


comptime TPB: Int = 256  # Threads per block


# ═══════════════════════════════════════════════════════════════════════════
# Hidden State Min-Max Scaling
# ═══════════════════════════════════════════════════════════════════════════


def scale_hidden_kernel[
    BATCH: Int,
    LATENT: Int,
    dtype: DType where dtype.is_floating_point(),
](hidden: LayoutTensor[dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin],):
    """Scale hidden states to [0, 1] per sample via min-max normalization.

    Each thread handles one sample (all LATENT elements).
    Grid: (BATCH + TPB - 1) // TPB blocks of TPB threads.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var offset = b * LATENT

    # Find min/max for this sample
    var min_val = rebind[Scalar[dtype]](hidden[offset])
    var max_val = min_val
    for i in range(1, LATENT):
        var v = rebind[Scalar[dtype]](hidden[offset + i])
        if v < min_val:
            min_val = v
        if v > max_val:
            max_val = v

    var delta = max_val - min_val
    if delta > Scalar[dtype](1e-8):
        var inv_delta = Scalar[dtype](1.0) / delta
        for i in range(LATENT):
            var v = rebind[Scalar[dtype]](hidden[offset + i])
            hidden[offset + i] = (v - min_val) * inv_delta


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Entropy Gradient: softmax(logits) - target
# ═══════════════════════════════════════════════════════════════════════════


def ce_policy_grad_kernel[
    BATCH: Int,
    ACT: Int,
    PRED_OUT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    grad_out: LayoutTensor[
        dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
    ],
    pred_outputs: LayoutTensor[
        dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
    ],
    policy_targets: LayoutTensor[
        dtype, Layout.row_major(BATCH * ACT), MutAnyOrigin
    ],
    scale: Scalar[dtype],
):
    """Compute policy CE gradient: scale * (softmax(logits) - target).

    One thread per batch sample. Writes to the first ACT elements of grad_out
    for each sample.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var pred_off = b * PRED_OUT
    var pol_off = b * ACT

    # Numerically stable softmax over policy logits
    var max_val = rebind[Scalar[dtype]](pred_outputs[pred_off])
    for a in range(1, ACT):
        var v = rebind[Scalar[dtype]](pred_outputs[pred_off + a])
        if v > max_val:
            max_val = v

    var sum_exp = Scalar[dtype](0.0)
    for a in range(ACT):
        sum_exp += exp(
            rebind[Scalar[dtype]](pred_outputs[pred_off + a]) - max_val
        )

    var inv_sum = Scalar[dtype](1.0) / sum_exp
    for a in range(ACT):
        var prob = (
            exp(rebind[Scalar[dtype]](pred_outputs[pred_off + a]) - max_val)
            * inv_sum
        )
        var target = rebind[Scalar[dtype]](policy_targets[pol_off + a])
        grad_out[pred_off + a] = (prob - target) * scale


def ce_value_grad_kernel[
    BATCH: Int,
    BINS: Int,
    ACT: Int,
    PRED_OUT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    grad_out: LayoutTensor[
        dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
    ],
    pred_outputs: LayoutTensor[
        dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
    ],
    value_targets: LayoutTensor[
        dtype, Layout.row_major(BATCH * BINS), MutAnyOrigin
    ],
    scale: Scalar[dtype],
):
    """Compute value CE gradient: scale * (softmax(value_logits) - target).

    One thread per batch sample. Writes to elements [ACT..ACT+BINS) of grad_out.
    value_targets is pre-encoded as two-hot distribution [BATCH * BINS].
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var pred_off = b * PRED_OUT + ACT  # Value logits start after policy
    var tgt_off = b * BINS

    # Softmax over value logits
    var max_val = rebind[Scalar[dtype]](pred_outputs[pred_off])
    for i in range(1, BINS):
        var v = rebind[Scalar[dtype]](pred_outputs[pred_off + i])
        if v > max_val:
            max_val = v

    var sum_exp = Scalar[dtype](0.0)
    for i in range(BINS):
        sum_exp += exp(
            rebind[Scalar[dtype]](pred_outputs[pred_off + i]) - max_val
        )

    var inv_sum = Scalar[dtype](1.0) / sum_exp
    for i in range(BINS):
        var prob = (
            exp(rebind[Scalar[dtype]](pred_outputs[pred_off + i]) - max_val)
            * inv_sum
        )
        var target = rebind[Scalar[dtype]](value_targets[tgt_off + i])
        grad_out[b * PRED_OUT + ACT + i] = (prob - target) * scale


def ce_reward_grad_kernel[
    BATCH: Int,
    BINS: Int,
    DYN_OUT: Int,
    LATENT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    grad_out: LayoutTensor[
        dtype, Layout.row_major(BATCH * DYN_OUT), MutAnyOrigin
    ],
    dyn_outputs: LayoutTensor[
        dtype, Layout.row_major(BATCH * DYN_OUT), MutAnyOrigin
    ],
    reward_targets: LayoutTensor[
        dtype, Layout.row_major(BATCH * BINS), MutAnyOrigin
    ],
    scale: Scalar[dtype],
):
    """Compute reward CE gradient: scale * (softmax(reward_logits) - target).

    One thread per batch sample. Writes to elements [LATENT..LATENT+BINS) of grad_out.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var rew_off = b * DYN_OUT + LATENT  # Reward logits start after hidden
    var tgt_off = b * BINS

    var max_val = rebind[Scalar[dtype]](dyn_outputs[rew_off])
    for i in range(1, BINS):
        var v = rebind[Scalar[dtype]](dyn_outputs[rew_off + i])
        if v > max_val:
            max_val = v

    var sum_exp = Scalar[dtype](0.0)
    for i in range(BINS):
        sum_exp += exp(
            rebind[Scalar[dtype]](dyn_outputs[rew_off + i]) - max_val
        )

    var inv_sum = Scalar[dtype](1.0) / sum_exp
    for i in range(BINS):
        var prob = (
            exp(rebind[Scalar[dtype]](dyn_outputs[rew_off + i]) - max_val)
            * inv_sum
        )
        var target = rebind[Scalar[dtype]](reward_targets[tgt_off + i])
        grad_out[rew_off + i] = (prob - target) * scale


# ═══════════════════════════════════════════════════════════════════════════
# Two-Hot Encoding
# ═══════════════════════════════════════════════════════════════════════════


def two_hot_encode_kernel[
    BATCH: Int,
    BINS: Int,
    dtype: DType where dtype.is_floating_point(),
](
    targets_out: LayoutTensor[
        dtype, Layout.row_major(BATCH * BINS), MutAnyOrigin
    ],
    scalar_values: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    v_min: Scalar[dtype],
    v_max: Scalar[dtype],
):
    """Encode scalar values as two-hot categorical distributions.

    One thread per batch sample.
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var val = rebind[Scalar[dtype]](scalar_values[b])
    # Clamp
    if val < v_min:
        val = v_min
    if val > v_max:
        val = v_max

    var step = (v_max - v_min) / Scalar[dtype](BINS - 1)
    var pos = (val - v_min) / step

    var lo = Int(pos)
    if lo >= BINS - 1:
        lo = BINS - 2
    if lo < 0:
        lo = 0
    var hi = lo + 1
    var frac = pos - Scalar[dtype](lo)

    var out_off = b * BINS
    for i in range(BINS):
        targets_out[out_off + i] = Scalar[dtype](0.0)
    targets_out[out_off + lo] = Scalar[dtype](1.0) - frac
    targets_out[out_off + hi] = frac


# ═══════════════════════════════════════════════════════════════════════════
# Dynamics Input Assembly
# ═══════════════════════════════════════════════════════════════════════════


def build_dyn_input_kernel[
    BATCH: Int,
    LATENT: Int,
    ACT: Int,
    DYN_IN: Int,
    dtype: DType where dtype.is_floating_point(),
](
    dyn_input: LayoutTensor[
        dtype, Layout.row_major(BATCH * DYN_IN), MutAnyOrigin
    ],
    hidden_states: LayoutTensor[
        dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
    ],
    actions: LayoutTensor[dtype, Layout.row_major(BATCH * ACT), MutAnyOrigin],
):
    """Assemble dynamics input: [hidden_state || one_hot_action] per sample.

    One thread per element in the output [BATCH * DYN_IN].
    """
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * DYN_IN:
        return

    var b = idx // DYN_IN
    var d = idx % DYN_IN

    if d < LATENT:
        dyn_input[idx] = hidden_states[b * LATENT + d]
    else:
        dyn_input[idx] = actions[b * ACT + (d - LATENT)]


# ═══════════════════════════════════════════════════════════════════════════
# Dynamics Output Extraction
# ═══════════════════════════════════════════════════════════════════════════


def extract_hidden_kernel[
    BATCH: Int,
    LATENT: Int,
    DYN_OUT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    next_hidden: LayoutTensor[
        dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
    ],
    dyn_output: LayoutTensor[
        dtype, Layout.row_major(BATCH * DYN_OUT), MutAnyOrigin
    ],
):
    """Extract hidden state from dynamics output (first LATENT elements per sample).
    """
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * LATENT:
        return

    var b = idx // LATENT
    var d = idx % LATENT
    next_hidden[idx] = dyn_output[b * DYN_OUT + d]


# ═══════════════════════════════════════════════════════════════════════════
# Hidden Gradient from Dynamics Input Gradient
# ═══════════════════════════════════════════════════════════════════════════


def extract_hidden_grad_kernel[
    BATCH: Int,
    LATENT: Int,
    DYN_IN: Int,
    dtype: DType where dtype.is_floating_point(),
](
    grad_hidden: LayoutTensor[
        dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
    ],
    grad_dyn_in: LayoutTensor[
        dtype, Layout.row_major(BATCH * DYN_IN), MutAnyOrigin
    ],
):
    """Extract hidden state gradient from dynamics input gradient."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * LATENT:
        return

    var b = idx // LATENT
    var d = idx % LATENT
    grad_hidden[idx] = grad_dyn_in[b * DYN_IN + d]


# ═══════════════════════════════════════════════════════════════════════════
# Gradient Accumulation and Scaling
# ═══════════════════════════════════════════════════════════════════════════


def add_scaled_kernel[
    SIZE: Int,
    dtype: DType where dtype.is_floating_point(),
](
    dst: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    src: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    scale: Scalar[dtype],
):
    """Formula : dst += src * scale (elementwise)."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SIZE:
        return
    dst[idx] = (
        rebind[Scalar[dtype]](dst[idx])
        + rebind[Scalar[dtype]](src[idx]) * scale
    )


def scale_kernel[
    SIZE: Int,
    dtype: DType where dtype.is_floating_point(),
](
    data: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    scale: Scalar[dtype],
):
    """Formula : data *= scale (elementwise)."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SIZE:
        return
    data[idx] = rebind[Scalar[dtype]](data[idx]) * scale


def copy_kernel[
    SIZE: Int,
    dtype: DType where dtype.is_floating_point(),
](
    dst: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    src: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
):
    """Formula: dst = src (elementwise copy)."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SIZE:
        return
    dst[idx] = src[idx]


# ═══════════════════════════════════════════════════════════════════════════
# Hidden Gradient: dual consumer scaling (prediction + dynamics)
# ═══════════════════════════════════════════════════════════════════════════


def set_hidden_grad_for_dyn_kernel[
    BATCH: Int,
    LATENT: Int,
    DYN_OUT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    grad_dyn_out: LayoutTensor[
        dtype, Layout.row_major(BATCH * DYN_OUT), MutAnyOrigin
    ],
    grad_hidden: LayoutTensor[
        dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
    ],
    scale: Scalar[dtype],
):
    """Copy hidden gradient into dynamics output gradient (first LATENT elements), scaled.

    grad_dyn_out[:, :LATENT] = grad_hidden * scale
    """
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * LATENT:
        return

    var b = idx // LATENT
    var d = idx % LATENT
    grad_dyn_out[b * DYN_OUT + d] = (
        rebind[Scalar[dtype]](grad_hidden[idx]) * scale
    )


# ═══════════════════════════════════════════════════════════════════════════
# MCTS Target Storage (per-env circular buffer, same write_idx as replay)
# ═══════════════════════════════════════════════════════════════════════════


def to_play_from_episode_step_kernel[
    N_ENVS: Int,
    dtype: DType where dtype.is_floating_point(),
](
    ep_steps: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    to_play_out: LayoutTensor[
        DType.uint8, Layout.row_major(N_ENVS), MutAnyOrigin
    ],
):
    """Compute per-env player-to-move from per-env episode-step counter.

    Assumes player 0 starts after each reset and players strictly alternate
    (the standard convention for two-player turn-based games like TicTacToe
    and Connect Four). One thread per env. ep_steps[e] is the count of
    transitions stored in the current episode for env e (so ep_steps=0 on
    the first move). to_play_out[e] = ep_steps[e] mod 2.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var s = Int(rebind[Scalar[dtype]](ep_steps[e]))
    to_play_out[e] = UInt8(s & 1)


def store_mcts_targets_kernel[
    N_ENVS: Int,
    PER_ENV_CAP: Int,
    ACT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    # Input: MCTS policies/values/to-play for this step (from CPU)
    policies_in: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ],
    values_in: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    to_play_in: LayoutTensor[
        DType.uint8, Layout.row_major(N_ENVS), MutAnyOrigin
    ],
    # Output: per-env circular buffers
    policy_buf: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PER_ENV_CAP * ACT), MutAnyOrigin
    ],
    value_buf: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PER_ENV_CAP), MutAnyOrigin
    ],
    to_play_buf: LayoutTensor[
        DType.uint8, Layout.row_major(N_ENVS * PER_ENV_CAP), MutAnyOrigin
    ],
    write_idx: Scalar[DType.int32],
):
    """Store MCTS policy/value/to-play targets into per-env circular buffer.

    One thread per env. Writes at the same write_idx used by
    GPUSequenceReplayBuffer.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var w = Int(write_idx)

    # Store policy [ACT dims]
    var pol_base = e * PER_ENV_CAP * ACT + w * ACT
    var pol_src = e * ACT
    for a in range(ACT):
        policy_buf[pol_base + a] = policies_in[pol_src + a]

    # Store value + player-to-move [1 scalar each]
    value_buf[e * PER_ENV_CAP + w] = values_in[e]
    to_play_buf[e * PER_ENV_CAP + w] = to_play_in[e]


# ═══════════════════════════════════════════════════════════════════════════
# Combined Sequence + MCTS Target Sampling
# ═══════════════════════════════════════════════════════════════════════════


def sample_seq_with_targets_kernel[
    BATCH: Int,
    H: Int,  # Unroll length K (we sample H+1 obs/policies, H actions)
    N_TD: Int,  # n-step horizon for value bootstrap (rewards/dones/values
    #            extend to H+N_TD)
    N_ENVS: Int,
    PER_ENV_CAP: Int,
    OBS_DIM: Int,
    ACT_DIM: Int,
    dtype: DType where dtype.is_floating_point(),
](
    # Replay buffer storage (per-env circular)
    buf_obs: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PER_ENV_CAP * OBS_DIM), MutAnyOrigin
    ],
    buf_actions: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PER_ENV_CAP * ACT_DIM), MutAnyOrigin
    ],
    buf_rewards: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PER_ENV_CAP), MutAnyOrigin
    ],
    buf_dones: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PER_ENV_CAP), MutAnyOrigin
    ],
    # MCTS target storage (parallel per-env circular)
    buf_policies: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PER_ENV_CAP * ACT_DIM), MutAnyOrigin
    ],
    buf_values: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PER_ENV_CAP), MutAnyOrigin
    ],
    buf_to_play: LayoutTensor[
        DType.uint8, Layout.row_major(N_ENVS * PER_ENV_CAP), MutAnyOrigin
    ],
    # Output batch buffers — TIME-MAJOR layout so the forward pass can
    # read [t, b, *] in contiguous BATCH-sized slices.
    # Window sizes: obs/policies extend H+N_TD+1 timesteps so reanalyze
    # can fresh-forward at every bootstrap position; rewards/dones extend
    # H+N_TD; values/to_play extend H+N_TD+1.
    batch_obs: LayoutTensor[
        dtype,
        Layout.row_major((H + N_TD + 1) * BATCH * OBS_DIM),
        MutAnyOrigin,
    ],
    batch_actions: LayoutTensor[
        dtype, Layout.row_major(H * BATCH * ACT_DIM), MutAnyOrigin
    ],
    batch_rewards: LayoutTensor[
        dtype, Layout.row_major((H + N_TD) * BATCH), MutAnyOrigin
    ],
    batch_dones: LayoutTensor[
        dtype, Layout.row_major((H + N_TD) * BATCH), MutAnyOrigin
    ],
    batch_policies: LayoutTensor[
        dtype,
        Layout.row_major((H + N_TD + 1) * BATCH * ACT_DIM),
        MutAnyOrigin,
    ],
    batch_values: LayoutTensor[
        dtype, Layout.row_major((H + N_TD + 1) * BATCH), MutAnyOrigin
    ],
    batch_to_play: LayoutTensor[
        DType.uint8,
        Layout.row_major((H + N_TD + 1) * BATCH),
        MutAnyOrigin,
    ],
    # Buffer state
    buf_size: Scalar[DType.int32],
    buf_write_idx: Scalar[DType.int32],
    rng_seed: Scalar[DType.uint32],
):
    """Sample BATCH sequences for MuZero K-step unrolled training.

    Layout (time-major so forward pass sees contiguous BATCH-sized slices):
      obs[t, b, *]:        t in [0, H+N_TD]   — H+1 unroll + N_TD bootstrap
      policies[t, b, *]:   t in [0, H+N_TD]
      actions[t, b, *]:    t in [0, H)
      rewards[t, b]:       t in [0, H+N_TD)
      dones[t, b]:         t in [0, H+N_TD)
      values[t, b]:        t in [0, H+N_TD]
      to_play[t, b]:       t in [0, H+N_TD]

    The extra H+1..H+N_TD timesteps for rewards/dones/values let the
    n-step kernel sum N_TD rewards from any base position k in [0, H]
    and fetch a bootstrap value at k+N_TD. The extra obs/policies are
    used by reanalyze (fresh forward pass at bootstrap positions).
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= BATCH:
        return

    var env_idx = tid % N_ENVS
    var sz = Int(buf_size)
    var wptr = Int(buf_write_idx)

    comptime WIN = H + N_TD  # Window length in timesteps minus 1

    # Rejection sampling to find valid sequence start.
    # We need WIN+1 valid timesteps available in the buffer.
    var philox = PhiloxRandom(
        seed=UInt64(rng_seed) + UInt64(tid * 137 + 1),
        offset=0,
    )
    var start = -1
    var max_start = sz - WIN - 1
    if max_start < 0:
        max_start = 0

    for _attempt in range(64):
        var rand_vals = philox.step_uniform()
        var candidate = Int(
            Scalar[dtype](rand_vals[0]) * Scalar[dtype](max_start + 1)
        )
        if candidate > max_start:
            candidate = max_start
        var actual = (wptr - sz + candidate + PER_ENV_CAP) % PER_ENV_CAP

        # Check no episode boundary in unroll window [0, H-1]; the
        # bootstrap window [H, H+N_TD-1] may cross terminal — the n-step
        # kernel handles that via the dones stream.
        var valid = True
        for t in range(H - 1):
            var idx = (actual + t) % PER_ENV_CAP
            if buf_dones[env_idx * PER_ENV_CAP + idx] > Scalar[dtype](0.5):
                valid = False
                break

        if valid:
            start = actual
            break

    if start < 0:
        start = (wptr - sz + PER_ENV_CAP) % PER_ENV_CAP

    # ── Gather full window (time-major output) ────────────────────────
    # obs / policies / values / to_play: t in [0, H+N_TD]
    for t in range(WIN + 1):
        var buf_idx = (start + t) % PER_ENV_CAP
        var env_obs_base = env_idx * PER_ENV_CAP * OBS_DIM + buf_idx * OBS_DIM
        var obs_out_base = t * BATCH * OBS_DIM + tid * OBS_DIM
        for d in range(OBS_DIM):
            batch_obs[obs_out_base + d] = buf_obs[env_obs_base + d]

        var env_pol_base = env_idx * PER_ENV_CAP * ACT_DIM + buf_idx * ACT_DIM
        var pol_out_base = t * BATCH * ACT_DIM + tid * ACT_DIM
        for a in range(ACT_DIM):
            batch_policies[pol_out_base + a] = buf_policies[env_pol_base + a]

        batch_values[t * BATCH + tid] = buf_values[
            env_idx * PER_ENV_CAP + buf_idx
        ]
        batch_to_play[t * BATCH + tid] = buf_to_play[
            env_idx * PER_ENV_CAP + buf_idx
        ]

    # actions / rewards / dones: t in [0, H+N_TD)
    for t in range(WIN):
        var buf_idx = (start + t) % PER_ENV_CAP
        # Actions only for t in [0, H) — only H actions are used in unroll
        comptime if False:
            pass
        if t < H:
            var env_act_base = (
                env_idx * PER_ENV_CAP * ACT_DIM + buf_idx * ACT_DIM
            )
            var act_out_base = t * BATCH * ACT_DIM + tid * ACT_DIM
            for d in range(ACT_DIM):
                batch_actions[act_out_base + d] = buf_actions[
                    env_act_base + d
                ]

        batch_rewards[t * BATCH + tid] = buf_rewards[
            env_idx * PER_ENV_CAP + buf_idx
        ]
        batch_dones[t * BATCH + tid] = buf_dones[
            env_idx * PER_ENV_CAP + buf_idx
        ]


# ═══════════════════════════════════════════════════════════════════════════
# N-Step Target Computation (on GPU)
# ═══════════════════════════════════════════════════════════════════════════


def nstep_value_targets_kernel[
    BATCH: Int,
    K: Int,
    N: Int,
    dtype: DType where dtype.is_floating_point(),
    BACKUP_TYPE: Int = 0,  # 0=N-step bootstrap, 1=Monte Carlo, 2=Lambda
](
    value_targets: LayoutTensor[
        dtype, Layout.row_major((K + 1) * BATCH), MutAnyOrigin
    ],
    reward_targets: LayoutTensor[
        dtype, Layout.row_major(K * BATCH), MutAnyOrigin
    ],
    batch_rewards: LayoutTensor[
        dtype, Layout.row_major((K + N) * BATCH), MutAnyOrigin
    ],
    batch_dones: LayoutTensor[
        dtype, Layout.row_major((K + N) * BATCH), MutAnyOrigin
    ],
    batch_values: LayoutTensor[
        dtype, Layout.row_major((K + N + 1) * BATCH), MutAnyOrigin
    ],
    batch_to_play: LayoutTensor[
        DType.uint8, Layout.row_major((K + N + 1) * BATCH), MutAnyOrigin
    ],
    gamma: Scalar[dtype],
):
    """Compute n-step bootstrapped value targets and reward targets.

    One thread per (batch, k) pair. Total threads: BATCH * (K+1).
    Value targets use n-step bootstrap with MCTS root values.
    Reward targets are just the raw rewards.

    All input/output tensors use TIME-MAJOR layout: index = t * BATCH + b.
    The reward/done/value/to_play windows extend to length K+N (or K+N+1
    for value/to_play) so the kernel can sum N rewards from any base
    position k in [0, K] and access a bootstrap value at k+N. This
    matches the muzero-general convention where the n-step horizon is
    independent of the unroll length.

    Two-player sign flip (muzero-general/replay_buffer.py:242-259):
    rewards and the bootstrap value are stored from the perspective of
    the player-to-move at each step. When the target's perspective
    player (batch_to_play[k * BATCH + b]) differs, we negate. For
    single-player envs (batch_to_play all zeros) this is a no-op.
    """
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * (K + 1):
        return

    var b = idx % BATCH
    var k = idx // BATCH
    var perspective = rebind[Scalar[DType.uint8]](
        batch_to_play[k * BATCH + b]
    )

    # Value target: n-step bootstrapped return
    var n_step_return = Scalar[dtype](0.0)
    var gamma_power = Scalar[dtype](1.0)
    var steps_used = 0
    var hit_terminal = False

    for i in range(N):
        var step_k = k + i
        # Window covers up to K+N steps; never out of bounds for k in [0,K].
        var rew = rebind[Scalar[dtype]](batch_rewards[step_k * BATCH + b])
        var step_player = rebind[Scalar[DType.uint8]](
            batch_to_play[step_k * BATCH + b]
        )
        if step_player != perspective:
            rew = -rew
        n_step_return += gamma_power * rew
        gamma_power *= gamma
        steps_used += 1

        if rebind[Scalar[dtype]](batch_dones[step_k * BATCH + b]) > Scalar[
            dtype
        ](0.5):
            hit_terminal = True
            break

    # Bootstrap with MCTS value if not terminal and full N steps consumed.
    # Monte Carlo (BACKUP_TYPE=1) never bootstraps — uses full episode return.
    comptime if BACKUP_TYPE != 1:
        if not hit_terminal and steps_used == N:
            var boot_k = k + N  # always in [N, K+N], within window
            var boot_v = rebind[Scalar[dtype]](
                batch_values[boot_k * BATCH + b]
            )
            var boot_player = rebind[Scalar[DType.uint8]](
                batch_to_play[boot_k * BATCH + b]
            )
            if boot_player != perspective:
                boot_v = -boot_v
            n_step_return += gamma_power * boot_v

    value_targets[k * BATCH + b] = n_step_return

    # Reward target (only for k < K). Reward target is in the perspective
    # of the step's player (predicted reward the dynamics net outputs);
    # no sign flip here.
    if k < K:
        reward_targets[k * BATCH + b] = batch_rewards[k * BATCH + b]


# ═══════════════════════════════════════════════════════════════════════════
# Value Distribution Decode (used by GPU reanalyze)
# ═══════════════════════════════════════════════════════════════════════════


def decode_value_dist_kernel[
    N: Int,  # number of samples to decode (= timesteps * batch)
    BINS: Int,
    PRED_OUT: Int,  # full prediction output dim (= ACT + BINS)
    ACT_DIM: Int,
    dtype: DType where dtype.is_floating_point(),
](
    pred_logits: LayoutTensor[
        dtype, Layout.row_major(N * PRED_OUT), MutAnyOrigin
    ],
    value_out: LayoutTensor[dtype, Layout.row_major(N), MutAnyOrigin],
    v_min: Scalar[dtype],
    v_max: Scalar[dtype],
    eps: Scalar[dtype] = Scalar[dtype](0.001),
):
    """Decode a value distribution to a scalar via softmax expectation +
    inverse MuZero scalar transform.

    Each row of `pred_logits` is the prediction-net output for one sample
    in TIME-MAJOR layout (i.e. row order matches batch_values layout).
    The last BINS values per row are the value-distribution logits over
    bins evenly spaced in [v_min, v_max]; the first ACT_DIM values are
    the policy logits and are ignored here.

    For row r:
      p_i = softmax(value_logits[r])
      h(v) = sum_i p_i * (v_min + i * step)        # expectation in transformed space
      v    = inverse_h(h(v))                        # original scalar value
    where inverse_h is muzero-general's inverse scalar transform with the
    same eps used in the forward transform.

    Used by GPU reanalyze (use_last_model_value): replaces stale
    `batch_values` with fresh predictions from the current network so
    the n-step kernel can bootstrap with up-to-date values.
    """
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= N:
        return

    var base = idx * PRED_OUT + ACT_DIM  # start of value logits

    # Numerically stable softmax over BINS logits
    var max_l = rebind[Scalar[dtype]](pred_logits[base])
    for i in range(1, BINS):
        var li = rebind[Scalar[dtype]](pred_logits[base + i])
        if li > max_l:
            max_l = li
    var sum_exp = Scalar[dtype](0.0)
    for i in range(BINS):
        sum_exp += exp(
            rebind[Scalar[dtype]](pred_logits[base + i]) - max_l
        )

    # Expectation under softmax: h(v) (transformed-space scalar)
    # NB: avoid Float64 intermediates — Apple Metal rejects f64.
    var step = (v_max - v_min) / Scalar[dtype](Int32(BINS - 1))
    var hv = Scalar[dtype](0.0)
    for i in range(BINS):
        var p = exp(
            rebind[Scalar[dtype]](pred_logits[base + i]) - max_l
        ) / sum_exp
        var atom = v_min + Scalar[dtype](Int32(i)) * step
        hv += p * atom

    # Inverse MuZero scalar transform: closed-form inverse of
    # h(x) = sign(x)*(sqrt(|x|+1)-1) + eps*x
    var sign = Scalar[dtype](1.0) if hv >= Scalar[dtype](0.0) else Scalar[
        dtype
    ](-1.0)
    var abs_hv = hv if hv >= Scalar[dtype](0.0) else -hv
    var inner = sqrt(
        Scalar[dtype](1.0) + Scalar[dtype](4.0) * eps * (abs_hv + Scalar[
            dtype
        ](1.0) + eps)
    )
    var f = (inner - Scalar[dtype](1.0)) / (Scalar[dtype](2.0) * eps)
    value_out[idx] = sign * (f * f - Scalar[dtype](1.0))


# ═══════════════════════════════════════════════════════════════════════════
# Scalar Transform (GPU)
# ═══════════════════════════════════════════════════════════════════════════


def scalar_transform_kernel[
    SIZE: Int,
    dtype: DType where dtype.is_floating_point(),
](
    data: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    eps: Scalar[dtype],
):
    """Apply MuZero scalar transform in-place: h(x) = sign(x)(sqrt(|x|+1)-1) + eps*x.

    One thread per element.
    """
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SIZE:
        return

    var x = rebind[Scalar[dtype]](data[idx])
    var sign = Scalar[dtype](1.0) if x >= Scalar[dtype](0.0) else Scalar[dtype](
        -1.0
    )
    var abs_x = x if x >= Scalar[dtype](0.0) else -x
    data[idx] = (
        sign * (sqrt(abs_x + Scalar[dtype](1.0)) - Scalar[dtype](1.0)) + eps * x
    )
