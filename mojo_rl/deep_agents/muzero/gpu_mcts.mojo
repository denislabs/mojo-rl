"""GPU MCTS — Fully GPU-resident Monte Carlo Tree Search for MuZero.

Root parallelization: one independent MCTS tree per environment, all running
in parallel on GPU. Network evaluations (dynamics + prediction) are batched
across all environments.

Architecture (per simulation round, all n_envs in parallel):
  1. GPU kernel: PUCT selection [1 thread/env] → pending expansion buffer
  2. GPU kernel: build dynamics input [n_envs × (LATENT + ACT)]
  3. GPU forward: dynamics [BATCH=n_envs] → next hidden + reward
  4. GPU forward: prediction [BATCH=n_envs] → policy + value
  5. GPU kernel: expand node + set priors [1 thread/env]
  6. GPU kernel: backup [1 thread/env]
  After all simulations:
  7. GPU kernel: visit counts → action selection

No CPU↔GPU sync during MCTS — only the final actions are read back.

References:
  - Root parallelization: Wikipedia MCTS article
  - GPU Monte Carlo: Buzer & Cazenave, 2023 (LION 17) — 420x speedup
  - MuZero: Schrittwieser et al., 2020 (Nature)
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.random.philox import Random as PhiloxRandom
from std.math import sqrt, log, exp
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model.model import Model
from mojo_rl.nn.optimizer.optimizer import Optimizer
from mojo_rl.nn.training import Network, GPUNetworkState

comptime TPB: Int = 256
comptime MAX_DEPTH: Int = 32  # Maximum tree depth for search paths


# ═══════════════════════════════════════════════════════════════════════════
# GPU Tree Storage Layout
# ═══════════════════════════════════════════════════════════════════════════
#
# All n_envs trees stored in flat device buffers with strided layout:
#   node_visit_count: [n_envs × MAX_NODES × ACT]  (int32 stored as dtype)
#   node_total_value: [n_envs × MAX_NODES × ACT]
#   node_prior:       [n_envs × MAX_NODES × ACT]
#   node_reward:      [n_envs × MAX_NODES × ACT]
#   node_child_idx:   [n_envs × MAX_NODES × ACT]  (-1 = unexpanded)
#   node_total_visits: [n_envs × MAX_NODES]
#   hidden_states:    [n_envs × MAX_NODES × LATENT]
#   node_count:       [n_envs]  (number of nodes per tree)
#
# Indexing: tree_offset = env_idx * MAX_NODES
#           node_act_offset = (tree_offset + node_idx) * ACT + action


# ═══════════════════════════════════════════════════════════════════════════
# GPU MCTS Kernels
# ═══════════════════════════════════════════════════════════════════════════


def gpu_mcts_init_root_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    LATENT: Int,
    PRED_OUT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    # Node storage
    visit_count: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    prior: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    reward: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    child_idx: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_visits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ],
    node_count: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Prediction output (from representation + prediction forward on root obs)
    pred_output: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PRED_OUT), MutAnyOrigin
    ],
    # MinMax tracking
    min_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    max_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Dirichlet noise
    noise_fraction: Scalar[dtype],
    rng_seed: Scalar[DType.uint32],
):
    """Initialize root node for each env's tree from prediction output.

    Sets prior from softmax of policy logits, adds Dirichlet noise.
    One thread per environment.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT

    # Zero all node data for this env's tree
    for n in range(MAX_NODES):
        for a in range(ACT):
            var idx = tree_off + n * ACT + a
            visit_count[idx] = Scalar[dtype](0.0)
            total_value[idx] = Scalar[dtype](0.0)
            prior[idx] = Scalar[dtype](1.0) / Scalar[dtype](ACT)
            reward[idx] = Scalar[dtype](0.0)
            child_idx[idx] = Scalar[dtype](-1.0)
        total_visits[e * MAX_NODES + n] = Scalar[dtype](0.0)

    # Set root prior from softmax of policy logits
    var pred_off = e * PRED_OUT
    var max_logit = rebind[Scalar[dtype]](pred_output[pred_off])
    for a in range(1, ACT):
        var v = rebind[Scalar[dtype]](pred_output[pred_off + a])
        if v > max_logit:
            max_logit = v
    var sum_exp = Scalar[dtype](0.0)
    for a in range(ACT):
        var v = exp(
            rebind[Scalar[dtype]](pred_output[pred_off + a]) - max_logit
        )
        prior[tree_off + a] = v
        sum_exp += v
    for a in range(ACT):
        prior[tree_off + a] = (
            rebind[Scalar[dtype]](prior[tree_off + a]) / sum_exp
        )

    # Add Dirichlet noise (approximate with Exponential(1) / sum)
    var philox = PhiloxRandom(
        seed=UInt64(rng_seed) + UInt64(e * 137 + 1), offset=0
    )
    var noise_sum = Scalar[dtype](0.0)
    for a in range(ACT):
        var rand_vals = philox.step_uniform()
        var noise_val = -log(Scalar[dtype](rand_vals[0]) + Scalar[dtype](1e-8))
        reward[
            tree_off + a
        ] = noise_val  # Temporarily store noise in reward[root]
        noise_sum += noise_val
    for a in range(ACT):
        var noise_val = rebind[Scalar[dtype]](reward[tree_off + a]) / noise_sum
        prior[tree_off + a] = (Scalar[dtype](1.0) - noise_fraction) * rebind[
            Scalar[dtype]
        ](prior[tree_off + a]) + noise_fraction * noise_val
        reward[tree_off + a] = Scalar[dtype](0.0)  # Reset reward

    # Initialize tree state
    node_count[e] = Scalar[dtype](1.0)  # Root is node 0
    min_q[e] = Scalar[dtype](1e18)
    max_q[e] = Scalar[dtype](-1e18)


def gpu_mcts_select_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    # Node storage (read)
    visit_count: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    prior: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    child_idx: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_visits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ],
    node_count: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # MinMax Q normalization
    min_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    max_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Output: pending expansion info
    pending_parent: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    pending_action: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Output: search paths for backup
    search_paths: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin
    ],
    action_paths: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin
    ],
    path_lengths: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # PUCT constants
    c_base: Scalar[dtype],
    c_init: Scalar[dtype],
):
    """PUCT selection: traverse each env's tree from root to leaf.

    One thread per environment. Sequential tree traversal within each tree,
    parallel across all n_envs trees.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var tv_off = e * MAX_NODES
    var path_off = e * MAX_DEPTH

    var node_idx = 0
    var depth = 0
    search_paths[path_off] = Scalar[dtype](0.0)

    var mn = rebind[Scalar[dtype]](min_q[e])
    var mx = rebind[Scalar[dtype]](max_q[e])
    var q_range = mx - mn

    while depth < MAX_DEPTH - 1:
        # PUCT selection
        var n_total = rebind[Scalar[dtype]](total_visits[tv_off + node_idx])
        var sqrt_total = sqrt(n_total + Scalar[dtype](1e-8))
        var c = log((Scalar[dtype](1.0) + n_total + c_base) / c_base) + c_init

        var best_action = 0
        var best_score = Scalar[dtype](-1e18)

        for a in range(ACT):
            var na_off = tree_off + node_idx * ACT + a
            var n_a = rebind[Scalar[dtype]](visit_count[na_off])

            var q_val: Scalar[dtype]
            if n_a > Scalar[dtype](0.5):
                q_val = rebind[Scalar[dtype]](total_value[na_off]) / n_a
                # MinMax normalize
                if q_range > Scalar[dtype](1e-8):
                    q_val = (q_val - mn) / q_range
            else:
                q_val = Scalar[dtype](0.0)

            var p = rebind[Scalar[dtype]](prior[na_off])
            var exploration = c * p * sqrt_total / (Scalar[dtype](1.0) + n_a)
            var score = q_val + exploration

            if score > best_score:
                best_score = score
                best_action = a

        action_paths[path_off + depth] = Scalar[dtype](best_action)

        # Check if child exists
        var child = rebind[Scalar[dtype]](
            child_idx[tree_off + node_idx * ACT + best_action]
        )
        if child < Scalar[dtype](0.0):
            # Leaf — record pending expansion
            pending_parent[e] = Scalar[dtype](node_idx)
            pending_action[e] = Scalar[dtype](best_action)
            path_lengths[e] = Scalar[dtype](depth + 1)
            return

        # Descend to child
        node_idx = Int(child)
        depth += 1
        search_paths[path_off + depth] = Scalar[dtype](node_idx)

    # Max depth reached — use current node as leaf
    pending_parent[e] = Scalar[dtype](node_idx)
    pending_action[e] = Scalar[dtype](0)
    path_lengths[e] = Scalar[dtype](depth + 1)


def gpu_mcts_expand_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    LATENT: Int,
    PRED_OUT: Int,
    DYN_OUT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    # Node storage (write)
    visit_count: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    prior: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    reward_buf: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    child_idx: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_visits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ],
    node_count: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Hidden state pool
    hidden_states: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * LATENT), MutAnyOrigin
    ],
    # Pending expansion info
    pending_parent: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    pending_action: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Network outputs
    dyn_output: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * DYN_OUT), MutAnyOrigin
    ],
    pred_output: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PRED_OUT), MutAnyOrigin
    ],
    # Value support
    v_min: Scalar[dtype],
    v_max: Scalar[dtype],
    # Output: leaf values for backup
    leaf_values: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
):
    """Create child nodes from dynamics/prediction outputs.

    One thread per environment. Each thread:
    1. Extracts hidden state from dynamics output → hidden pool
    2. Scales hidden state [0,1]
    3. Extracts reward from dynamics output
    4. Sets child prior from prediction softmax
    5. Links parent → child
    6. Decodes leaf value from prediction output
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var tv_off = e * MAX_NODES
    var h_off = e * MAX_NODES * LATENT

    var parent = Int(rebind[Scalar[dtype]](pending_parent[e]))
    var action = Int(rebind[Scalar[dtype]](pending_action[e]))
    var child_node_idx = Int(rebind[Scalar[dtype]](node_count[e]))

    # Guard against overflow
    if child_node_idx >= MAX_NODES:
        leaf_values[e] = Scalar[dtype](0.0)
        return

    # 1. Extract hidden state from dynamics output
    var dyn_off = e * DYN_OUT
    var child_h_off = h_off + child_node_idx * LATENT
    for i in range(LATENT):
        hidden_states[child_h_off + i] = dyn_output[dyn_off + i]

    # 2. Scale hidden state to [0,1]
    var h_min = rebind[Scalar[dtype]](hidden_states[child_h_off])
    var h_max = h_min
    for i in range(1, LATENT):
        var v = rebind[Scalar[dtype]](hidden_states[child_h_off + i])
        if v < h_min:
            h_min = v
        if v > h_max:
            h_max = v
    var h_delta = h_max - h_min
    if h_delta > Scalar[dtype](1e-8):
        for i in range(LATENT):
            var v = rebind[Scalar[dtype]](hidden_states[child_h_off + i])
            hidden_states[child_h_off + i] = (v - h_min) / h_delta

    # 3. Extract reward from dynamics output
    comptime NUM_REW_BINS = DYN_OUT - LATENT
    var rew_decoded: Scalar[dtype]
    if NUM_REW_BINS == 1:
        # Scalar reward: read raw value, tanh for bounding
        var raw = rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT])
        var e_pos = exp(raw)
        var e_neg = exp(-raw)
        rew_decoded = (e_pos - e_neg) / (e_pos + e_neg)  # tanh
    else:
        # Categorical reward: softmax + expectation over bins
        var rew_max_val = rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT])
        for i in range(1, NUM_REW_BINS):
            var v = rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT + i])
            if v > rew_max_val:
                rew_max_val = v
        var rew_sum_exp = Scalar[dtype](0.0)
        for i in range(NUM_REW_BINS):
            rew_sum_exp += exp(
                rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT + i])
                - rew_max_val
            )
        rew_decoded = Scalar[dtype](0.0)
        var rew_step = (v_max - v_min) / Scalar[dtype](NUM_REW_BINS - 1)
        for i in range(NUM_REW_BINS):
            var prob = (
                exp(
                    rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT + i])
                    - rew_max_val
                )
                / rew_sum_exp
            )
            rew_decoded += prob * (v_min + Scalar[dtype](i) * rew_step)
        # Inverse scalar transform
        var sign_r = Scalar[dtype](1.0) if rew_decoded >= Scalar[dtype](
            0.0
        ) else Scalar[dtype](-1.0)
        var abs_r = (
            rew_decoded if rew_decoded >= Scalar[dtype](0.0) else -rew_decoded
        )
        var eps_r = Scalar[dtype](0.001)
        var inner_r = sqrt(
            Scalar[dtype](1.0)
            + Scalar[dtype](4.0) * eps_r * (abs_r + Scalar[dtype](1.0) + eps_r)
        )
        var f_r = (inner_r - Scalar[dtype](1.0)) / (Scalar[dtype](2.0) * eps_r)
        rew_decoded = sign_r * (f_r * f_r - Scalar[dtype](1.0))

    # Set parent reward
    reward_buf[tree_off + parent * ACT + action] = rew_decoded

    # 4. Set child prior from prediction softmax
    var pred_off = e * PRED_OUT
    var child_tree_off = tree_off + child_node_idx * ACT
    var p_max = rebind[Scalar[dtype]](pred_output[pred_off])
    for a in range(1, ACT):
        var v = rebind[Scalar[dtype]](pred_output[pred_off + a])
        if v > p_max:
            p_max = v
    var p_sum = Scalar[dtype](0.0)
    for a in range(ACT):
        var v = exp(rebind[Scalar[dtype]](pred_output[pred_off + a]) - p_max)
        prior[child_tree_off + a] = v
        p_sum += v
    for a in range(ACT):
        prior[child_tree_off + a] = (
            rebind[Scalar[dtype]](prior[child_tree_off + a]) / p_sum
        )
        visit_count[child_tree_off + a] = Scalar[dtype](0.0)
        total_value[child_tree_off + a] = Scalar[dtype](0.0)
        reward_buf[child_tree_off + a] = Scalar[dtype](0.0)
        child_idx[child_tree_off + a] = Scalar[dtype](-1.0)
    total_visits[tv_off + child_node_idx] = Scalar[dtype](0.0)

    # 5. Link parent → child
    child_idx[tree_off + parent * ACT + action] = Scalar[dtype](child_node_idx)
    node_count[e] = Scalar[dtype](child_node_idx + 1)

    # 6. Decode leaf value
    comptime NUM_VAL_BINS = PRED_OUT - ACT
    if NUM_VAL_BINS == 1:
        # Scalar value: read raw output, tanh for [-1, 1] bounding
        var raw_v = rebind[Scalar[dtype]](pred_output[pred_off + ACT])
        var ev_pos = exp(raw_v)
        var ev_neg = exp(-raw_v)
        leaf_values[e] = (ev_pos - ev_neg) / (ev_pos + ev_neg)  # tanh
    else:
        # Categorical value: softmax + expectation over bins
        var val_max = rebind[Scalar[dtype]](pred_output[pred_off + ACT])
        for i in range(1, NUM_VAL_BINS):
            var v = rebind[Scalar[dtype]](pred_output[pred_off + ACT + i])
            if v > val_max:
                val_max = v
        var val_sum_exp = Scalar[dtype](0.0)
        for i in range(NUM_VAL_BINS):
            val_sum_exp += exp(
                rebind[Scalar[dtype]](pred_output[pred_off + ACT + i]) - val_max
            )
        var val_step = (v_max - v_min) / Scalar[dtype](NUM_VAL_BINS - 1)
        var val_decoded = Scalar[dtype](0.0)
        for i in range(NUM_VAL_BINS):
            var prob = (
                exp(
                    rebind[Scalar[dtype]](pred_output[pred_off + ACT + i])
                    - val_max
                )
                / val_sum_exp
            )
            val_decoded += prob * (v_min + Scalar[dtype](i) * val_step)

        # Inverse scalar transform: h^{-1}(y)
        var sign = Scalar[dtype](1.0) if val_decoded >= Scalar[dtype](
            0.0
        ) else Scalar[dtype](-1.0)
        var abs_y = (
            val_decoded if val_decoded >= Scalar[dtype](0.0) else -val_decoded
        )
        var eps = Scalar[dtype](0.001)
        var inner = sqrt(
            Scalar[dtype](1.0)
            + Scalar[dtype](4.0) * eps * (abs_y + Scalar[dtype](1.0) + eps)
        )
        var f = (inner - Scalar[dtype](1.0)) / (Scalar[dtype](2.0) * eps)
        leaf_values[e] = sign * (f * f - Scalar[dtype](1.0))


def gpu_mcts_backup_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    # Node storage (read/write)
    visit_count: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    reward_buf: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_visits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ],
    min_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    max_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Search paths
    search_paths: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin
    ],
    action_paths: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin
    ],
    path_lengths: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    leaf_values: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    gamma: Scalar[dtype],
):
    """Backup leaf values through search paths.

    One thread per environment. Walks backwards from leaf to root,
    accumulating discounted returns and updating visit counts.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var tv_off = e * MAX_NODES
    var path_off = e * MAX_DEPTH
    var path_len = Int(rebind[Scalar[dtype]](path_lengths[e]))

    var value = rebind[Scalar[dtype]](leaf_values[e])

    # Walk backwards from leaf parent to root
    for i in range(path_len):
        var idx = path_len - 1 - i
        var node_idx = Int(rebind[Scalar[dtype]](search_paths[path_off + idx]))
        var action = Int(rebind[Scalar[dtype]](action_paths[path_off + idx]))

        var na_off = tree_off + node_idx * ACT + action
        value = rebind[Scalar[dtype]](reward_buf[na_off]) + gamma * value

        visit_count[na_off] = rebind[Scalar[dtype]](
            visit_count[na_off]
        ) + Scalar[dtype](1.0)
        total_value[na_off] = rebind[Scalar[dtype]](total_value[na_off]) + value
        total_visits[tv_off + node_idx] = rebind[Scalar[dtype]](
            total_visits[tv_off + node_idx]
        ) + Scalar[dtype](1.0)

        # Update MinMax
        var n_a = rebind[Scalar[dtype]](visit_count[na_off])
        var mean_q = rebind[Scalar[dtype]](total_value[na_off]) / n_a
        if mean_q < rebind[Scalar[dtype]](min_q[e]):
            min_q[e] = mean_q
        if mean_q > rebind[Scalar[dtype]](max_q[e]):
            max_q[e] = mean_q


def gpu_mcts_extract_actions_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    visit_count: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    actions_out: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    policies_out: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ],
):
    """Extract action from root visit counts (argmax) and policy (normalized visits).

    One thread per environment.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var root_off = e * MAX_NODES * ACT  # Root is always node 0

    # Sum visit counts and find argmax
    var total = Scalar[dtype](0.0)
    var best_action = 0
    var best_count = rebind[Scalar[dtype]](visit_count[root_off])
    for a in range(ACT):
        var count = rebind[Scalar[dtype]](visit_count[root_off + a])
        total += count
        if count > best_count:
            best_count = count
            best_action = a

    actions_out[e] = Scalar[dtype](best_action)

    # Normalized policy
    if total > Scalar[dtype](0.5):
        for a in range(ACT):
            policies_out[e * ACT + a] = (
                rebind[Scalar[dtype]](visit_count[root_off + a]) / total
            )
    else:
        for a in range(ACT):
            policies_out[e * ACT + a] = Scalar[dtype](1.0) / Scalar[dtype](ACT)


# ═══════════════════════════════════════════════════════════════════════════
# Self-Play Kernels (legal masking + negated backup)
# ═══════════════════════════════════════════════════════════════════════════


def gpu_mcts_apply_legal_mask_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    prior: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    legal_masks: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ],
):
    """Mask root prior with legal action mask and renormalize.

    For board games: zero out illegal actions in root prior, then
    renormalize so probabilities sum to 1.

    One thread per environment. Call AFTER init_root, BEFORE simulations.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var root_off = e * MAX_NODES * ACT  # Root is always node 0

    # Apply mask
    var sum_p = Scalar[dtype](0.0)
    for a in range(ACT):
        var legal = rebind[Scalar[dtype]](legal_masks[e * ACT + a])
        if legal < Scalar[dtype](0.5):
            prior[root_off + a] = Scalar[dtype](0.0)
        else:
            sum_p += rebind[Scalar[dtype]](prior[root_off + a])

    # Renormalize
    if sum_p > Scalar[dtype](1e-8):
        for a in range(ACT):
            prior[root_off + a] = (
                rebind[Scalar[dtype]](prior[root_off + a]) / sum_p
            )
    else:
        # All actions masked — uniform over legal actions (fallback)
        var n_legal = Scalar[dtype](0.0)
        for a in range(ACT):
            if rebind[Scalar[dtype]](legal_masks[e * ACT + a]) > Scalar[dtype](
                0.5
            ):
                n_legal += Scalar[dtype](1.0)
        if n_legal > Scalar[dtype](0.5):
            var inv_n = Scalar[dtype](1.0) / n_legal
            for a in range(ACT):
                if rebind[Scalar[dtype]](legal_masks[e * ACT + a]) > Scalar[
                    dtype
                ](0.5):
                    prior[root_off + a] = inv_n


def gpu_mcts_backup_negated_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    # Node storage (read/write)
    visit_count: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    reward_buf: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_visits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ],
    min_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    max_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Search paths
    search_paths: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin
    ],
    action_paths: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin
    ],
    path_lengths: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    leaf_values: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
):
    """Backup with value negation for two-player zero-sum games.

    At each backup level, the value is NEGATED (parent's perspective
    is opposite to child's perspective). No discount (gamma=1 for board games).
    No per-step reward (only terminal reward).

    value = -value  at each level

    One thread per environment.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var tv_off = e * MAX_NODES
    var path_off = e * MAX_DEPTH
    var path_len = Int(rebind[Scalar[dtype]](path_lengths[e]))

    var value = rebind[Scalar[dtype]](leaf_values[e])

    # Walk backwards from leaf parent to root, NEGATING at each level
    for i in range(path_len):
        var idx = path_len - 1 - i
        var node_idx = Int(rebind[Scalar[dtype]](search_paths[path_off + idx]))
        var action = Int(rebind[Scalar[dtype]](action_paths[path_off + idx]))

        # Negate value (parent sees opposite of child)
        value = -value

        var na_off = tree_off + node_idx * ACT + action
        visit_count[na_off] = rebind[Scalar[dtype]](
            visit_count[na_off]
        ) + Scalar[dtype](1.0)
        total_value[na_off] = rebind[Scalar[dtype]](total_value[na_off]) + value
        total_visits[tv_off + node_idx] = rebind[Scalar[dtype]](
            total_visits[tv_off + node_idx]
        ) + Scalar[dtype](1.0)

        # Update MinMax
        var n_a = rebind[Scalar[dtype]](visit_count[na_off])
        var mean_q = rebind[Scalar[dtype]](total_value[na_off]) / n_a
        if mean_q < rebind[Scalar[dtype]](min_q[e]):
            min_q[e] = mean_q
        if mean_q > rebind[Scalar[dtype]](max_q[e]):
            max_q[e] = mean_q


def gpu_mcts_extract_actions_masked_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    visit_count: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    legal_masks: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ],
    actions_out: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    policies_out: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ],
):
    """Extract actions respecting legal mask — choose only among legal actions.

    One thread per environment.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var root_off = e * MAX_NODES * ACT

    var total = Scalar[dtype](0.0)
    var best_action = -1
    var best_count = Scalar[dtype](-1.0)

    for a in range(ACT):
        var legal = rebind[Scalar[dtype]](legal_masks[e * ACT + a])
        if legal > Scalar[dtype](0.5):
            var count = rebind[Scalar[dtype]](visit_count[root_off + a])
            total += count
            if count > best_count or best_action < 0:
                best_count = count
                best_action = a

    if best_action < 0:
        best_action = 0  # Fallback (shouldn't happen with valid game state)

    actions_out[e] = Scalar[dtype](best_action)

    # Normalized policy (only over legal actions)
    if total > Scalar[dtype](0.5):
        for a in range(ACT):
            var legal = rebind[Scalar[dtype]](legal_masks[e * ACT + a])
            if legal > Scalar[dtype](0.5):
                policies_out[e * ACT + a] = (
                    rebind[Scalar[dtype]](visit_count[root_off + a]) / total
                )
            else:
                policies_out[e * ACT + a] = Scalar[dtype](0.0)
    else:
        for a in range(ACT):
            policies_out[e * ACT + a] = Scalar[dtype](0.0)
        policies_out[e * ACT + best_action] = Scalar[dtype](1.0)


def gpu_mcts_extract_actions_temp_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    visit_count: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    legal_masks: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ],
    ep_steps: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    actions_out: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    policies_out: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ],
    temp_threshold: Int,
    rng_seed: Scalar[DType.uint32],
):
    """Extract actions with temperature annealing.

    Like alpha-zero-general:
      - First temp_threshold moves: temp=1, sample proportionally from visit counts
      - After temp_threshold: temp=0, pick argmax (one-hot policy target)

    One thread per environment.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var root_off = e * MAX_NODES * ACT
    var move_count = Int(rebind[Scalar[dtype]](ep_steps[e]))

    # Compute visit counts for legal actions
    var total = Scalar[dtype](0.0)
    var best_action = -1
    var best_count = Scalar[dtype](-1.0)

    for a in range(ACT):
        var legal = rebind[Scalar[dtype]](legal_masks[e * ACT + a])
        if legal > Scalar[dtype](0.5):
            var count = rebind[Scalar[dtype]](visit_count[root_off + a])
            total += count
            if count > best_count or best_action < 0:
                best_count = count
                best_action = a

    if best_action < 0:
        best_action = 0

    if move_count < temp_threshold and total > Scalar[dtype](0.5):
        # Temperature = 1: proportional policy + sample proportionally
        # Output proportional policy
        for a in range(ACT):
            var legal = rebind[Scalar[dtype]](legal_masks[e * ACT + a])
            if legal > Scalar[dtype](0.5):
                policies_out[e * ACT + a] = (
                    rebind[Scalar[dtype]](visit_count[root_off + a]) / total
                )
            else:
                policies_out[e * ACT + a] = Scalar[dtype](0.0)

        # Sample action proportionally using PhiloxRandom
        var philox = PhiloxRandom(
            seed=UInt64(rng_seed) + UInt64(e * 7919 + move_count * 6271),
            offset=0,
        )
        var rand_vals = philox.step_uniform()
        var rand = Scalar[dtype](rand_vals[0])

        var cumsum = Scalar[dtype](0.0)
        var sampled = best_action  # Fallback to best
        for a in range(ACT):
            var legal = rebind[Scalar[dtype]](legal_masks[e * ACT + a])
            if legal > Scalar[dtype](0.5):
                cumsum += rebind[Scalar[dtype]](
                    visit_count[root_off + a]
                ) / total
                if rand < cumsum:
                    sampled = a
                    break
        actions_out[e] = Scalar[dtype](sampled)
    else:
        # Temperature = 0: greedy argmax + one-hot policy target
        actions_out[e] = Scalar[dtype](best_action)

        for a in range(ACT):
            policies_out[e * ACT + a] = Scalar[dtype](0.0)
        policies_out[e * ACT + best_action] = Scalar[dtype](1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Game State Kernels (AlphaZero mode — true game rules)
# ═══════════════════════════════════════════════════════════════════════════


def gpu_mcts_copy_parent_state_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    STATE_SIZE: Int,
    dtype: DType where dtype.is_floating_point(),
](
    expansion_states: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * STATE_SIZE), MutAnyOrigin
    ],
    game_states: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * STATE_SIZE), MutAnyOrigin
    ],
    pending_parent: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
):
    """Copy parent game state to expansion staging buffer.

    One thread per env. Copies STATE_SIZE floats from parent node's
    game state slot to the flat expansion buffer for env.step().
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var parent = Int(rebind[Scalar[dtype]](pending_parent[e]))
    var src_off = e * MAX_NODES * STATE_SIZE + parent * STATE_SIZE
    var dst_off = e * STATE_SIZE

    for i in range(STATE_SIZE):
        expansion_states[dst_off + i] = game_states[src_off + i]


def gpu_mcts_store_child_state_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    STATE_SIZE: Int,
    dtype: DType where dtype.is_floating_point(),
](
    game_states: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * STATE_SIZE), MutAnyOrigin
    ],
    expansion_states: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * STATE_SIZE), MutAnyOrigin
    ],
    node_count: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
):
    """Copy expansion result (after env.step) to child node's game state slot.

    One thread per env. The child index is node_count (before increment —
    expand kernel will increment it).
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var child_idx = Int(rebind[Scalar[dtype]](node_count[e]))
    if child_idx >= MAX_NODES:
        return

    var src_off = e * STATE_SIZE
    var dst_off = e * MAX_NODES * STATE_SIZE + child_idx * STATE_SIZE

    for i in range(STATE_SIZE):
        game_states[dst_off + i] = expansion_states[src_off + i]


def gpu_mcts_copy_root_state_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    STATE_SIZE: Int,
    dtype: DType where dtype.is_floating_point(),
](
    game_states: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * STATE_SIZE), MutAnyOrigin
    ],
    env_states: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * STATE_SIZE), MutAnyOrigin
    ],
):
    """Copy current env states to root node (node 0) of each env's tree.

    Called once at start of each MCTS search.
    One thread per env.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var src_off = e * STATE_SIZE
    var dst_off = e * MAX_NODES * STATE_SIZE  # Root = node 0

    for i in range(STATE_SIZE):
        game_states[dst_off + i] = env_states[src_off + i]


def gpu_mcts_expand_alphazero_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    PRED_OUT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    # Node storage
    visit_count: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    prior: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    reward_buf: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    child_idx: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_visits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ],
    node_count: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Pending expansion
    pending_parent: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    pending_action: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Prediction output [N_ENVS * PRED_OUT] = (policy_logits[ACT], value[1])
    pred_output: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PRED_OUT), MutAnyOrigin
    ],
    # Reward from env.step [N_ENVS]
    step_rewards: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Output
    leaf_values: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
):
    """AlphaZero-specific expand: uses env.step reward and scalar value with tanh.

    Unlike MuZero expand, this kernel:
    - Reads reward directly from env.step output (not dynamics network)
    - Reads value as a single scalar with tanh activation (not categorical)
    - Does NOT touch hidden_states (game states handled separately)
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var tv_off = e * MAX_NODES
    var parent = Int(rebind[Scalar[dtype]](pending_parent[e]))
    var action = Int(rebind[Scalar[dtype]](pending_action[e]))
    var child_node_idx = Int(rebind[Scalar[dtype]](node_count[e]))

    if child_node_idx >= MAX_NODES:
        leaf_values[e] = Scalar[dtype](0.0)
        return

    # Set parent reward from env.step
    reward_buf[tree_off + parent * ACT + action] = step_rewards[e]

    # Set child prior from softmax of policy logits
    var pred_off = e * PRED_OUT
    var child_off = tree_off + child_node_idx * ACT
    var p_max = rebind[Scalar[dtype]](pred_output[pred_off])
    for a in range(1, ACT):
        var v = rebind[Scalar[dtype]](pred_output[pred_off + a])
        if v > p_max:
            p_max = v
    var p_sum = Scalar[dtype](0.0)
    for a in range(ACT):
        var v = exp(rebind[Scalar[dtype]](pred_output[pred_off + a]) - p_max)
        prior[child_off + a] = v
        p_sum += v
    for a in range(ACT):
        prior[child_off + a] = (
            rebind[Scalar[dtype]](prior[child_off + a]) / p_sum
        )
        visit_count[child_off + a] = Scalar[dtype](0.0)
        total_value[child_off + a] = Scalar[dtype](0.0)
        reward_buf[child_off + a] = Scalar[dtype](0.0)
        child_idx[child_off + a] = Scalar[dtype](-1.0)
    total_visits[tv_off + child_node_idx] = Scalar[dtype](0.0)

    # Link parent → child
    child_idx[tree_off + parent * ACT + action] = Scalar[dtype](child_node_idx)
    node_count[e] = Scalar[dtype](child_node_idx + 1)

    # Decode leaf value:
    # If env.step returned done (game over), use step reward directly.
    # Otherwise, use tanh(network_output) as value estimate.
    var step_done = rebind[Scalar[dtype]](
        step_rewards[e]
    )  # reward doubles as done signal
    # Actually step_rewards is the reward. We need done flag separately.
    # For now: if reward is +1 or -1 (terminal), use it. If 0, use network.
    var abs_rew = step_done if step_done >= Scalar[dtype](0.0) else -step_done
    if abs_rew > Scalar[dtype](0.5):
        # Terminal state: use game outcome directly
        leaf_values[e] = step_done
    else:
        # Non-terminal: use network value prediction
        var raw_v = rebind[Scalar[dtype]](pred_output[pred_off + ACT])
        var ev_p = exp(raw_v)
        var ev_n = exp(-raw_v)
        leaf_values[e] = (ev_p - ev_n) / (ev_p + ev_n)


# ═══════════════════════════════════════════════════════════════════════════
# Batched Fused Kernels (BATCH_SIMS leaves per env per round)
# ═══════════════════════════════════════════════════════════════════════════


def gpu_mcts_batched_select_and_copy_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    BATCH_SIMS: Int,
    STATE_SIZE: Int,
    dtype: DType where dtype.is_floating_point(),
](
    # Node storage
    visit_count: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    prior: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    child_idx: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_visits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ],
    node_count: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    min_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    max_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Game states (for AlphaZero copy)
    game_states: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * STATE_SIZE), MutAnyOrigin
    ],
    # Output: BATCH_SIMS pending expansions per env
    pending_parents: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS), MutAnyOrigin
    ],
    pending_actions: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS), MutAnyOrigin
    ],
    expansion_states: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS * STATE_SIZE), MutAnyOrigin
    ],
    # Search paths for backup (only last sim's path stored for simplicity)
    search_paths: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS * MAX_DEPTH), MutAnyOrigin
    ],
    action_paths: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS * MAX_DEPTH), MutAnyOrigin
    ],
    path_lengths: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS), MutAnyOrigin
    ],
    # PUCT constants
    c_base: Scalar[dtype],
    c_init: Scalar[dtype],
):
    """Fused: select BATCH_SIMS leaves per env with virtual losses + copy parent game states.

    One thread per env. Each thread sequentially selects BATCH_SIMS leaves,
    applying virtual loss (+3 visits) after each selection to encourage diversity.
    Then copies parent game states to the expansion staging buffer.

    Combines gpu_mcts_select_kernel + gpu_mcts_copy_parent_state_kernel.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var tv_off = e * MAX_NODES
    var mn = rebind[Scalar[dtype]](min_q[e])
    var mx = rebind[Scalar[dtype]](max_q[e])
    var q_range = mx - mn

    comptime VIRTUAL_LOSS: Int = 3

    # Use shared memory for root visit counts to prevent NVIDIA compiler
    # from caching global memory reads across loop iterations.
    # Each thread gets ACT+1 slots: [tid*(ACT+1) .. tid*(ACT+1)+ACT) = per-action visits,
    # slot [tid*(ACT+1)+ACT] = total visits sum.
    from std.gpu.memory import AddressSpace
    comptime SLOTS_PER_THREAD = ACT + 1
    var s_root = LayoutTensor[
        dtype,
        Layout.row_major(TPB * SLOTS_PER_THREAD),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var tid = Int(thread_idx.x)
    var s_off = tid * SLOTS_PER_THREAD

    # Load root visit counts into shared memory
    var s_total = Scalar[dtype](0.0)
    for _a in range(ACT):
        var vc_val = rebind[Scalar[dtype]](visit_count[tree_off + _a])
        s_root[s_off + _a] = vc_val
        s_total += vc_val
    s_root[s_off + ACT] = s_total

    for s in range(BATCH_SIMS):
        # ── PUCT Selection ──────────────────────────────────
        var node_idx = 0
        var depth = 0
        var path_off = (e * BATCH_SIMS + s) * MAX_DEPTH
        search_paths[path_off] = Scalar[dtype](0.0)

        while depth < MAX_DEPTH - 1:
            # Use shared memory for root, global for deeper nodes
            var n_total: Scalar[dtype]
            if node_idx == 0:
                n_total = rebind[Scalar[dtype]](s_root[s_off + ACT])
            else:
                n_total = rebind[Scalar[dtype]](total_visits[tv_off + node_idx])
            var sqrt_total = sqrt(n_total + Scalar[dtype](1e-8))
            var c = (
                log((Scalar[dtype](1.0) + n_total + c_base) / c_base) + c_init
            )

            var best_action = 0
            var best_score = Scalar[dtype](-1e18)
            for a in range(ACT):
                var na_off = tree_off + node_idx * ACT + a
                # Use shared memory for root actions
                var n_a: Scalar[dtype]
                if node_idx == 0:
                    n_a = rebind[Scalar[dtype]](s_root[s_off + a])
                else:
                    n_a = rebind[Scalar[dtype]](visit_count[na_off])
                var q_val: Scalar[dtype]
                if n_a > Scalar[dtype](0.5):
                    q_val = rebind[Scalar[dtype]](total_value[na_off]) / n_a
                    if q_range > Scalar[dtype](1e-8):
                        q_val = (q_val - mn) / q_range
                else:
                    q_val = Scalar[dtype](0.0)
                var p = rebind[Scalar[dtype]](prior[na_off])
                var score = q_val + c * p * sqrt_total / (
                    Scalar[dtype](1.0) + n_a
                )
                if score > best_score:
                    best_score = score
                    best_action = a

            action_paths[path_off + depth] = Scalar[dtype](best_action)

            var child = rebind[Scalar[dtype]](
                child_idx[tree_off + node_idx * ACT + best_action]
            )
            if child < Scalar[dtype](0.0):
                # Leaf found
                var sim_off = e * BATCH_SIMS + s
                pending_parents[sim_off] = Scalar[dtype](node_idx)
                pending_actions[sim_off] = Scalar[dtype](best_action)
                path_lengths[sim_off] = Scalar[dtype](depth + 1)

                # Apply virtual loss to global memory
                visit_count[tree_off + node_idx * ACT + best_action] = rebind[
                    Scalar[dtype]
                ](
                    visit_count[tree_off + node_idx * ACT + best_action]
                ) + Scalar[
                    dtype
                ](
                    VIRTUAL_LOSS
                )
                total_visits[tv_off + node_idx] = rebind[Scalar[dtype]](
                    total_visits[tv_off + node_idx]
                ) + Scalar[dtype](VIRTUAL_LOSS)

                # Update shared memory for root
                if node_idx == 0:
                    s_root[s_off + best_action] = rebind[Scalar[dtype]](
                        s_root[s_off + best_action]
                    ) + Scalar[dtype](VIRTUAL_LOSS)
                    s_root[s_off + ACT] = rebind[Scalar[dtype]](
                        s_root[s_off + ACT]
                    ) + Scalar[dtype](VIRTUAL_LOSS)

                # ── Copy parent game state to expansion buffer ──
                var parent_gs_off = (
                    e * MAX_NODES * STATE_SIZE + node_idx * STATE_SIZE
                )
                var exp_off = sim_off * STATE_SIZE
                for i in range(STATE_SIZE):
                    expansion_states[exp_off + i] = game_states[
                        parent_gs_off + i
                    ]

                break

            node_idx = Int(child)
            depth += 1
            search_paths[path_off + depth] = Scalar[dtype](node_idx)

        # If max depth reached without finding leaf
        if depth >= MAX_DEPTH - 1:
            var sim_off = e * BATCH_SIMS + s
            pending_parents[sim_off] = Scalar[dtype](node_idx)
            pending_actions[sim_off] = Scalar[dtype](0)
            path_lengths[sim_off] = Scalar[dtype](depth + 1)
            var parent_gs_off = (
                e * MAX_NODES * STATE_SIZE + node_idx * STATE_SIZE
            )
            var exp_off = sim_off * STATE_SIZE
            for i in range(STATE_SIZE):
                expansion_states[exp_off + i] = game_states[parent_gs_off + i]


def gpu_mcts_batched_expand_backup_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    BATCH_SIMS: Int,
    PRED_OUT: Int,
    STATE_SIZE: Int,
    dtype: DType where dtype.is_floating_point(),
](
    # Node storage
    visit_count: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    prior: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    reward_buf: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    child_idx: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_visits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ],
    node_count: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    min_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    max_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Game states
    game_states: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * STATE_SIZE), MutAnyOrigin
    ],
    expansion_states: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS * STATE_SIZE), MutAnyOrigin
    ],
    # Pending expansions [N_ENVS * BATCH_SIMS]
    pending_parents: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS), MutAnyOrigin
    ],
    pending_actions: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS), MutAnyOrigin
    ],
    # Prediction output [N_ENVS * BATCH_SIMS * PRED_OUT]
    pred_output: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS * PRED_OUT), MutAnyOrigin
    ],
    # Rewards from env.step [N_ENVS * BATCH_SIMS]
    step_rewards: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS), MutAnyOrigin
    ],
    # Search paths [N_ENVS * BATCH_SIMS * MAX_DEPTH]
    search_paths: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS * MAX_DEPTH), MutAnyOrigin
    ],
    action_paths: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS * MAX_DEPTH), MutAnyOrigin
    ],
    path_lengths: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS), MutAnyOrigin
    ],
):
    """Fused: expand BATCH_SIMS nodes + negated backup + remove virtual losses.

    One thread per env. Processes all BATCH_SIMS expansions sequentially.
    Combines: store_child_state + expand_alphazero + backup_negated + virtual loss removal.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var tv_off = e * MAX_NODES

    comptime VIRTUAL_LOSS: Int = 3

    for s in range(BATCH_SIMS):
        var sim_off = e * BATCH_SIMS + s
        var parent = Int(rebind[Scalar[dtype]](pending_parents[sim_off]))
        var action = Int(rebind[Scalar[dtype]](pending_actions[sim_off]))
        var child_node_idx = Int(rebind[Scalar[dtype]](node_count[e]))

        if child_node_idx >= MAX_NODES:
            # Remove virtual loss even if we can't expand
            visit_count[tree_off + parent * ACT + action] = rebind[
                Scalar[dtype]
            ](visit_count[tree_off + parent * ACT + action]) - Scalar[dtype](
                VIRTUAL_LOSS
            )
            total_visits[tv_off + parent] = rebind[Scalar[dtype]](
                total_visits[tv_off + parent]
            ) - Scalar[dtype](VIRTUAL_LOSS)
            continue

        # ── Store child game state ──────────────────────────
        var exp_gs_off = sim_off * STATE_SIZE
        var child_gs_off = (
            e * MAX_NODES * STATE_SIZE + child_node_idx * STATE_SIZE
        )
        for i in range(STATE_SIZE):
            game_states[child_gs_off + i] = expansion_states[exp_gs_off + i]

        # ── Set parent reward from env.step ─────────────────
        reward_buf[tree_off + parent * ACT + action] = step_rewards[sim_off]

        # ── Set child prior from softmax ────────────────────
        var pred_off = sim_off * PRED_OUT
        var child_off = tree_off + child_node_idx * ACT
        var p_max = rebind[Scalar[dtype]](pred_output[pred_off])
        for a in range(1, ACT):
            var v = rebind[Scalar[dtype]](pred_output[pred_off + a])
            if v > p_max:
                p_max = v
        var p_sum = Scalar[dtype](0.0)
        for a in range(ACT):
            var v = exp(
                rebind[Scalar[dtype]](pred_output[pred_off + a]) - p_max
            )
            prior[child_off + a] = v
            p_sum += v
        for a in range(ACT):
            prior[child_off + a] = (
                rebind[Scalar[dtype]](prior[child_off + a]) / p_sum
            )
            visit_count[child_off + a] = Scalar[dtype](0.0)
            total_value[child_off + a] = Scalar[dtype](0.0)
            reward_buf[child_off + a] = Scalar[dtype](0.0)
            child_idx[child_off + a] = Scalar[dtype](-1.0)
        total_visits[tv_off + child_node_idx] = Scalar[dtype](0.0)

        # ── Link parent → child ─────────────────────────────
        child_idx[tree_off + parent * ACT + action] = Scalar[dtype](
            child_node_idx
        )
        node_count[e] = Scalar[dtype](child_node_idx + 1)

        # ── Decode leaf value ───────────────────────────────
        # If terminal (|reward| > 0.5), use game outcome directly
        var step_rew = rebind[Scalar[dtype]](step_rewards[sim_off])
        var abs_rew = step_rew if step_rew >= Scalar[dtype](0.0) else -step_rew
        var leaf_value: Scalar[dtype]
        if abs_rew > Scalar[dtype](0.5):
            leaf_value = step_rew  # Terminal: use actual game outcome
        else:
            var raw_v = rebind[Scalar[dtype]](pred_output[pred_off + ACT])
            var ev_p = exp(raw_v)
            var ev_n = exp(-raw_v)
            leaf_value = (ev_p - ev_n) / (
                ev_p + ev_n
            )  # Non-terminal: network estimate

        # ── Remove virtual loss ─────────────────────────────
        visit_count[tree_off + parent * ACT + action] = rebind[Scalar[dtype]](
            visit_count[tree_off + parent * ACT + action]
        ) - Scalar[dtype](VIRTUAL_LOSS)
        total_visits[tv_off + parent] = rebind[Scalar[dtype]](
            total_visits[tv_off + parent]
        ) - Scalar[dtype](VIRTUAL_LOSS)

        # ── Negated backup ──────────────────────────────────
        var value = leaf_value
        var path_off = sim_off * MAX_DEPTH
        var path_len = Int(rebind[Scalar[dtype]](path_lengths[sim_off]))

        for i in range(path_len):
            var idx = path_len - 1 - i
            var node = Int(rebind[Scalar[dtype]](search_paths[path_off + idx]))
            var act = Int(rebind[Scalar[dtype]](action_paths[path_off + idx]))

            value = -value  # Negate for zero-sum

            var na_off = tree_off + node * ACT + act
            visit_count[na_off] = rebind[Scalar[dtype]](
                visit_count[na_off]
            ) + Scalar[dtype](1.0)
            total_value[na_off] = (
                rebind[Scalar[dtype]](total_value[na_off]) + value
            )
            total_visits[tv_off + node] = rebind[Scalar[dtype]](
                total_visits[tv_off + node]
            ) + Scalar[dtype](1.0)

            var n_a = rebind[Scalar[dtype]](visit_count[na_off])
            var mean_q = rebind[Scalar[dtype]](total_value[na_off]) / n_a
            if mean_q < rebind[Scalar[dtype]](min_q[e]):
                min_q[e] = mean_q
            if mean_q > rebind[Scalar[dtype]](max_q[e]):
                max_q[e] = mean_q


# ═══════════════════════════════════════════════════════════════════════════
# MuZero Batched Fused Kernels (hidden states + dynamics network)
# ═══════════════════════════════════════════════════════════════════════════


def gpu_mcts_batched_select_and_build_dyn_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    BATCH_SIMS: Int,
    LATENT: Int,
    DYN_IN: Int,
    dtype: DType where dtype.is_floating_point(),
](
    # Node storage
    visit_count: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    prior: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    child_idx: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_visits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ],
    node_count: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    min_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    max_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Hidden states
    hidden_states: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * LATENT), MutAnyOrigin
    ],
    # Output: batched dynamics inputs [N_ENVS * BATCH_SIMS * DYN_IN]
    dyn_inputs: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS * DYN_IN), MutAnyOrigin
    ],
    # Output: pending info + paths [N_ENVS * BATCH_SIMS * ...]
    pending_parents: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS), MutAnyOrigin
    ],
    pending_actions: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS), MutAnyOrigin
    ],
    search_paths: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS * MAX_DEPTH), MutAnyOrigin
    ],
    action_paths: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS * MAX_DEPTH), MutAnyOrigin
    ],
    path_lengths: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS), MutAnyOrigin
    ],
    # PUCT
    c_base: Scalar[dtype],
    c_init: Scalar[dtype],
):
    """Fused: select BATCH_SIMS leaves per env + build dynamics input.

    MuZero version: copies parent hidden state + one-hot action into dyn_input.
    Applies virtual losses for diversity.
    One thread per env.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var tv_off = e * MAX_NODES
    var mn = rebind[Scalar[dtype]](min_q[e])
    var mx = rebind[Scalar[dtype]](max_q[e])
    var q_range = mx - mn

    comptime VIRTUAL_LOSS: Int = 3

    for s in range(BATCH_SIMS):
        var node_idx = 0
        var depth = 0
        var path_off = (e * BATCH_SIMS + s) * MAX_DEPTH
        search_paths[path_off] = Scalar[dtype](0.0)

        while depth < MAX_DEPTH - 1:
            var n_total = rebind[Scalar[dtype]](total_visits[tv_off + node_idx])
            var sqrt_total = sqrt(n_total + Scalar[dtype](1e-8))
            var c = (
                log((Scalar[dtype](1.0) + n_total + c_base) / c_base) + c_init
            )

            var best_action = 0
            var best_score = Scalar[dtype](-1e18)
            for a in range(ACT):
                var na_off = tree_off + node_idx * ACT + a
                var n_a = rebind[Scalar[dtype]](visit_count[na_off])
                var q_val: Scalar[dtype]
                if n_a > Scalar[dtype](0.5):
                    q_val = rebind[Scalar[dtype]](total_value[na_off]) / n_a
                    if q_range > Scalar[dtype](1e-8):
                        q_val = (q_val - mn) / q_range
                else:
                    q_val = Scalar[dtype](0.0)
                var p = rebind[Scalar[dtype]](prior[na_off])
                var score = q_val + c * p * sqrt_total / (
                    Scalar[dtype](1.0) + n_a
                )
                if score > best_score:
                    best_score = score
                    best_action = a

            action_paths[path_off + depth] = Scalar[dtype](best_action)

            var child = rebind[Scalar[dtype]](
                child_idx[tree_off + node_idx * ACT + best_action]
            )
            if child < Scalar[dtype](0.0):
                var sim_off = e * BATCH_SIMS + s
                pending_parents[sim_off] = Scalar[dtype](node_idx)
                pending_actions[sim_off] = Scalar[dtype](best_action)
                path_lengths[sim_off] = Scalar[dtype](depth + 1)

                # Virtual loss
                visit_count[tree_off + node_idx * ACT + best_action] = rebind[
                    Scalar[dtype]
                ](
                    visit_count[tree_off + node_idx * ACT + best_action]
                ) + Scalar[
                    dtype
                ](
                    VIRTUAL_LOSS
                )
                total_visits[tv_off + node_idx] = rebind[Scalar[dtype]](
                    total_visits[tv_off + node_idx]
                ) + Scalar[dtype](VIRTUAL_LOSS)

                # Build dynamics input: [hidden || one_hot_action]
                var dyn_off = sim_off * DYN_IN
                var h_off = e * MAX_NODES * LATENT + node_idx * LATENT
                for i in range(LATENT):
                    dyn_inputs[dyn_off + i] = hidden_states[h_off + i]
                for a in range(ACT):
                    dyn_inputs[dyn_off + LATENT + a] = Scalar[dtype](0.0)
                dyn_inputs[dyn_off + LATENT + best_action] = Scalar[dtype](1.0)
                break

            node_idx = Int(child)
            depth += 1
            search_paths[path_off + depth] = Scalar[dtype](node_idx)

        if depth >= MAX_DEPTH - 1:
            var sim_off = e * BATCH_SIMS + s
            pending_parents[sim_off] = Scalar[dtype](node_idx)
            pending_actions[sim_off] = Scalar[dtype](0)
            path_lengths[sim_off] = Scalar[dtype](depth + 1)
            var dyn_off = sim_off * DYN_IN
            var h_off = e * MAX_NODES * LATENT + node_idx * LATENT
            for i in range(LATENT):
                dyn_inputs[dyn_off + i] = hidden_states[h_off + i]
            for a in range(ACT):
                dyn_inputs[dyn_off + LATENT + a] = Scalar[dtype](0.0)


def gpu_mcts_batched_expand_backup_muzero_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    BATCH_SIMS: Int,
    LATENT: Int,
    PRED_OUT: Int,
    DYN_OUT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    # Node storage
    visit_count: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    prior: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    reward_buf: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    child_idx: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ],
    total_visits: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ],
    node_count: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    min_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    max_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Hidden states
    hidden_states: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * LATENT), MutAnyOrigin
    ],
    # Pending [N_ENVS * BATCH_SIMS]
    pending_parents: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS), MutAnyOrigin
    ],
    pending_actions: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS), MutAnyOrigin
    ],
    # Dynamics output [N_ENVS * BATCH_SIMS * DYN_OUT]
    dyn_output: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS * DYN_OUT), MutAnyOrigin
    ],
    # Prediction output [N_ENVS * BATCH_SIMS * PRED_OUT]
    pred_output: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS * PRED_OUT), MutAnyOrigin
    ],
    # Search paths [N_ENVS * BATCH_SIMS * MAX_DEPTH]
    search_paths: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS * MAX_DEPTH), MutAnyOrigin
    ],
    action_paths: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS * MAX_DEPTH), MutAnyOrigin
    ],
    path_lengths: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * BATCH_SIMS), MutAnyOrigin
    ],
    # Value support
    v_min: Scalar[dtype],
    v_max: Scalar[dtype],
    gamma: Scalar[dtype],
    # Whether to negate backup (two-player) — passed as Scalar[DType.bool]
    negate: Scalar[DType.bool],
):
    """Fused: extract hidden + expand + backup for BATCH_SIMS MuZero leaves.

    One thread per env. For each of BATCH_SIMS expansions:
    1. Extract hidden state from dynamics output → hidden_states pool
    2. Scale hidden state (MinMax)
    3. Decode reward from dynamics output
    4. Set child prior from prediction softmax
    5. Decode leaf value (scalar or categorical)
    6. Remove virtual losses
    7. Backup (standard or negated)
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var tree_off = e * MAX_NODES * ACT
    var tv_off = e * MAX_NODES

    comptime VIRTUAL_LOSS: Int = 3
    comptime NUM_VAL_BINS = PRED_OUT - ACT
    comptime NUM_REW_BINS = DYN_OUT - LATENT

    for s in range(BATCH_SIMS):
        var sim_off = e * BATCH_SIMS + s
        var parent = Int(rebind[Scalar[dtype]](pending_parents[sim_off]))
        var action = Int(rebind[Scalar[dtype]](pending_actions[sim_off]))
        var child_node_idx = Int(rebind[Scalar[dtype]](node_count[e]))

        # Remove virtual loss
        visit_count[tree_off + parent * ACT + action] = rebind[Scalar[dtype]](
            visit_count[tree_off + parent * ACT + action]
        ) - Scalar[dtype](VIRTUAL_LOSS)
        total_visits[tv_off + parent] = rebind[Scalar[dtype]](
            total_visits[tv_off + parent]
        ) - Scalar[dtype](VIRTUAL_LOSS)

        if child_node_idx >= MAX_NODES:
            continue

        # 1. Extract hidden state from dyn_output
        var dyn_off = sim_off * DYN_OUT
        var child_h_off = e * MAX_NODES * LATENT + child_node_idx * LATENT
        for i in range(LATENT):
            hidden_states[child_h_off + i] = dyn_output[dyn_off + i]

        # 2. Scale hidden state (MinMax)
        var h_min = rebind[Scalar[dtype]](hidden_states[child_h_off])
        var h_max = h_min
        for i in range(1, LATENT):
            var v = rebind[Scalar[dtype]](hidden_states[child_h_off + i])
            if v < h_min:
                h_min = v
            if v > h_max:
                h_max = v
        var h_delta = h_max - h_min
        if h_delta > Scalar[dtype](1e-8):
            for i in range(LATENT):
                var v = rebind[Scalar[dtype]](hidden_states[child_h_off + i])
                hidden_states[child_h_off + i] = (v - h_min) / h_delta

        # 3. Decode reward
        var rew_decoded: Scalar[dtype]
        if NUM_REW_BINS == 1:
            var raw = rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT])
            var e_pos = exp(raw)
            var e_neg = exp(-raw)
            rew_decoded = (e_pos - e_neg) / (e_pos + e_neg)
        else:
            var rew_max = rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT])
            for i in range(1, NUM_REW_BINS):
                var v = rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT + i])
                if v > rew_max:
                    rew_max = v
            var rew_se = Scalar[dtype](0.0)
            for i in range(NUM_REW_BINS):
                rew_se += exp(
                    rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT + i])
                    - rew_max
                )
            rew_decoded = Scalar[dtype](0.0)
            var rew_step = (v_max - v_min) / Scalar[dtype](NUM_REW_BINS - 1)
            for i in range(NUM_REW_BINS):
                var prob = (
                    exp(
                        rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT + i])
                        - rew_max
                    )
                    / rew_se
                )
                rew_decoded += prob * (v_min + Scalar[dtype](i) * rew_step)
        reward_buf[tree_off + parent * ACT + action] = rew_decoded

        # 4. Set child prior from prediction softmax
        var pred_off = sim_off * PRED_OUT
        var child_off = tree_off + child_node_idx * ACT
        var p_max = rebind[Scalar[dtype]](pred_output[pred_off])
        for a in range(1, ACT):
            var v = rebind[Scalar[dtype]](pred_output[pred_off + a])
            if v > p_max:
                p_max = v
        var p_sum = Scalar[dtype](0.0)
        for a in range(ACT):
            var v = exp(
                rebind[Scalar[dtype]](pred_output[pred_off + a]) - p_max
            )
            prior[child_off + a] = v
            p_sum += v
        for a in range(ACT):
            prior[child_off + a] = (
                rebind[Scalar[dtype]](prior[child_off + a]) / p_sum
            )
            visit_count[child_off + a] = Scalar[dtype](0.0)
            total_value[child_off + a] = Scalar[dtype](0.0)
            reward_buf[child_off + a] = Scalar[dtype](0.0)
            child_idx[child_off + a] = Scalar[dtype](-1.0)
        total_visits[tv_off + child_node_idx] = Scalar[dtype](0.0)

        # Link
        child_idx[tree_off + parent * ACT + action] = Scalar[dtype](
            child_node_idx
        )
        node_count[e] = Scalar[dtype](child_node_idx + 1)

        # 5. Decode leaf value
        var leaf_value: Scalar[dtype]
        if NUM_VAL_BINS == 1:
            var raw_v = rebind[Scalar[dtype]](pred_output[pred_off + ACT])
            var ev_p = exp(raw_v)
            var ev_n = exp(-raw_v)
            leaf_value = (ev_p - ev_n) / (ev_p + ev_n)
        else:
            var val_max = rebind[Scalar[dtype]](pred_output[pred_off + ACT])
            for i in range(1, NUM_VAL_BINS):
                var v = rebind[Scalar[dtype]](pred_output[pred_off + ACT + i])
                if v > val_max:
                    val_max = v
            var val_se = Scalar[dtype](0.0)
            for i in range(NUM_VAL_BINS):
                val_se += exp(
                    rebind[Scalar[dtype]](pred_output[pred_off + ACT + i])
                    - val_max
                )
            var val_step = (v_max - v_min) / Scalar[dtype](NUM_VAL_BINS - 1)
            leaf_value = Scalar[dtype](0.0)
            for i in range(NUM_VAL_BINS):
                var prob = (
                    exp(
                        rebind[Scalar[dtype]](pred_output[pred_off + ACT + i])
                        - val_max
                    )
                    / val_se
                )
                leaf_value += prob * (v_min + Scalar[dtype](i) * val_step)

        # 6. Backup (negated or standard)
        var value = leaf_value
        var path_off_s = sim_off * MAX_DEPTH
        var path_len = Int(rebind[Scalar[dtype]](path_lengths[sim_off]))

        for i in range(path_len):
            var idx = path_len - 1 - i
            var node = Int(
                rebind[Scalar[dtype]](search_paths[path_off_s + idx])
            )
            var act = Int(rebind[Scalar[dtype]](action_paths[path_off_s + idx]))

            if Bool(negate):
                value = -value
            else:
                value = (
                    rebind[Scalar[dtype]](
                        reward_buf[tree_off + node * ACT + act]
                    )
                    + gamma * value
                )

            var na_off = tree_off + node * ACT + act
            visit_count[na_off] = rebind[Scalar[dtype]](
                visit_count[na_off]
            ) + Scalar[dtype](1.0)
            total_value[na_off] = (
                rebind[Scalar[dtype]](total_value[na_off]) + value
            )
            total_visits[tv_off + node] = rebind[Scalar[dtype]](
                total_visits[tv_off + node]
            ) + Scalar[dtype](1.0)

            var n_a = rebind[Scalar[dtype]](visit_count[na_off])
            var mean_q = rebind[Scalar[dtype]](total_value[na_off]) / n_a
            if mean_q < rebind[Scalar[dtype]](min_q[e]):
                min_q[e] = mean_q
            if mean_q > rebind[Scalar[dtype]](max_q[e]):
                max_q[e] = mean_q


def gpu_mcts_build_dyn_input_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    LATENT: Int,
    DYN_IN: Int,
    dtype: DType where dtype.is_floating_point(),
](
    dyn_input: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * DYN_IN), MutAnyOrigin
    ],
    hidden_states: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * LATENT), MutAnyOrigin
    ],
    pending_parent: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    pending_action: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
):
    """Build dynamics input [hidden || one_hot_action] for each env's pending expansion.

    One thread per environment.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    var parent = Int(rebind[Scalar[dtype]](pending_parent[e]))
    var action = Int(rebind[Scalar[dtype]](pending_action[e]))
    var h_off = e * MAX_NODES * LATENT + parent * LATENT
    var d_off = e * DYN_IN

    # Copy hidden state
    for i in range(LATENT):
        dyn_input[d_off + i] = hidden_states[h_off + i]

    # One-hot action
    for a in range(ACT):
        dyn_input[d_off + LATENT + a] = Scalar[dtype](0.0)
    dyn_input[d_off + LATENT + action] = Scalar[dtype](1.0)


def gpu_mcts_copy_pred_input_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    LATENT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    pred_input: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * LATENT), MutAnyOrigin
    ],
    hidden_states: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * LATENT), MutAnyOrigin
    ],
    node_count: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    dyn_output: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * (LATENT + LATENT)), MutAnyOrigin
    ],
):
    """Copy newly expanded child hidden states to prediction input buffer.

    The child's hidden state was written to hidden_states by the expand kernel.
    We need to copy it to pred_input for the prediction forward call.
    One thread per environment.
    """
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return

    # The child index is node_count - 1 (expand kernel already incremented)
    var child_idx_val = Int(rebind[Scalar[dtype]](node_count[e])) - 1
    if child_idx_val < 0:
        return

    var h_off = e * MAX_NODES * LATENT + child_idx_val * LATENT
    var p_off = e * LATENT

    for i in range(LATENT):
        pred_input[p_off + i] = hidden_states[h_off + i]


# ═══════════════════════════════════════════════════════════════════════════
# GPU MCTS State
# ═══════════════════════════════════════════════════════════════════════════


struct GPUMCTSState[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    STATE_SIZE: Int = 0,
    BATCH_SIMS: Int = 8,
](Movable):
    """GPU-resident MCTS tree storage for n_envs parallel searches.

    All tree data lives on GPU. No CPU↔GPU transfer during search.

    When STATE_SIZE > 0 (AlphaZero mode), stores actual game states
    per tree node for expansion via true game rules.
    """

    comptime PRED_OUT: Int = Self.ACT + Self.BINS
    comptime DYN_IN: Int = Self.LATENT + Self.ACT
    comptime DYN_OUT: Int = Self.LATENT + Self.BINS

    # Node data [N_ENVS × MAX_NODES × ACT]
    var visit_count: DeviceBuffer[dtype]
    var total_value: DeviceBuffer[dtype]
    var prior: DeviceBuffer[dtype]
    var reward: DeviceBuffer[dtype]
    var child_idx: DeviceBuffer[dtype]

    # Per-node scalar [N_ENVS × MAX_NODES]
    var total_visits: DeviceBuffer[dtype]

    # Hidden state pool [N_ENVS × MAX_NODES × LATENT] (MuZero mode)
    var hidden_states: DeviceBuffer[dtype]

    # Game state pool [N_ENVS × MAX_NODES × STATE_SIZE] (AlphaZero mode)
    # Stores actual game states for expansion via env.step()
    var game_states: DeviceBuffer[dtype]

    # Staging buffer for env.step expansion [N_ENVS × STATE_SIZE]
    var expansion_states: DeviceBuffer[dtype]
    # Obs output from env.step [N_ENVS × OBS_DIM] — reuses pred_input
    # Legal masks from env.step [N_ENVS × ACT]
    var expansion_legal_masks: DeviceBuffer[dtype]

    # Per-env scalars [N_ENVS]
    var node_count: DeviceBuffer[dtype]
    var min_q: DeviceBuffer[dtype]
    var max_q: DeviceBuffer[dtype]

    # Selection output
    var pending_parent: DeviceBuffer[dtype]
    var pending_action: DeviceBuffer[dtype]
    var search_paths: DeviceBuffer[dtype]  # [N_ENVS × MAX_DEPTH]
    var action_paths: DeviceBuffer[dtype]  # [N_ENVS × MAX_DEPTH]
    var path_lengths: DeviceBuffer[dtype]
    var leaf_values: DeviceBuffer[dtype]

    # Batched network I/O
    var dyn_input: DeviceBuffer[dtype]  # [N_ENVS × DYN_IN]
    var dyn_output: DeviceBuffer[dtype]  # [N_ENVS × DYN_OUT]
    var pred_input: DeviceBuffer[dtype]  # [N_ENVS × LATENT]
    var pred_output: DeviceBuffer[dtype]  # [N_ENVS × PRED_OUT]

    # Final output
    var actions_out: DeviceBuffer[dtype]  # [N_ENVS]
    var policies_out: DeviceBuffer[dtype]  # [N_ENVS × ACT]

    # Legal mask (for self-play / board games)
    var legal_masks: DeviceBuffer[dtype]  # [N_ENVS × ACT]

    def __init__(out self, ctx: DeviceContext) raises:
        comptime NODE_ACT_SIZE = Self.N_ENVS * Self.MAX_NODES * Self.ACT
        comptime NODE_SIZE = Self.N_ENVS * Self.MAX_NODES
        comptime HIDDEN_SIZE = Self.N_ENVS * Self.MAX_NODES * Self.LATENT

        self.visit_count = ctx.enqueue_create_buffer[dtype](NODE_ACT_SIZE)
        self.total_value = ctx.enqueue_create_buffer[dtype](NODE_ACT_SIZE)
        self.prior = ctx.enqueue_create_buffer[dtype](NODE_ACT_SIZE)
        self.reward = ctx.enqueue_create_buffer[dtype](NODE_ACT_SIZE)
        self.child_idx = ctx.enqueue_create_buffer[dtype](NODE_ACT_SIZE)
        self.total_visits = ctx.enqueue_create_buffer[dtype](NODE_SIZE)
        self.hidden_states = ctx.enqueue_create_buffer[dtype](HIDDEN_SIZE)
        self.node_count = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.min_q = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.max_q = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)

        # Batched pending/path buffers [N_ENVS * BATCH_SIMS * ...]
        comptime BS = Self.BATCH_SIMS
        self.pending_parent = ctx.enqueue_create_buffer[dtype](Self.N_ENVS * BS)
        self.pending_action = ctx.enqueue_create_buffer[dtype](Self.N_ENVS * BS)
        self.search_paths = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * BS * MAX_DEPTH
        )
        self.action_paths = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * BS * MAX_DEPTH
        )
        self.path_lengths = ctx.enqueue_create_buffer[dtype](Self.N_ENVS * BS)
        self.leaf_values = ctx.enqueue_create_buffer[dtype](Self.N_ENVS * BS)

        self.dyn_input = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * BS * Self.DYN_IN
        )
        self.dyn_output = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * BS * Self.DYN_OUT
        )
        self.pred_input = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * BS * Self.LATENT
        )
        self.pred_output = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * BS * Self.PRED_OUT
        )

        self.actions_out = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.policies_out = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.ACT
        )
        self.legal_masks = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.ACT
        )

        # Game state storage (AlphaZero mode: STATE_SIZE > 0)
        comptime GS_SIZE = Self.N_ENVS * Self.MAX_NODES * Self.STATE_SIZE
        self.game_states = ctx.enqueue_create_buffer[dtype](
            GS_SIZE if GS_SIZE > 0 else 1
        )
        comptime EXP_SIZE = Self.N_ENVS * BS * Self.STATE_SIZE
        self.expansion_states = ctx.enqueue_create_buffer[dtype](
            EXP_SIZE if EXP_SIZE > 0 else 1
        )
        self.expansion_legal_masks = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * BS * Self.ACT
        )

    def __init__(out self, *, deinit take: Self):
        self.visit_count = take.visit_count^
        self.total_value = take.total_value^
        self.prior = take.prior^
        self.reward = take.reward^
        self.child_idx = take.child_idx^
        self.total_visits = take.total_visits^
        self.hidden_states = take.hidden_states^
        self.game_states = take.game_states^
        self.expansion_states = take.expansion_states^
        self.expansion_legal_masks = take.expansion_legal_masks^
        self.node_count = take.node_count^
        self.min_q = take.min_q^
        self.max_q = take.max_q^
        self.pending_parent = take.pending_parent^
        self.pending_action = take.pending_action^
        self.search_paths = take.search_paths^
        self.action_paths = take.action_paths^
        self.path_lengths = take.path_lengths^
        self.leaf_values = take.leaf_values^
        self.dyn_input = take.dyn_input^
        self.dyn_output = take.dyn_output^
        self.pred_input = take.pred_input^
        self.pred_output = take.pred_output^
        self.actions_out = take.actions_out^
        self.policies_out = take.policies_out^
        self.legal_masks = take.legal_masks^
