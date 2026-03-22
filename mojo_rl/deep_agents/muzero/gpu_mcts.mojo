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


fn gpu_mcts_init_root_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    LATENT: Int,
    PRED_OUT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    # Node storage
    visit_count: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin],
    total_value: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin],
    prior: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin],
    reward: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin],
    child_idx: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin],
    total_visits: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin],
    node_count: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Prediction output (from representation + prediction forward on root obs)
    pred_output: LayoutTensor[dtype, Layout.row_major(N_ENVS * PRED_OUT), MutAnyOrigin],
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
        var v = exp(rebind[Scalar[dtype]](pred_output[pred_off + a]) - max_logit)
        prior[tree_off + a] = v
        sum_exp += v
    for a in range(ACT):
        prior[tree_off + a] = rebind[Scalar[dtype]](prior[tree_off + a]) / sum_exp

    # Add Dirichlet noise (approximate with Exponential(1) / sum)
    var philox = PhiloxRandom(
        seed=UInt64(rng_seed) + UInt64(e * 137 + 1), offset=0
    )
    var noise_sum = Scalar[dtype](0.0)
    for a in range(ACT):
        var rand_vals = philox.step_uniform()
        var noise_val = -log(Scalar[dtype](rand_vals[0]) + Scalar[dtype](1e-8))
        reward[tree_off + a] = noise_val  # Temporarily store noise in reward[root]
        noise_sum += noise_val
    for a in range(ACT):
        var noise_val = rebind[Scalar[dtype]](reward[tree_off + a]) / noise_sum
        prior[tree_off + a] = (
            (Scalar[dtype](1.0) - noise_fraction) * rebind[Scalar[dtype]](prior[tree_off + a])
            + noise_fraction * noise_val
        )
        reward[tree_off + a] = Scalar[dtype](0.0)  # Reset reward

    # Initialize tree state
    node_count[e] = Scalar[dtype](1.0)  # Root is node 0
    min_q[e] = Scalar[dtype](1e18)
    max_q[e] = Scalar[dtype](-1e18)


fn gpu_mcts_select_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    # Node storage (read)
    visit_count: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin],
    total_value: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin],
    prior: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin],
    child_idx: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin],
    total_visits: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin],
    node_count: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # MinMax Q normalization
    min_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    max_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Output: pending expansion info
    pending_parent: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    pending_action: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Output: search paths for backup
    search_paths: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin],
    action_paths: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin],
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
        var c = log(
            (Scalar[dtype](1.0) + n_total + c_base) / c_base
        ) + c_init

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


fn gpu_mcts_expand_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    LATENT: Int,
    PRED_OUT: Int,
    DYN_OUT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    # Node storage (write)
    visit_count: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin],
    total_value: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin],
    prior: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin],
    reward_buf: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin],
    child_idx: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin],
    total_visits: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin],
    node_count: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Hidden state pool
    hidden_states: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * LATENT), MutAnyOrigin],
    # Pending expansion info
    pending_parent: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    pending_action: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Network outputs
    dyn_output: LayoutTensor[dtype, Layout.row_major(N_ENVS * DYN_OUT), MutAnyOrigin],
    pred_output: LayoutTensor[dtype, Layout.row_major(N_ENVS * PRED_OUT), MutAnyOrigin],
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

    # 3. Extract reward (decode categorical)
    var step = (v_max - v_min) / Scalar[dtype](PRED_OUT - ACT - 1)
    var rew_max_val = rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT])
    for i in range(1, DYN_OUT - LATENT):
        var v = rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT + i])
        if v > rew_max_val:
            rew_max_val = v
    var rew_sum_exp = Scalar[dtype](0.0)
    for i in range(DYN_OUT - LATENT):
        rew_sum_exp += exp(rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT + i]) - rew_max_val)
    var rew_decoded = Scalar[dtype](0.0)
    var num_rew_bins = DYN_OUT - LATENT
    var rew_step = (v_max - v_min) / Scalar[dtype](num_rew_bins - 1)
    for i in range(num_rew_bins):
        var prob = exp(rebind[Scalar[dtype]](dyn_output[dyn_off + LATENT + i]) - rew_max_val) / rew_sum_exp
        rew_decoded += prob * (v_min + Scalar[dtype](i) * rew_step)

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
        prior[child_tree_off + a] = rebind[Scalar[dtype]](prior[child_tree_off + a]) / p_sum
        visit_count[child_tree_off + a] = Scalar[dtype](0.0)
        total_value[child_tree_off + a] = Scalar[dtype](0.0)
        reward_buf[child_tree_off + a] = Scalar[dtype](0.0)
        child_idx[child_tree_off + a] = Scalar[dtype](-1.0)
    total_visits[tv_off + child_node_idx] = Scalar[dtype](0.0)

    # 5. Link parent → child
    child_idx[tree_off + parent * ACT + action] = Scalar[dtype](child_node_idx)
    node_count[e] = Scalar[dtype](child_node_idx + 1)

    # 6. Decode leaf value (categorical → scalar → inverse transform)
    var val_max = rebind[Scalar[dtype]](pred_output[pred_off + ACT])
    var num_val_bins = PRED_OUT - ACT
    for i in range(1, num_val_bins):
        var v = rebind[Scalar[dtype]](pred_output[pred_off + ACT + i])
        if v > val_max:
            val_max = v
    var val_sum_exp = Scalar[dtype](0.0)
    for i in range(num_val_bins):
        val_sum_exp += exp(rebind[Scalar[dtype]](pred_output[pred_off + ACT + i]) - val_max)
    var val_step = (v_max - v_min) / Scalar[dtype](num_val_bins - 1)
    var val_decoded = Scalar[dtype](0.0)
    for i in range(num_val_bins):
        var prob = exp(rebind[Scalar[dtype]](pred_output[pred_off + ACT + i]) - val_max) / val_sum_exp
        val_decoded += prob * (v_min + Scalar[dtype](i) * val_step)

    # Inverse scalar transform: h^{-1}(y)
    var sign = Scalar[dtype](1.0) if val_decoded >= Scalar[dtype](0.0) else Scalar[dtype](-1.0)
    var abs_y = val_decoded if val_decoded >= Scalar[dtype](0.0) else -val_decoded
    var eps = Scalar[dtype](0.001)
    var inner = sqrt(Scalar[dtype](1.0) + Scalar[dtype](4.0) * eps * (abs_y + Scalar[dtype](1.0) + eps))
    var f = (inner - Scalar[dtype](1.0)) / (Scalar[dtype](2.0) * eps)
    leaf_values[e] = sign * (f * f - Scalar[dtype](1.0))


fn gpu_mcts_backup_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    # Node storage (read/write)
    visit_count: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin],
    total_value: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin],
    reward_buf: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin],
    total_visits: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin],
    min_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    max_q: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    # Search paths
    search_paths: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin],
    action_paths: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin],
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

        visit_count[na_off] = rebind[Scalar[dtype]](visit_count[na_off]) + Scalar[dtype](1.0)
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


fn gpu_mcts_extract_actions_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    visit_count: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin],
    actions_out: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    policies_out: LayoutTensor[dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin],
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
            policies_out[e * ACT + a] = rebind[Scalar[dtype]](
                visit_count[root_off + a]
            ) / total
    else:
        for a in range(ACT):
            policies_out[e * ACT + a] = Scalar[dtype](1.0) / Scalar[dtype](ACT)


fn gpu_mcts_build_dyn_input_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    LATENT: Int,
    DYN_IN: Int,
    dtype: DType where dtype.is_floating_point(),
](
    dyn_input: LayoutTensor[dtype, Layout.row_major(N_ENVS * DYN_IN), MutAnyOrigin],
    hidden_states: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * LATENT), MutAnyOrigin],
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


fn gpu_mcts_copy_pred_input_kernel[
    N_ENVS: Int,
    MAX_NODES: Int,
    LATENT: Int,
    dtype: DType where dtype.is_floating_point(),
](
    pred_input: LayoutTensor[dtype, Layout.row_major(N_ENVS * LATENT), MutAnyOrigin],
    hidden_states: LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * LATENT), MutAnyOrigin],
    node_count: LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin],
    dyn_output: LayoutTensor[dtype, Layout.row_major(N_ENVS * (LATENT + LATENT)), MutAnyOrigin],
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
](Movable):
    """GPU-resident MCTS tree storage for n_envs parallel searches.

    All tree data lives on GPU. No CPU↔GPU transfer during search.
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

    # Hidden state pool [N_ENVS × MAX_NODES × LATENT]
    var hidden_states: DeviceBuffer[dtype]

    # Per-env scalars [N_ENVS]
    var node_count: DeviceBuffer[dtype]
    var min_q: DeviceBuffer[dtype]
    var max_q: DeviceBuffer[dtype]

    # Selection output
    var pending_parent: DeviceBuffer[dtype]
    var pending_action: DeviceBuffer[dtype]
    var search_paths: DeviceBuffer[dtype]   # [N_ENVS × MAX_DEPTH]
    var action_paths: DeviceBuffer[dtype]   # [N_ENVS × MAX_DEPTH]
    var path_lengths: DeviceBuffer[dtype]
    var leaf_values: DeviceBuffer[dtype]

    # Batched network I/O
    var dyn_input: DeviceBuffer[dtype]      # [N_ENVS × DYN_IN]
    var dyn_output: DeviceBuffer[dtype]     # [N_ENVS × DYN_OUT]
    var pred_input: DeviceBuffer[dtype]     # [N_ENVS × LATENT]
    var pred_output: DeviceBuffer[dtype]    # [N_ENVS × PRED_OUT]

    # Final output
    var actions_out: DeviceBuffer[dtype]    # [N_ENVS]
    var policies_out: DeviceBuffer[dtype]   # [N_ENVS × ACT]

    fn __init__(out self, ctx: DeviceContext) raises:
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

        self.pending_parent = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.pending_action = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.search_paths = ctx.enqueue_create_buffer[dtype](Self.N_ENVS * MAX_DEPTH)
        self.action_paths = ctx.enqueue_create_buffer[dtype](Self.N_ENVS * MAX_DEPTH)
        self.path_lengths = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.leaf_values = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)

        self.dyn_input = ctx.enqueue_create_buffer[dtype](Self.N_ENVS * Self.DYN_IN)
        self.dyn_output = ctx.enqueue_create_buffer[dtype](Self.N_ENVS * Self.DYN_OUT)
        self.pred_input = ctx.enqueue_create_buffer[dtype](Self.N_ENVS * Self.LATENT)
        self.pred_output = ctx.enqueue_create_buffer[dtype](Self.N_ENVS * Self.PRED_OUT)

        self.actions_out = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.policies_out = ctx.enqueue_create_buffer[dtype](Self.N_ENVS * Self.ACT)

    fn __init__(out self, *, deinit take: Self):
        self.visit_count = take.visit_count^
        self.total_value = take.total_value^
        self.prior = take.prior^
        self.reward = take.reward^
        self.child_idx = take.child_idx^
        self.total_visits = take.total_visits^
        self.hidden_states = take.hidden_states^
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
