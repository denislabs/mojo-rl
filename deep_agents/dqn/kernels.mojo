from layout import (
    Layout,
    LayoutTensor,
)
from std.gpu import (
    block_dim,
    block_idx,
    thread_idx,
)

# =============================================================================
# GPU Kernels for DQN Operations
# =============================================================================


@always_inline
fn dqn_td_target_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    NUM_ACTIONS: Int,
](
    # Outputs
    targets: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    # Inputs
    next_q_values: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    rewards: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    gamma: Scalar[dtype],
):
    """Compute TD targets for standard DQN: target = r + gamma * max_a Q(s', a) * (1 - done).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH_SIZE:
        return

    var max_q = next_q_values[b, 0]
    for a in range(1, NUM_ACTIONS):
        var q = next_q_values[b, a]
        if q > max_q:
            max_q = q

    var done_mask = Scalar[dtype](1.0) - dones[b]
    targets[b] = rewards[b] + gamma * max_q * done_mask


@always_inline
fn dqn_double_td_target_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    NUM_ACTIONS: Int,
](
    # Outputs
    targets: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    # Inputs
    online_next_q: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    target_next_q: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    rewards: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
    gamma: Scalar[dtype],
):
    """Compute TD targets for Double DQN: target = r + gamma * Q_target(s', argmax_a Q_online(s', a)) * (1 - done).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH_SIZE:
        return

    var best_action = 0
    var best_q = online_next_q[b, 0]
    for a in range(1, NUM_ACTIONS):
        var q = online_next_q[b, a]
        if q > best_q:
            best_q = q
            best_action = a

    var target_q = target_next_q[b, best_action]
    var done_mask = Scalar[dtype](1.0) - dones[b]
    targets[b] = rewards[b] + gamma * target_q * done_mask
