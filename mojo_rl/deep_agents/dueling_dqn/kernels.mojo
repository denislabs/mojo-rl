"""GPU kernels for Dueling DQN operations.

Provides dueling-specific kernels for combining V+A into Q and transforming
gradients from Q-space back to dueling output space. Also re-exports TD target
kernels from DQN since those are identical once Q-values are computed.
"""

from layout import (
    Layout,
    LayoutTensor,
)
from std.gpu import (
    block_dim,
    block_idx,
    thread_idx,
)

# Re-export DQN TD target kernels (same computation once we have Q-values)
from mojo_rl.deep_agents.dqn.kernels import (
    dqn_td_target_kernel,
    dqn_double_td_target_kernel,
)


# =============================================================================
# Dueling-Specific GPU Kernels
# =============================================================================


@always_inline
fn dueling_combine_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    NUM_ACTIONS: Int,
    DUELING_OUT: Int,
](
    # Outputs
    q_values: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
    # Inputs
    dueling_output: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, DUELING_OUT), MutAnyOrigin
    ],
):
    """Combine dueling output [V, A1..An] into Q-values: Q(s,a) = V(s) + (A(s,a) - mean(A)).

    Each thread handles one batch sample.

    Parameters:
        dtype: Data type.
        BATCH_SIZE: Batch size.
        NUM_ACTIONS: Number of discrete actions.
        DUELING_OUT: Dueling model output dimension (1 + NUM_ACTIONS).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH_SIZE:
        return

    # V(s) is the first element
    var v_s = dueling_output[b, 0]

    # Compute mean advantage
    var mean_adv = Scalar[dtype](0.0)
    for a in range(NUM_ACTIONS):
        mean_adv += dueling_output[b, 1 + a]
    mean_adv /= Scalar[dtype](NUM_ACTIONS)

    # Q(s,a) = V(s) + (A(s,a) - mean(A))
    for a in range(NUM_ACTIONS):
        var adv = dueling_output[b, 1 + a]
        q_values[b, a] = v_s + (adv - mean_adv)


@always_inline
fn dueling_grad_kernel[
    dtype: DType,
    BATCH_SIZE: Int,
    NUM_ACTIONS: Int,
    DUELING_OUT: Int,
](
    # Outputs
    dueling_grad: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, DUELING_OUT), MutAnyOrigin
    ],
    # Inputs
    dq_grad: LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, NUM_ACTIONS), MutAnyOrigin
    ],
):
    """Transform dQ gradients to dueling output gradients.

    dV = sum(dQ_j)
    dA_i = dQ_i - (1/n) * sum(dQ_j)

    Each thread handles one batch sample.

    Parameters:
        dtype: Data type.
        BATCH_SIZE: Batch size.
        NUM_ACTIONS: Number of discrete actions.
        DUELING_OUT: Dueling model output dimension (1 + NUM_ACTIONS).
    """
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH_SIZE:
        return

    # Compute sum of dQ gradients
    var sum_dq = Scalar[dtype](0.0)
    for a in range(NUM_ACTIONS):
        sum_dq += dq_grad[b, a]

    # dV = sum(dQ)
    dueling_grad[b, 0] = sum_dq

    # dA_i = dQ_i - (1/n) * sum(dQ)
    var one_over_n = Scalar[dtype](1.0) / Scalar[dtype](NUM_ACTIONS)
    for a in range(NUM_ACTIONS):
        dueling_grad[b, 1 + a] = dq_grad[b, a] - one_over_n * sum_dq
