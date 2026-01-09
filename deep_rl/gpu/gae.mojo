"""Generalized Advantage Estimation (GAE) kernel for GPU RL algorithms.

This kernel computes advantages and returns using GAE, which is used by
A2C, PPO, and other policy gradient methods.

Reference: "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
           (Schulman et al., 2016)
"""

from gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor

# =============================================================================
# Constants
# =============================================================================

comptime dtype = DType.float32


# =============================================================================
# GAE Kernel
# =============================================================================


fn compute_gae_kernel[
    NUM_ENVS: Int, ROLLOUT_LEN: Int
](
    gamma: Scalar[dtype],
    gae_lambda: Scalar[dtype],
    rollout_rewards: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), ImmutAnyOrigin
    ],
    rollout_dones: LayoutTensor[
        DType.int32, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), ImmutAnyOrigin
    ],
    rollout_values: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), ImmutAnyOrigin
    ],
    bootstrap_values: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, 1), ImmutAnyOrigin
    ],
    rollout_advantages: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin
    ],
    rollout_returns: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin
    ],
):
    """Compute GAE advantages and returns.

    Each thread processes one environment, computing advantages backwards
    through the rollout using the GAE formula:
        δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t = δ_t + γ * λ * (1 - done_t) * A_{t+1}

    Args:
        gamma: Discount factor (typically 0.99).
        gae_lambda: GAE lambda parameter (typically 0.95).
        rollout_rewards: Rewards for each step [NUM_ENVS, ROLLOUT_LEN].
        rollout_dones: Done flags for each step [NUM_ENVS, ROLLOUT_LEN].
        rollout_values: Value estimates for each step [NUM_ENVS, ROLLOUT_LEN].
        bootstrap_values: Value estimates for final states [NUM_ENVS, 1].
        rollout_advantages: Output advantages [NUM_ENVS, ROLLOUT_LEN].
        rollout_returns: Output returns (advantages + values) [NUM_ENVS, ROLLOUT_LEN].
    """
    var env_idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env_idx >= NUM_ENVS:
        return

    var gae: Scalar[dtype] = 0
    var next_value = rebind[Scalar[dtype]](bootstrap_values[env_idx, 0])

    for t in range(ROLLOUT_LEN - 1, -1, -1):
        var reward = rebind[Scalar[dtype]](rollout_rewards[env_idx, t])
        var done = rollout_dones[env_idx, t]
        var value = rebind[Scalar[dtype]](rollout_values[env_idx, t])

        var not_done = Scalar[dtype](1.0) if done == 0 else Scalar[dtype](0.0)
        var delta = reward + gamma * next_value * not_done - value
        gae = delta + gamma * gae_lambda * not_done * gae

        rollout_advantages[env_idx, t] = gae
        rollout_returns[env_idx, t] = gae + value
        next_value = value
