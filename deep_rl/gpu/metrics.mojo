"""Episode tracking kernel for GPU RL algorithms.

This kernel tracks episode completions and accumulates rewards during
rollout collection, enabling metrics logging without CPU synchronization.
"""

from gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor

# =============================================================================
# Constants
# =============================================================================

comptime dtype = DType.float32


# =============================================================================
# Episode Tracking Kernel
# =============================================================================


fn track_episodes_kernel[
    NUM_ENVS: Int, ROLLOUT_LEN: Int
](
    rollout_rewards: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), ImmutAnyOrigin
    ],
    rollout_dones: LayoutTensor[
        DType.int32, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), ImmutAnyOrigin
    ],
    episode_rewards: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
    ],
    completed_episodes: LayoutTensor[
        DType.int32, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin
    ],
    completed_rewards: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, ROLLOUT_LEN), MutAnyOrigin
    ],
):
    """Track episode completions and accumulate rewards.

    Each thread processes one environment across the rollout, tracking:
    - Running reward accumulation within episodes
    - Episode completion events
    - Final rewards for completed episodes

    Args:
        rollout_rewards: Rewards for each step [NUM_ENVS, ROLLOUT_LEN].
        rollout_dones: Done flags for each step [NUM_ENVS, ROLLOUT_LEN].
        episode_rewards: Running reward accumulator per env [NUM_ENVS, 1].
            This persists across rollouts for episodes spanning multiple rollouts.
        completed_episodes: Count of completed episodes per env this rollout [NUM_ENVS, 1].
        completed_rewards: Store reward at each completion [NUM_ENVS, ROLLOUT_LEN].
            Index [env, i] stores the i-th completed episode's total reward.
    """
    var env_idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env_idx >= NUM_ENVS:
        return

    var running_reward = rebind[Scalar[dtype]](episode_rewards[env_idx, 0])
    var num_completed: Int32 = 0

    for t in range(ROLLOUT_LEN):
        var reward = rebind[Scalar[dtype]](rollout_rewards[env_idx, t])
        var done = rollout_dones[env_idx, t]

        running_reward += reward

        if done != 0:
            # Episode completed - store the total reward
            completed_rewards[env_idx, Int(num_completed)] = running_reward
            num_completed += 1
            running_reward = Scalar[dtype](0)  # Reset for next episode

    # Write back
    episode_rewards[env_idx, 0] = running_reward
    completed_episodes[env_idx, 0] = num_completed
