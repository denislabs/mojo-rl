"""Shared helper functions for on-policy agents (PPO).

Analog of offpolicy_helpers.mojo — extracts List-based GAE computation,
advantage normalization, and minibatch shuffling into one place.

These helpers work with List[Scalar[dtype]] rollout buffers (heap-allocated)
as opposed to the InlineArray-based compute_gae_inline / normalize_inline
utilities in core/utils/ which are used by A2C (stack-allocated buffers).

Usage in PPO collect_rollout/compute_advantages/update_epochs methods:
    compute_gae_list[dtype](
        state.buffer_rewards, state.buffer_values, state.buffer_dones,
        last_value, state.buffer_idx, gamma, gae_lambda,
        state._advantages, state._returns
    )
    normalize_advantages_list[dtype](state._advantages, state.buffer_idx)
    fisher_yates_shuffle(state._indices, state.buffer_idx)
"""

from std.math import sqrt
from std.random import random_float64

from mojo_rl.nn.constants import dtype as default_dtype


fn compute_gae_list[
    DTYPE: DType
](
    buffer_rewards: List[Scalar[DTYPE]],
    buffer_values: List[Scalar[DTYPE]],
    buffer_dones: List[Bool],
    last_value: Scalar[DTYPE],
    buffer_len: Int,
    gamma: Float64,
    gae_lambda: Float64,
    mut advantages: List[Scalar[DTYPE]],
    mut returns: List[Scalar[DTYPE]],
) -> None:
    """Compute Generalized Advantage Estimation (GAE) over a rollout buffer.

    Iterates backwards over [0, buffer_len) to accumulate discounted TD
    residuals. Episode boundaries (done=True) reset the GAE accumulator
    and zero the bootstrap value for that step.

    Parameters:
        DTYPE: Data type for all tensors (float32 or float64).

    Args:
        buffer_rewards: Collected rewards per step [buffer_len].
        buffer_values: Critic value estimates per step [buffer_len].
        buffer_dones: Episode-done flags per step [buffer_len].
        last_value: Bootstrap value V(s_{T}) for the step after the last.
        buffer_len: Number of valid steps in the buffer.
        gamma: Discount factor γ.
        gae_lambda: GAE λ parameter (0=TD, 1=MC).
        advantages: Output list [rollout_len] — filled in-place.
        returns: Output list [rollout_len] — filled as advantage + value.
    """
    var gae = Scalar[DTYPE](0.0)
    var gae_decay = Scalar[DTYPE](gamma * gae_lambda)
    var gamma_s = Scalar[DTYPE](gamma)

    for t in range(buffer_len - 1, -1, -1):
        var next_val: Scalar[DTYPE]
        if t == buffer_len - 1:
            next_val = last_value
        else:
            next_val = buffer_values[t + 1]

        # Episode boundary: zero the bootstrap and reset GAE accumulator
        if buffer_dones[t]:
            next_val = Scalar[DTYPE](0.0)
            gae = Scalar[DTYPE](0.0)

        # TD residual: δ = r + γV(s') - V(s)
        var delta = buffer_rewards[t] + gamma_s * next_val - buffer_values[t]

        # GAE accumulation: A_t = δ_t + γλ A_{t+1}
        gae = delta + gae_decay * gae

        advantages[t] = gae
        returns[t] = gae + buffer_values[t]


fn normalize_advantages_list[
    DTYPE: DType
](mut advantages: List[Scalar[DTYPE]], n: Int, eps: Float64 = 1e-8) -> None:
    """Normalize advantages to zero mean and unit variance in-place.

    Operates on the first `n` elements of `advantages`. No-op when n <= 1.

    Parameters:
        DTYPE: Data type (float32 or float64).

    Args:
        advantages: Advantage list to normalize in-place.
        n: Number of valid elements to normalize.
        eps: Epsilon for numerical stability (default: 1e-8).
    """
    if n <= 1:
        return

    var mean = Scalar[DTYPE](0.0)
    for i in range(n):
        mean += advantages[i]
    mean /= Scalar[DTYPE](n)

    var var_sum = Scalar[DTYPE](0.0)
    for i in range(n):
        var diff = advantages[i] - mean
        var_sum += diff * diff

    var std = sqrt(var_sum / Scalar[DTYPE](n) + Scalar[DTYPE](eps))
    for i in range(n):
        advantages[i] = (advantages[i] - mean) / std


fn fisher_yates_shuffle(mut indices: List[Int], n: Int) -> None:
    """In-place Fisher-Yates shuffle of the first n elements in indices.

    Produces a uniformly random permutation in O(n) time. Used for
    minibatch sampling in PPO's update_epochs.

    Args:
        indices: Integer index list to shuffle in-place.
        n: Number of elements to shuffle (must be <= len(indices)).
    """
    for i in range(n - 1, 0, -1):
        var j = Int(random_float64() * Float64(i + 1))
        var temp = indices[i]
        indices[i] = indices[j]
        indices[j] = temp
