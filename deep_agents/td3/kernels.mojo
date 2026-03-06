"""TD3-specific GPU kernels.

## Noise
- add_gaussian_noise_kernel: Add clipped Gaussian noise to actions (TD3 target smoothing)
"""

from std.gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from std.math import sqrt, log, cos
from std.random.philox import Random as PhiloxRandom


# =============================================================================
# Gaussian Noise for TD3 Target Smoothing
# =============================================================================


@always_inline
fn add_gaussian_noise_kernel[
    dtype: DType,
    BATCH: Int,
    ACTION_DIM: Int,
](
    noisy_actions: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    actions: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin
    ],
    noise_std: Scalar[dtype],
    noise_clip: Scalar[dtype],
    action_min: Scalar[dtype],
    action_max: Scalar[dtype],
    rng_seed: Scalar[DType.uint32],
):
    """Add clipped Gaussian exploration noise to actions (TD3-style).

    Each element gets independent noise from N(0, noise_std²), clipped to
    [-noise_clip, noise_clip], then the result is clipped to [action_min, action_max].

    Uses PhiloxRandom for GPU-safe noise generation (no Float64).
    One thread per (batch, action_dim) element.

    Args:
        noisy_actions: Output noisy actions [BATCH, ACTION_DIM].
        actions:       Clean actions from actor [BATCH, ACTION_DIM].
        noise_std:     Noise standard deviation.
        noise_clip:    Maximum absolute noise value.
        action_min:    Minimum action value (e.g. -action_scale).
        action_max:    Maximum action value (e.g. +action_scale).
        rng_seed:      Random seed (should vary per call).
    """
    var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
    if tid >= BATCH * ACTION_DIM:
        return

    var b = tid // ACTION_DIM
    var a = tid % ACTION_DIM

    # PhiloxRandom Box-Muller for Gaussian noise
    var philox = PhiloxRandom(
        seed=UInt64(rng_seed) + UInt64(b) * UInt64(ACTION_DIM) + UInt64(a),
        offset=0,
    )
    var rand_vals = philox.step_uniform()
    var u1 = Float32(rand_vals[0]) + Float32(1e-8)
    var u2 = Float32(rand_vals[1])
    var mag = sqrt(-2.0 * log(u1))
    var z = Scalar[dtype](mag * cos(u2 * Float32(6.283185307179586)))

    # Scale and clip noise
    var noise = z * noise_std
    if noise < -noise_clip:
        noise = -noise_clip
    if noise > noise_clip:
        noise = noise_clip

    # Apply noise and clip to action range
    var noisy = actions[b, a] + noise
    if noisy < action_min:
        noisy = action_min
    if noisy > action_max:
        noisy = action_max

    noisy_actions[b, a] = noisy
