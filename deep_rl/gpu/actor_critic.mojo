"""GPU Actor-Critic networks for deep RL (DDPG, TD3, SAC).

This module provides GPU-accelerated Actor-Critic operations using
tiled matrix multiplication and block primitives.

Networks:
- Actor: obs -> relu -> relu -> tanh (scaled) -> action
- Critic: (obs, action) -> relu -> relu -> Q-value
- StochasticActor: obs -> relu -> relu -> (mean, log_std) for SAC

Uses kernels from linear.mojo and mlp.mojo:
- linear_forward_relu_kernel for hidden layers
- linear_forward_tanh_kernel for Actor output
- linear_forward_kernel for Critic output (no activation)
"""

from gpu import thread_idx, block_idx, block_dim, barrier, block
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from math import tanh as math_tanh, exp, log, sqrt


# ============================================================================
# ReLU Gradient Kernel (needed for Actor/Critic backward pass)
# ============================================================================

fn relu_grad_kernel[
    dtype: DType,
    SIZE: Int,
    TPB: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    pre_activation: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
):
    """Compute ReLU gradient: grad = 1 if x > 0 else 0."""
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)

    if global_i < SIZE:
        x = rebind[Scalar[dtype]](pre_activation[global_i])
        output[global_i] = 1 if x > 0 else 0


fn relu_grad_mul_kernel[
    dtype: DType,
    SIZE: Int,
    TPB: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    upstream_grad: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
    pre_activation: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
):
    """Fused ReLU gradient * upstream: output = upstream * (1 if x > 0 else 0)."""
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)

    if global_i < SIZE:
        upstream = rebind[Scalar[dtype]](upstream_grad[global_i])
        x = rebind[Scalar[dtype]](pre_activation[global_i])
        output[global_i] = upstream if x > 0 else 0


# ============================================================================
# Tanh Output Gradient Kernel (for Actor output layer)
# ============================================================================

fn tanh_grad_mul_kernel[
    dtype: DType,
    SIZE: Int,
    TPB: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    upstream_grad: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
    tanh_output: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
):
    """Fused tanh gradient * upstream: output = upstream * (1 - y^2) where y = tanh(x)."""
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)

    if global_i < SIZE:
        upstream = rebind[Scalar[dtype]](upstream_grad[global_i])
        y = rebind[Scalar[dtype]](tanh_output[global_i])
        output[global_i] = upstream * (1 - y * y)


# ============================================================================
# Action Scaling Kernel (for Actor output)
# ============================================================================

fn scale_action_kernel[
    dtype: DType,
    SIZE: Int,
    TPB: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    tanh_output: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
    action_scale: Scalar[dtype],
):
    """Scale tanh output to action range: output = tanh_output * action_scale."""
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)

    if global_i < SIZE:
        output[global_i] = rebind[Scalar[dtype]](tanh_output[global_i]) * action_scale


fn scale_grad_kernel[
    dtype: DType,
    SIZE: Int,
    TPB: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    upstream_grad: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
    action_scale: Scalar[dtype],
):
    """Scale gradient back through action scaling: output = upstream / action_scale."""
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)

    if global_i < SIZE:
        output[global_i] = rebind[Scalar[dtype]](upstream_grad[global_i]) / action_scale


# ============================================================================
# Concatenation Kernel (for Critic input: obs + action)
# ============================================================================

fn concat_obs_action_kernel[
    dtype: DType,
    BATCH: Int,
    OBS_DIM: Int,
    ACTION_DIM: Int,
    TPB: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(BATCH, OBS_DIM + ACTION_DIM), MutAnyOrigin],
    obs: LayoutTensor[dtype, Layout.row_major(BATCH, OBS_DIM), ImmutAnyOrigin],
    action: LayoutTensor[dtype, Layout.row_major(BATCH, ACTION_DIM), ImmutAnyOrigin],
):
    """Concatenate observations and actions: output = [obs, action]."""
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    comptime TOTAL_DIM = OBS_DIM + ACTION_DIM
    total_elements = BATCH * TOTAL_DIM

    if global_i < total_elements:
        batch_idx = global_i // TOTAL_DIM
        col_idx = global_i % TOTAL_DIM

        if col_idx < OBS_DIM:
            output[batch_idx, col_idx] = obs[batch_idx, col_idx]
        else:
            output[batch_idx, col_idx] = action[batch_idx, col_idx - OBS_DIM]


fn split_grad_kernel[
    dtype: DType,
    BATCH: Int,
    OBS_DIM: Int,
    ACTION_DIM: Int,
    TPB: Int,
](
    d_obs: LayoutTensor[dtype, Layout.row_major(BATCH, OBS_DIM), MutAnyOrigin],
    d_action: LayoutTensor[dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin],
    d_concat: LayoutTensor[dtype, Layout.row_major(BATCH, OBS_DIM + ACTION_DIM), ImmutAnyOrigin],
):
    """Split gradient from concatenated tensor back to obs and action gradients."""
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    comptime TOTAL_DIM = OBS_DIM + ACTION_DIM
    total_elements = BATCH * TOTAL_DIM

    if global_i < total_elements:
        batch_idx = global_i // TOTAL_DIM
        col_idx = global_i % TOTAL_DIM

        if col_idx < OBS_DIM:
            d_obs[batch_idx, col_idx] = d_concat[batch_idx, col_idx]
        else:
            d_action[batch_idx, col_idx - OBS_DIM] = d_concat[batch_idx, col_idx]


# ============================================================================
# Stochastic Actor Kernels (for SAC)
# ============================================================================

fn split_mean_log_std_kernel[
    dtype: DType,
    BATCH: Int,
    ACTION_DIM: Int,
    TPB: Int,
](
    mean: LayoutTensor[dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin],
    log_std: LayoutTensor[dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin],
    combined: LayoutTensor[dtype, Layout.row_major(BATCH, ACTION_DIM * 2), ImmutAnyOrigin],
):
    """Split combined output into mean and log_std for SAC."""
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    total_elements = BATCH * ACTION_DIM

    if global_i < total_elements:
        batch_idx = global_i // ACTION_DIM
        action_idx = global_i % ACTION_DIM

        mean[batch_idx, action_idx] = combined[batch_idx, action_idx]
        log_std[batch_idx, action_idx] = combined[batch_idx, ACTION_DIM + action_idx]


fn clamp_log_std_kernel[
    dtype: DType,
    SIZE: Int,
    TPB: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    log_std: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
    log_std_min: Scalar[dtype],
    log_std_max: Scalar[dtype],
):
    """Clamp log_std to valid range."""
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)

    if global_i < SIZE:
        val = rebind[Scalar[dtype]](log_std[global_i])
        if val < log_std_min:
            output[global_i] = log_std_min
        elif val > log_std_max:
            output[global_i] = log_std_max
        else:
            output[global_i] = val


fn sample_gaussian_kernel[
    dtype: DType,
    SIZE: Int,
    TPB: Int,
](
    action: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    mean: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
    std: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
    noise: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
):
    """Sample from Gaussian: action = mean + std * noise (noise pre-generated on CPU)."""
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)

    if global_i < SIZE:
        m = rebind[Scalar[dtype]](mean[global_i])
        s = rebind[Scalar[dtype]](std[global_i])
        n = rebind[Scalar[dtype]](noise[global_i])
        action[global_i] = m + s * n


fn compute_log_prob_kernel[
    dtype: DType,
    BATCH: Int,
    ACTION_DIM: Int,
    TPB: Int,
](
    log_prob: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    action: LayoutTensor[dtype, Layout.row_major(BATCH, ACTION_DIM), ImmutAnyOrigin],
    mean: LayoutTensor[dtype, Layout.row_major(BATCH, ACTION_DIM), ImmutAnyOrigin],
    log_std: LayoutTensor[dtype, Layout.row_major(BATCH, ACTION_DIM), ImmutAnyOrigin],
):
    """Compute log probability of actions under Gaussian policy.

    log_prob = sum(-0.5 * ((action - mean) / std)^2 - log_std - 0.5 * log(2*pi))

    Note: This kernel uses one thread per batch element, summing over action dimensions.
    For large ACTION_DIM, consider a reduction-based approach.
    """
    batch_idx = Int(block_dim.x * block_idx.x + thread_idx.x)

    if batch_idx < BATCH:
        var total: Scalar[dtype] = 0
        comptime LOG_2PI: Float64 = 1.8378770664093453  # log(2 * pi)

        @parameter
        for i in range(ACTION_DIM):
            a = rebind[Scalar[dtype]](action[batch_idx, i])
            m = rebind[Scalar[dtype]](mean[batch_idx, i])
            ls = rebind[Scalar[dtype]](log_std[batch_idx, i])
            std = exp(ls)

            z = (a - m) / std
            total += -0.5 * z * z - ls - 0.5 * LOG_2PI.cast[dtype]()

        log_prob[batch_idx] = total


fn squash_action_kernel[
    dtype: DType,
    SIZE: Int,
    TPB: Int,
](
    output: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    pre_squash: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
    action_scale: Scalar[dtype],
):
    """Squash action through tanh and scale: output = tanh(pre_squash) * action_scale."""
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)

    if global_i < SIZE:
        x = rebind[Scalar[dtype]](pre_squash[global_i])
        output[global_i] = math_tanh(x) * action_scale


fn squash_log_prob_correction_kernel[
    dtype: DType,
    BATCH: Int,
    ACTION_DIM: Int,
    TPB: Int,
](
    log_prob: LayoutTensor[dtype, Layout.row_major(BATCH), MutAnyOrigin],
    pre_squash: LayoutTensor[dtype, Layout.row_major(BATCH, ACTION_DIM), ImmutAnyOrigin],
):
    """Apply tanh squashing correction to log probability.

    Correction: sum(log(1 - tanh(x)^2 + eps))
    This accounts for the change of variables when applying tanh.
    """
    batch_idx = Int(block_dim.x * block_idx.x + thread_idx.x)

    if batch_idx < BATCH:
        var correction: Scalar[dtype] = 0
        comptime EPS: Float64 = 1e-6

        @parameter
        for i in range(ACTION_DIM):
            x = rebind[Scalar[dtype]](pre_squash[batch_idx, i])
            tanh_x = math_tanh(x)
            # log(1 - tanh^2 + eps) = log(sech^2 + eps)
            correction += log(1 - tanh_x * tanh_x + EPS.cast[dtype]())

        log_prob[batch_idx] = rebind[Scalar[dtype]](log_prob[batch_idx]) - correction


# ============================================================================
# Re-export commonly used kernels from linear.mojo and mlp.mojo
# (These are imported by users of this module)
# ============================================================================

# Forward pass kernels:
# - linear_forward_relu_kernel: For Actor/Critic hidden layers (from linear.mojo)
# - linear_forward_tanh_kernel: For Actor output layer (from mlp.mojo)
# - linear_forward_kernel: For Critic output layer (from linear.mojo or mlp.mojo)

# Backward pass kernels:
# - linear_backward_dW_kernel: Weight gradient (from linear.mojo)
# - linear_backward_db_kernel: Bias gradient (from linear.mojo)
# - linear_backward_dx_kernel: Input gradient (from linear.mojo)
# - tanh_grad_kernel: Tanh gradient (from mlp.mojo)
# - elementwise_mul_kernel: For gradient chaining (from mlp.mojo)

# Optimizer kernels:
# - adam_update_kernel: Adam optimizer (from linear.mojo or mlp.mojo)
# - soft_update_kernel: Target network update (from linear.mojo or mlp.mojo)
