from layout import Layout, LayoutTensor
from gpu import thread_idx, block_idx, block_dim
from math import sqrt

# =============================================================================
# Constants (module-level)
# =============================================================================

comptime dtype = DType.float32

# =============================================================================
# Reduce and SGD Kernels - Parameterized for composability
# =============================================================================


fn reduce_kernel[
    SIZE: Int, NUM_ENVS: Int
](
    reduced: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    per_env: LayoutTensor[
        dtype, Layout.row_major(NUM_ENVS, SIZE), ImmutAnyOrigin
    ],
):
    """Generic reduce kernel - averages gradients across environments."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SIZE:
        return
    var sum_val: Scalar[dtype] = 0
    for env in range(NUM_ENVS):
        sum_val += rebind[Scalar[dtype]](per_env[env, idx])
    reduced[idx] = sum_val / Scalar[dtype](NUM_ENVS)


fn sgd_kernel[
    SIZE: Int
](
    weights: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    grads: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
    lr: Scalar[dtype],
):
    """Generic SGD kernel - updates weights with gradients."""
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SIZE:
        return
    weights[idx] = rebind[Scalar[dtype]](weights[idx]) - lr * rebind[
        Scalar[dtype]
    ](grads[idx])


fn adam_kernel[
    SIZE: Int
](
    weights: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    grads: LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin],
    m: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    v: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin],
    lr: Scalar[dtype],
    beta1: Scalar[dtype],
    beta2: Scalar[dtype],
    eps: Scalar[dtype],
    bias_correction1: Scalar[dtype],
    bias_correction2: Scalar[dtype],
):
    """Adam optimizer kernel.

    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad^2
    weights -= lr * (m / bias_correction1) / (sqrt(v / bias_correction2) + eps)
    """
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= SIZE:
        return

    var g = rebind[Scalar[dtype]](grads[idx])
    var m_val = rebind[Scalar[dtype]](m[idx])
    var v_val = rebind[Scalar[dtype]](v[idx])

    # Update moments
    var m_new = beta1 * m_val + (Scalar[dtype](1) - beta1) * g
    var v_new = beta2 * v_val + (Scalar[dtype](1) - beta2) * g * g

    # Bias-corrected estimates
    var m_hat = m_new / bias_correction1
    var v_hat = v_new / bias_correction2

    # Update weights
    weights[idx] = rebind[Scalar[dtype]](weights[idx]) - lr * m_hat / (
        sqrt(v_hat) + eps
    )

    # Store updated moments
    m[idx] = m_new
    v[idx] = v_new
