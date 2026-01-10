"""Test if @always_inline functions can be used as modular sub-kernels inside GPU code."""

from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from math import exp

comptime dtype = DType.float32


# =============================================================================
# Sub-kernel functions (modular building blocks)
# =============================================================================


@always_inline
fn inline_matmul_element[
    M: Int, K: Int, N: Int
](
    row: Int,
    col: Int,
    A: LayoutTensor[dtype, Layout.row_major(M, K), ImmutAnyOrigin],
    B: LayoutTensor[dtype, Layout.row_major(K, N), ImmutAnyOrigin],
) -> Scalar[dtype]:
    """Compute a single element of A @ B."""
    var acc: Scalar[dtype] = 0
    for k in range(K):
        acc += rebind[Scalar[dtype]](A[row, k]) * rebind[Scalar[dtype]](B[k, col])
    return acc


@always_inline
fn inline_relu(x: Scalar[dtype]) -> Scalar[dtype]:
    """ReLU activation."""
    return x if x > 0 else Scalar[dtype](0)


@always_inline
fn inline_softmax_row[N: Int](
    row: Int,
    logits: LayoutTensor[dtype, Layout.row_major(1, N), ImmutAnyOrigin],
    probs: LayoutTensor[dtype, Layout.row_major(1, N), MutAnyOrigin],
):
    """Compute softmax for a single row (in-thread, no sync needed)."""
    # Find max for numerical stability
    var max_val: Scalar[dtype] = rebind[Scalar[dtype]](logits[0, 0])
    for j in range(N):
        var v = rebind[Scalar[dtype]](logits[0, j])
        if v > max_val:
            max_val = v

    # Compute exp and sum
    var sum_exp: Scalar[dtype] = 0
    for j in range(N):
        var e = exp(rebind[Scalar[dtype]](logits[0, j]) - max_val)
        probs[0, j] = e
        sum_exp += e

    # Normalize
    for j in range(N):
        probs[0, j] = rebind[Scalar[dtype]](probs[0, j]) / sum_exp


# =============================================================================
# Mega fused kernel using inline sub-kernels
# =============================================================================


fn fused_forward_kernel[
    NUM_ENVS: Int,
    OBS_DIM: Int,
    HIDDEN_DIM: Int,
    NUM_ACTIONS: Int,
](
    # Inputs
    obs: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, OBS_DIM), ImmutAnyOrigin],
    W1: LayoutTensor[dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin],
    b1: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin],
    W_actor: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), ImmutAnyOrigin],
    b_actor: LayoutTensor[dtype, Layout.row_major(NUM_ACTIONS), ImmutAnyOrigin],
    W_critic: LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, 1), ImmutAnyOrigin],
    b_critic: LayoutTensor[dtype, Layout.row_major(1), ImmutAnyOrigin],
    # Outputs
    hidden: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin],
    probs: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), MutAnyOrigin],
    values: LayoutTensor[dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin],
):
    """Fused forward pass: obs -> hidden -> (probs, value).

    Each thread handles one environment (like the original a2c.mojo approach).
    Uses inline sub-kernels for modularity.
    """
    var env_idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env_idx >= NUM_ENVS:
        return

    # Get this env's observation row
    var obs_row = LayoutTensor[dtype, Layout.row_major(1, OBS_DIM), ImmutAnyOrigin](
        obs.ptr + env_idx * OBS_DIM
    )

    # === Layer 1: obs @ W1 + b1 -> ReLU -> hidden ===
    for h in range(HIDDEN_DIM):
        var acc = inline_matmul_element[1, OBS_DIM, HIDDEN_DIM](0, h, obs_row, W1)
        acc += rebind[Scalar[dtype]](b1[h])
        hidden[env_idx, h] = inline_relu(acc)

    # Get this env's hidden row
    var hidden_row = LayoutTensor[dtype, Layout.row_major(1, HIDDEN_DIM), ImmutAnyOrigin](
        hidden.ptr + env_idx * HIDDEN_DIM
    )

    # === Actor: hidden @ W_actor + b_actor -> softmax -> probs ===
    # First compute logits in probs buffer
    var logits_row = LayoutTensor[dtype, Layout.row_major(1, NUM_ACTIONS), MutAnyOrigin](
        probs.ptr + env_idx * NUM_ACTIONS
    )
    for a in range(NUM_ACTIONS):
        var acc = inline_matmul_element[1, HIDDEN_DIM, NUM_ACTIONS](0, a, hidden_row, W_actor)
        acc += rebind[Scalar[dtype]](b_actor[a])
        logits_row[0, a] = acc

    # Softmax in-place
    var logits_immut = LayoutTensor[dtype, Layout.row_major(1, NUM_ACTIONS), ImmutAnyOrigin](
        probs.ptr + env_idx * NUM_ACTIONS
    )
    inline_softmax_row[NUM_ACTIONS](0, logits_immut, logits_row)

    # === Critic: hidden @ W_critic + b_critic -> value ===
    var val_acc: Scalar[dtype] = 0
    for h in range(HIDDEN_DIM):
        val_acc += rebind[Scalar[dtype]](hidden_row[0, h]) * rebind[Scalar[dtype]](W_critic[h, 0])
    val_acc += rebind[Scalar[dtype]](b_critic[0])
    values[env_idx, 0] = val_acc


# =============================================================================
# Test
# =============================================================================


fn main() raises:
    print("Testing inline sub-kernels inside GPU mega kernel")
    print("=" * 55)

    comptime NUM_ENVS = 64
    comptime OBS_DIM = 4
    comptime HIDDEN_DIM = 32
    comptime NUM_ACTIONS = 2

    with DeviceContext() as ctx:
        # Allocate buffers
        var obs_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * OBS_DIM)
        var W1_buf = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
        var b1_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
        var W_actor_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * NUM_ACTIONS)
        var b_actor_buf = ctx.enqueue_create_buffer[dtype](NUM_ACTIONS)
        var W_critic_buf = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * 1)
        var b_critic_buf = ctx.enqueue_create_buffer[dtype](1)
        var hidden_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * HIDDEN_DIM)
        var probs_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * NUM_ACTIONS)
        var values_buf = ctx.enqueue_create_buffer[dtype](NUM_ENVS * 1)

        # Initialize with simple values
        with obs_buf.map_to_host() as h:
            for i in range(NUM_ENVS * OBS_DIM):
                h[i] = Scalar[dtype](0.1)
        with W1_buf.map_to_host() as h:
            for i in range(OBS_DIM * HIDDEN_DIM):
                h[i] = Scalar[dtype](0.1)
        with b1_buf.map_to_host() as h:
            for i in range(HIDDEN_DIM):
                h[i] = Scalar[dtype](0.0)
        with W_actor_buf.map_to_host() as h:
            for i in range(HIDDEN_DIM * NUM_ACTIONS):
                h[i] = Scalar[dtype](0.1)
        with b_actor_buf.map_to_host() as h:
            for i in range(NUM_ACTIONS):
                h[i] = Scalar[dtype](0.0)
        with W_critic_buf.map_to_host() as h:
            for i in range(HIDDEN_DIM * 1):
                h[i] = Scalar[dtype](0.1)
        with b_critic_buf.map_to_host() as h:
            h[0] = Scalar[dtype](0.0)

        # Create tensors
        var obs = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, OBS_DIM), ImmutAnyOrigin](obs_buf)
        var W1 = LayoutTensor[dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin](W1_buf)
        var b1 = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin](b1_buf)
        var W_actor = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, NUM_ACTIONS), ImmutAnyOrigin](W_actor_buf)
        var b_actor = LayoutTensor[dtype, Layout.row_major(NUM_ACTIONS), ImmutAnyOrigin](b_actor_buf)
        var W_critic = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, 1), ImmutAnyOrigin](W_critic_buf)
        var b_critic = LayoutTensor[dtype, Layout.row_major(1), ImmutAnyOrigin](b_critic_buf)
        var hidden = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, HIDDEN_DIM), MutAnyOrigin](hidden_buf)
        var probs = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, NUM_ACTIONS), MutAnyOrigin](probs_buf)
        var values = LayoutTensor[dtype, Layout.row_major(NUM_ENVS, 1), MutAnyOrigin](values_buf)

        # Run fused kernel
        ctx.enqueue_function_checked[
            fused_forward_kernel[NUM_ENVS, OBS_DIM, HIDDEN_DIM, NUM_ACTIONS],
            fused_forward_kernel[NUM_ENVS, OBS_DIM, HIDDEN_DIM, NUM_ACTIONS],
        ](
            obs, W1, b1, W_actor, b_actor, W_critic, b_critic,
            hidden, probs, values,
            grid_dim=(1, 1),
            block_dim=(NUM_ENVS, 1),
        )
        ctx.synchronize()

        # Check results
        with probs_buf.map_to_host() as p, values_buf.map_to_host() as v:
            print("Env 0 probs:", p[0], p[1], "(should sum to 1.0)")
            print("Env 0 value:", v[0])

            var prob_sum = Float64(p[0]) + Float64(p[1])
            if abs(prob_sum - 1.0) < 0.001:
                print("SUCCESS: Softmax probs sum to 1.0")
            else:
                print("ERROR: Probs sum to", prob_sum)

    print("=" * 55)
    print("Test passed! @always_inline sub-kernels work inside GPU kernels.")
