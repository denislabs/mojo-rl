"""Tests for GPU Actor-Critic operations.

Run with: pixi run -e apple mojo run test_gpu_actor_critic.mojo
"""

from gpu import thread_idx, block_idx, block_dim, barrier, block
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from builtin.math import abs
from math import tanh as math_tanh, exp


# Test configuration
comptime dtype = DType.float32


# ============================================================================
# GPU Kernels (inline for testing)
# ============================================================================

fn relu_grad_kernel[dt: DType, SIZE: Int, TPB: Int](
    output: LayoutTensor[dt, Layout.row_major(SIZE), MutAnyOrigin],
    pre_activation: LayoutTensor[dt, Layout.row_major(SIZE), ImmutAnyOrigin],
):
    """Compute ReLU gradient: grad = 1 if x > 0 else 0."""
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if global_i < SIZE:
        x = rebind[Scalar[dt]](pre_activation[global_i])
        output[global_i] = 1 if x > 0 else 0


fn concat_obs_action_kernel[dt: DType, BATCH: Int, OBS_DIM: Int, ACTION_DIM: Int, TPB: Int](
    output: LayoutTensor[dt, Layout.row_major(BATCH, OBS_DIM + ACTION_DIM), MutAnyOrigin],
    obs: LayoutTensor[dt, Layout.row_major(BATCH, OBS_DIM), ImmutAnyOrigin],
    action: LayoutTensor[dt, Layout.row_major(BATCH, ACTION_DIM), ImmutAnyOrigin],
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


fn split_mean_log_std_kernel[dt: DType, BATCH: Int, ACTION_DIM: Int, TPB: Int](
    mean: LayoutTensor[dt, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin],
    log_std: LayoutTensor[dt, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin],
    combined: LayoutTensor[dt, Layout.row_major(BATCH, ACTION_DIM * 2), ImmutAnyOrigin],
):
    """Split combined output into mean and log_std for SAC."""
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    total_elements = BATCH * ACTION_DIM

    if global_i < total_elements:
        batch_idx = global_i // ACTION_DIM
        action_idx = global_i % ACTION_DIM

        mean[batch_idx, action_idx] = combined[batch_idx, action_idx]
        log_std[batch_idx, action_idx] = combined[batch_idx, ACTION_DIM + action_idx]


fn clamp_log_std_kernel[dt: DType, SIZE: Int, TPB: Int](
    output: LayoutTensor[dt, Layout.row_major(SIZE), MutAnyOrigin],
    log_std: LayoutTensor[dt, Layout.row_major(SIZE), ImmutAnyOrigin],
    log_std_min: Scalar[dt],
    log_std_max: Scalar[dt],
):
    """Clamp log_std to valid range."""
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if global_i < SIZE:
        val = rebind[Scalar[dt]](log_std[global_i])
        if val < log_std_min:
            output[global_i] = log_std_min
        elif val > log_std_max:
            output[global_i] = log_std_max
        else:
            output[global_i] = val


fn squash_action_kernel[dt: DType, SIZE: Int, TPB: Int](
    output: LayoutTensor[dt, Layout.row_major(SIZE), MutAnyOrigin],
    pre_squash: LayoutTensor[dt, Layout.row_major(SIZE), ImmutAnyOrigin],
    action_scale: Scalar[dt],
):
    """Squash action through tanh and scale: output = tanh(pre_squash) * action_scale."""
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if global_i < SIZE:
        x = rebind[Scalar[dt]](pre_squash[global_i])
        output[global_i] = math_tanh(x) * action_scale


fn linear_forward_relu_kernel[dt: DType, BATCH: Int, IN_DIM: Int, OUT_DIM: Int, TILE: Int](
    output: LayoutTensor[dt, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    x: LayoutTensor[dt, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dt, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    b: LayoutTensor[dt, Layout.row_major(OUT_DIM), ImmutAnyOrigin],
):
    """Fused forward with ReLU: y = max(0, x @ W + b)."""
    local_row = Int(thread_idx.y)
    local_col = Int(thread_idx.x)
    global_row = Int(block_idx.y) * TILE + local_row
    global_col = Int(block_idx.x) * TILE + local_col

    x_shared = LayoutTensor[
        dt, Layout.row_major(TILE, TILE), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    W_shared = LayoutTensor[
        dt, Layout.row_major(TILE, TILE), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: Scalar[dt] = 0
    if global_col < OUT_DIM:
        acc = rebind[Scalar[dt]](b[global_col])

    comptime num_tiles = (IN_DIM + TILE - 1) // TILE

    @parameter
    for tile_idx in range(num_tiles):
        x_col = tile_idx * TILE + local_col
        if global_row < BATCH and x_col < IN_DIM:
            x_shared[local_row, local_col] = x[global_row, x_col]
        else:
            x_shared[local_row, local_col] = 0

        W_row = tile_idx * TILE + local_row
        if W_row < IN_DIM and global_col < OUT_DIM:
            W_shared[local_row, local_col] = W[W_row, global_col]
        else:
            W_shared[local_row, local_col] = 0

        barrier()

        @parameter
        for k in range(TILE):
            acc += rebind[Scalar[dt]](x_shared[local_row, k]) * rebind[Scalar[dt]](W_shared[k, local_col])

        barrier()

    if global_row < BATCH and global_col < OUT_DIM:
        output[global_row, global_col] = acc if acc > 0 else 0


fn linear_forward_tanh_kernel[dt: DType, BATCH: Int, IN_DIM: Int, OUT_DIM: Int, TILE: Int](
    output: LayoutTensor[dt, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    x: LayoutTensor[dt, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dt, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    b: LayoutTensor[dt, Layout.row_major(OUT_DIM), ImmutAnyOrigin],
):
    """Fused forward: y = tanh(x @ W + b)."""
    local_row = Int(thread_idx.y)
    local_col = Int(thread_idx.x)
    global_row = Int(block_idx.y) * TILE + local_row
    global_col = Int(block_idx.x) * TILE + local_col

    x_shared = LayoutTensor[
        dt, Layout.row_major(TILE, TILE), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    W_shared = LayoutTensor[
        dt, Layout.row_major(TILE, TILE), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: Scalar[dt] = 0
    if global_col < OUT_DIM:
        acc = rebind[Scalar[dt]](b[global_col])

    comptime num_tiles = (IN_DIM + TILE - 1) // TILE

    @parameter
    for tile_idx in range(num_tiles):
        x_col = tile_idx * TILE + local_col
        if global_row < BATCH and x_col < IN_DIM:
            x_shared[local_row, local_col] = x[global_row, x_col]
        else:
            x_shared[local_row, local_col] = 0

        W_row = tile_idx * TILE + local_row
        if W_row < IN_DIM and global_col < OUT_DIM:
            W_shared[local_row, local_col] = W[W_row, global_col]
        else:
            W_shared[local_row, local_col] = 0

        barrier()

        @parameter
        for k in range(TILE):
            acc += rebind[Scalar[dt]](x_shared[local_row, k]) * rebind[Scalar[dt]](W_shared[k, local_col])

        barrier()

    if global_row < BATCH and global_col < OUT_DIM:
        output[global_row, global_col] = math_tanh(acc)


fn linear_forward_kernel[dt: DType, BATCH: Int, IN_DIM: Int, OUT_DIM: Int, TILE: Int](
    output: LayoutTensor[dt, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin],
    x: LayoutTensor[dt, Layout.row_major(BATCH, IN_DIM), ImmutAnyOrigin],
    W: LayoutTensor[dt, Layout.row_major(IN_DIM, OUT_DIM), ImmutAnyOrigin],
    b: LayoutTensor[dt, Layout.row_major(OUT_DIM), ImmutAnyOrigin],
):
    """Forward: y = x @ W + b (no activation)."""
    local_row = Int(thread_idx.y)
    local_col = Int(thread_idx.x)
    global_row = Int(block_idx.y) * TILE + local_row
    global_col = Int(block_idx.x) * TILE + local_col

    x_shared = LayoutTensor[
        dt, Layout.row_major(TILE, TILE), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    W_shared = LayoutTensor[
        dt, Layout.row_major(TILE, TILE), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var acc: Scalar[dt] = 0
    if global_col < OUT_DIM:
        acc = rebind[Scalar[dt]](b[global_col])

    comptime num_tiles = (IN_DIM + TILE - 1) // TILE

    @parameter
    for tile_idx in range(num_tiles):
        x_col = tile_idx * TILE + local_col
        if global_row < BATCH and x_col < IN_DIM:
            x_shared[local_row, local_col] = x[global_row, x_col]
        else:
            x_shared[local_row, local_col] = 0

        W_row = tile_idx * TILE + local_row
        if W_row < IN_DIM and global_col < OUT_DIM:
            W_shared[local_row, local_col] = W[W_row, global_col]
        else:
            W_shared[local_row, local_col] = 0

        barrier()

        @parameter
        for k in range(TILE):
            acc += rebind[Scalar[dt]](x_shared[local_row, k]) * rebind[Scalar[dt]](W_shared[k, local_col])

        barrier()

    if global_row < BATCH and global_col < OUT_DIM:
        output[global_row, global_col] = acc


# ============================================================================
# Test Functions
# ============================================================================

fn test_relu_grad(ctx: DeviceContext) raises:
    """Test ReLU gradient kernel."""
    print("Testing relu_grad_kernel...")

    comptime SIZE = 16
    comptime TPB = 16

    out = ctx.enqueue_create_buffer[dtype](SIZE)
    pre_act = ctx.enqueue_create_buffer[dtype](SIZE)
    out.enqueue_fill(0)

    expected = ctx.enqueue_create_host_buffer[dtype](SIZE)
    expected.enqueue_fill(0)

    with pre_act.map_to_host() as pah:
        for i in range(SIZE):
            pah[i] = Float32(i) - Float32(8)  # [-8, -7, ..., 6, 7]
            expected[i] = Float32(1) if pah[i] > 0 else Float32(0)

    out_t = LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin](out.unsafe_ptr())
    pre_act_t = LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin](pre_act.unsafe_ptr())

    ctx.enqueue_function_checked[
        relu_grad_kernel[dtype, SIZE, TPB],
        relu_grad_kernel[dtype, SIZE, TPB]
    ](
        out_t, pre_act_t,
        grid_dim=(1,),
        block_dim=(TPB,),
    )
    ctx.synchronize()

    with out.map_to_host() as oh:
        for i in range(SIZE):
            if oh[i] != expected[i]:
                print("Mismatch at", i, ":", oh[i], "vs", expected[i])
                raise Error("relu_grad mismatch")
    print("  relu_grad: PASSED")


fn test_concat_obs_action(ctx: DeviceContext) raises:
    """Test observation-action concatenation kernel."""
    print("Testing concat_obs_action_kernel...")

    comptime BATCH = 4
    comptime OBS_DIM = 8
    comptime ACTION_DIM = 2
    comptime TOTAL_DIM = OBS_DIM + ACTION_DIM
    comptime TPB = 64

    out = ctx.enqueue_create_buffer[dtype](BATCH * TOTAL_DIM)
    obs = ctx.enqueue_create_buffer[dtype](BATCH * OBS_DIM)
    action = ctx.enqueue_create_buffer[dtype](BATCH * ACTION_DIM)
    out.enqueue_fill(0)

    with obs.map_to_host() as obsh, action.map_to_host() as acth:
        for i in range(BATCH * OBS_DIM):
            obsh[i] = Float32(1)
        for i in range(BATCH * ACTION_DIM):
            acth[i] = Float32(2)

    out_t = LayoutTensor[dtype, Layout.row_major(BATCH, TOTAL_DIM), MutAnyOrigin](out.unsafe_ptr())
    obs_t = LayoutTensor[dtype, Layout.row_major(BATCH, OBS_DIM), ImmutAnyOrigin](obs.unsafe_ptr())
    act_t = LayoutTensor[dtype, Layout.row_major(BATCH, ACTION_DIM), ImmutAnyOrigin](action.unsafe_ptr())

    comptime num_blocks = (BATCH * TOTAL_DIM + TPB - 1) // TPB
    ctx.enqueue_function_checked[
        concat_obs_action_kernel[dtype, BATCH, OBS_DIM, ACTION_DIM, TPB],
        concat_obs_action_kernel[dtype, BATCH, OBS_DIM, ACTION_DIM, TPB]
    ](
        out_t, obs_t, act_t,
        grid_dim=(num_blocks,),
        block_dim=(TPB,),
    )
    ctx.synchronize()

    with out.map_to_host() as oh:
        for batch in range(BATCH):
            for i in range(TOTAL_DIM):
                idx = batch * TOTAL_DIM + i
                expected_val = Float32(1) if i < OBS_DIM else Float32(2)
                if oh[idx] != expected_val:
                    print("Mismatch at batch", batch, "col", i, ":", oh[idx], "vs", expected_val)
                    raise Error("concat_obs_action mismatch")
    print("  concat_obs_action: PASSED")


fn test_actor_forward(ctx: DeviceContext) raises:
    """Test Actor forward pass: obs -> relu -> relu -> tanh."""
    print("Testing Actor forward pass (obs -> relu -> relu -> tanh)...")

    comptime BATCH = 4
    comptime OBS_DIM = 8
    comptime HIDDEN_DIM = 16
    comptime ACTION_DIM = 4
    comptime TILE = 4

    # Allocate buffers
    obs = ctx.enqueue_create_buffer[dtype](BATCH * OBS_DIM)
    W1 = ctx.enqueue_create_buffer[dtype](OBS_DIM * HIDDEN_DIM)
    b1 = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
    h1 = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN_DIM)
    W2 = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * HIDDEN_DIM)
    b2 = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
    h2 = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN_DIM)
    W3 = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * ACTION_DIM)
    b3 = ctx.enqueue_create_buffer[dtype](ACTION_DIM)
    action = ctx.enqueue_create_buffer[dtype](BATCH * ACTION_DIM)

    h1.enqueue_fill(0)
    h2.enqueue_fill(0)
    action.enqueue_fill(0)

    # Initialize weights
    with obs.map_to_host() as obsh, W1.map_to_host() as W1h, b1.map_to_host() as b1h:
        for i in range(BATCH * OBS_DIM):
            obsh[i] = Float32(0.1)
        for i in range(OBS_DIM * HIDDEN_DIM):
            W1h[i] = Float32(0.1)
        for i in range(HIDDEN_DIM):
            b1h[i] = Float32(0)

    with W2.map_to_host() as W2h, b2.map_to_host() as b2h:
        for i in range(HIDDEN_DIM * HIDDEN_DIM):
            W2h[i] = Float32(0.1)
        for i in range(HIDDEN_DIM):
            b2h[i] = Float32(0)

    with W3.map_to_host() as W3h, b3.map_to_host() as b3h:
        for i in range(HIDDEN_DIM * ACTION_DIM):
            W3h[i] = Float32(0.1)
        for i in range(ACTION_DIM):
            b3h[i] = Float32(0)

    # Create tensors
    obs_t = LayoutTensor[dtype, Layout.row_major(BATCH, OBS_DIM), ImmutAnyOrigin](obs.unsafe_ptr())
    W1_t = LayoutTensor[dtype, Layout.row_major(OBS_DIM, HIDDEN_DIM), ImmutAnyOrigin](W1.unsafe_ptr())
    b1_t = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin](b1.unsafe_ptr())
    h1_t = LayoutTensor[dtype, Layout.row_major(BATCH, HIDDEN_DIM), MutAnyOrigin](h1.unsafe_ptr())

    # Layer 1: obs -> h1 (ReLU)
    comptime blocks_y1 = (BATCH + TILE - 1) // TILE
    comptime blocks_x1 = (HIDDEN_DIM + TILE - 1) // TILE
    ctx.enqueue_function_checked[
        linear_forward_relu_kernel[dtype, BATCH, OBS_DIM, HIDDEN_DIM, TILE],
        linear_forward_relu_kernel[dtype, BATCH, OBS_DIM, HIDDEN_DIM, TILE]
    ](
        h1_t, obs_t, W1_t, b1_t,
        grid_dim=(blocks_x1, blocks_y1),
        block_dim=(TILE, TILE),
    )
    ctx.synchronize()

    # Layer 2: h1 -> h2 (ReLU)
    h1_immut = LayoutTensor[dtype, Layout.row_major(BATCH, HIDDEN_DIM), ImmutAnyOrigin](h1.unsafe_ptr())
    W2_t = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, HIDDEN_DIM), ImmutAnyOrigin](W2.unsafe_ptr())
    b2_t = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin](b2.unsafe_ptr())
    h2_t = LayoutTensor[dtype, Layout.row_major(BATCH, HIDDEN_DIM), MutAnyOrigin](h2.unsafe_ptr())

    comptime blocks_y2 = (BATCH + TILE - 1) // TILE
    comptime blocks_x2 = (HIDDEN_DIM + TILE - 1) // TILE
    ctx.enqueue_function_checked[
        linear_forward_relu_kernel[dtype, BATCH, HIDDEN_DIM, HIDDEN_DIM, TILE],
        linear_forward_relu_kernel[dtype, BATCH, HIDDEN_DIM, HIDDEN_DIM, TILE]
    ](
        h2_t, h1_immut, W2_t, b2_t,
        grid_dim=(blocks_x2, blocks_y2),
        block_dim=(TILE, TILE),
    )
    ctx.synchronize()

    # Layer 3: h2 -> action (tanh)
    h2_immut = LayoutTensor[dtype, Layout.row_major(BATCH, HIDDEN_DIM), ImmutAnyOrigin](h2.unsafe_ptr())
    W3_t = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, ACTION_DIM), ImmutAnyOrigin](W3.unsafe_ptr())
    b3_t = LayoutTensor[dtype, Layout.row_major(ACTION_DIM), ImmutAnyOrigin](b3.unsafe_ptr())
    action_t = LayoutTensor[dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin](action.unsafe_ptr())

    comptime blocks_y3 = (BATCH + TILE - 1) // TILE
    comptime blocks_x3 = (ACTION_DIM + TILE - 1) // TILE
    ctx.enqueue_function_checked[
        linear_forward_tanh_kernel[dtype, BATCH, HIDDEN_DIM, ACTION_DIM, TILE],
        linear_forward_tanh_kernel[dtype, BATCH, HIDDEN_DIM, ACTION_DIM, TILE]
    ](
        action_t, h2_immut, W3_t, b3_t,
        grid_dim=(blocks_x3, blocks_y3),
        block_dim=(TILE, TILE),
    )
    ctx.synchronize()

    # Verify: tanh output should be in [-1, 1]
    with action.map_to_host() as ah:
        for i in range(BATCH * ACTION_DIM):
            if ah[i] < -1 or ah[i] > 1:
                print("Action out of range at", i, ":", ah[i])
                raise Error("Actor forward: action out of range")
        print("  Actions in valid range [-1, 1]")
        print("  Sample action value:", ah[0])
    print("  Actor forward: PASSED")


fn test_critic_forward(ctx: DeviceContext) raises:
    """Test Critic forward pass: (obs, action) -> concat -> relu -> relu -> Q."""
    print("Testing Critic forward pass...")

    comptime BATCH = 4
    comptime OBS_DIM = 8
    comptime ACTION_DIM = 4
    comptime INPUT_DIM = OBS_DIM + ACTION_DIM
    comptime HIDDEN_DIM = 16
    comptime TILE = 4
    comptime TPB = 64

    # Allocate buffers
    obs = ctx.enqueue_create_buffer[dtype](BATCH * OBS_DIM)
    action = ctx.enqueue_create_buffer[dtype](BATCH * ACTION_DIM)
    concat = ctx.enqueue_create_buffer[dtype](BATCH * INPUT_DIM)
    W1 = ctx.enqueue_create_buffer[dtype](INPUT_DIM * HIDDEN_DIM)
    b1 = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
    h1 = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN_DIM)
    W2 = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * HIDDEN_DIM)
    b2 = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM)
    h2 = ctx.enqueue_create_buffer[dtype](BATCH * HIDDEN_DIM)
    W3 = ctx.enqueue_create_buffer[dtype](HIDDEN_DIM * 1)
    b3 = ctx.enqueue_create_buffer[dtype](1)
    q = ctx.enqueue_create_buffer[dtype](BATCH * 1)

    concat.enqueue_fill(0)
    h1.enqueue_fill(0)
    h2.enqueue_fill(0)
    q.enqueue_fill(0)

    # Initialize
    with obs.map_to_host() as obsh, action.map_to_host() as acth:
        for i in range(BATCH * OBS_DIM):
            obsh[i] = Float32(0.1)
        for i in range(BATCH * ACTION_DIM):
            acth[i] = Float32(0.5)

    with W1.map_to_host() as W1h, b1.map_to_host() as b1h:
        for i in range(INPUT_DIM * HIDDEN_DIM):
            W1h[i] = Float32(0.1)
        for i in range(HIDDEN_DIM):
            b1h[i] = Float32(0)

    with W2.map_to_host() as W2h, b2.map_to_host() as b2h:
        for i in range(HIDDEN_DIM * HIDDEN_DIM):
            W2h[i] = Float32(0.1)
        for i in range(HIDDEN_DIM):
            b2h[i] = Float32(0)

    with W3.map_to_host() as W3h, b3.map_to_host() as b3h:
        for i in range(HIDDEN_DIM):
            W3h[i] = Float32(0.1)
        b3h[0] = Float32(0)

    # Concatenate obs and action
    obs_t = LayoutTensor[dtype, Layout.row_major(BATCH, OBS_DIM), ImmutAnyOrigin](obs.unsafe_ptr())
    action_t = LayoutTensor[dtype, Layout.row_major(BATCH, ACTION_DIM), ImmutAnyOrigin](action.unsafe_ptr())
    concat_t = LayoutTensor[dtype, Layout.row_major(BATCH, INPUT_DIM), MutAnyOrigin](concat.unsafe_ptr())

    comptime num_blocks = (BATCH * INPUT_DIM + TPB - 1) // TPB
    ctx.enqueue_function_checked[
        concat_obs_action_kernel[dtype, BATCH, OBS_DIM, ACTION_DIM, TPB],
        concat_obs_action_kernel[dtype, BATCH, OBS_DIM, ACTION_DIM, TPB]
    ](
        concat_t, obs_t, action_t,
        grid_dim=(num_blocks,),
        block_dim=(TPB,),
    )
    ctx.synchronize()

    # Layer 1: concat -> h1 (ReLU)
    concat_immut = LayoutTensor[dtype, Layout.row_major(BATCH, INPUT_DIM), ImmutAnyOrigin](concat.unsafe_ptr())
    W1_t = LayoutTensor[dtype, Layout.row_major(INPUT_DIM, HIDDEN_DIM), ImmutAnyOrigin](W1.unsafe_ptr())
    b1_t = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin](b1.unsafe_ptr())
    h1_t = LayoutTensor[dtype, Layout.row_major(BATCH, HIDDEN_DIM), MutAnyOrigin](h1.unsafe_ptr())

    comptime blocks_y1 = (BATCH + TILE - 1) // TILE
    comptime blocks_x1 = (HIDDEN_DIM + TILE - 1) // TILE
    ctx.enqueue_function_checked[
        linear_forward_relu_kernel[dtype, BATCH, INPUT_DIM, HIDDEN_DIM, TILE],
        linear_forward_relu_kernel[dtype, BATCH, INPUT_DIM, HIDDEN_DIM, TILE]
    ](
        h1_t, concat_immut, W1_t, b1_t,
        grid_dim=(blocks_x1, blocks_y1),
        block_dim=(TILE, TILE),
    )
    ctx.synchronize()

    # Layer 2: h1 -> h2 (ReLU)
    h1_immut = LayoutTensor[dtype, Layout.row_major(BATCH, HIDDEN_DIM), ImmutAnyOrigin](h1.unsafe_ptr())
    W2_t = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, HIDDEN_DIM), ImmutAnyOrigin](W2.unsafe_ptr())
    b2_t = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM), ImmutAnyOrigin](b2.unsafe_ptr())
    h2_t = LayoutTensor[dtype, Layout.row_major(BATCH, HIDDEN_DIM), MutAnyOrigin](h2.unsafe_ptr())

    comptime blocks_y2 = (BATCH + TILE - 1) // TILE
    comptime blocks_x2 = (HIDDEN_DIM + TILE - 1) // TILE
    ctx.enqueue_function_checked[
        linear_forward_relu_kernel[dtype, BATCH, HIDDEN_DIM, HIDDEN_DIM, TILE],
        linear_forward_relu_kernel[dtype, BATCH, HIDDEN_DIM, HIDDEN_DIM, TILE]
    ](
        h2_t, h1_immut, W2_t, b2_t,
        grid_dim=(blocks_x2, blocks_y2),
        block_dim=(TILE, TILE),
    )
    ctx.synchronize()

    # Layer 3: h2 -> Q (no activation)
    h2_immut = LayoutTensor[dtype, Layout.row_major(BATCH, HIDDEN_DIM), ImmutAnyOrigin](h2.unsafe_ptr())
    W3_t = LayoutTensor[dtype, Layout.row_major(HIDDEN_DIM, 1), ImmutAnyOrigin](W3.unsafe_ptr())
    b3_t = LayoutTensor[dtype, Layout.row_major(1), ImmutAnyOrigin](b3.unsafe_ptr())
    q_t = LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin](q.unsafe_ptr())

    comptime blocks_y3 = (BATCH + TILE - 1) // TILE
    comptime blocks_x3 = (1 + TILE - 1) // TILE
    ctx.enqueue_function_checked[
        linear_forward_kernel[dtype, BATCH, HIDDEN_DIM, 1, TILE],
        linear_forward_kernel[dtype, BATCH, HIDDEN_DIM, 1, TILE]
    ](
        q_t, h2_immut, W3_t, b3_t,
        grid_dim=(blocks_x3, blocks_y3),
        block_dim=(TILE, TILE),
    )
    ctx.synchronize()

    with q.map_to_host() as qh:
        print("  Q-values:", qh[0], qh[1], qh[2], qh[3])
        # All Q-values should be equal (uniform init)
        var first = qh[0]
        for i in range(1, BATCH):
            if qh[i] != first:
                print("Q-values differ (unexpected with uniform init)")
                raise Error("Critic forward: Q-values differ")
    print("  Critic forward: PASSED")


fn test_stochastic_actor_kernels(ctx: DeviceContext) raises:
    """Test StochasticActor kernels for SAC."""
    print("Testing StochasticActor kernels...")

    comptime BATCH = 4
    comptime ACTION_DIM = 2
    comptime SIZE = BATCH * ACTION_DIM
    comptime TPB = 16

    # Test split_mean_log_std_kernel
    combined = ctx.enqueue_create_buffer[dtype](BATCH * ACTION_DIM * 2)
    mean = ctx.enqueue_create_buffer[dtype](SIZE)
    log_std = ctx.enqueue_create_buffer[dtype](SIZE)
    mean.enqueue_fill(0)
    log_std.enqueue_fill(0)

    with combined.map_to_host() as ch:
        for batch in range(BATCH):
            for i in range(ACTION_DIM):
                ch[batch * ACTION_DIM * 2 + i] = Float32(1)  # mean
            for i in range(ACTION_DIM):
                ch[batch * ACTION_DIM * 2 + ACTION_DIM + i] = Float32(-1)  # log_std

    combined_t = LayoutTensor[dtype, Layout.row_major(BATCH, ACTION_DIM * 2), ImmutAnyOrigin](combined.unsafe_ptr())
    mean_t = LayoutTensor[dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin](mean.unsafe_ptr())
    log_std_t = LayoutTensor[dtype, Layout.row_major(BATCH, ACTION_DIM), MutAnyOrigin](log_std.unsafe_ptr())

    comptime num_blocks = (SIZE + TPB - 1) // TPB
    ctx.enqueue_function_checked[
        split_mean_log_std_kernel[dtype, BATCH, ACTION_DIM, TPB],
        split_mean_log_std_kernel[dtype, BATCH, ACTION_DIM, TPB]
    ](
        mean_t, log_std_t, combined_t,
        grid_dim=(num_blocks,),
        block_dim=(TPB,),
    )
    ctx.synchronize()

    with mean.map_to_host() as mh, log_std.map_to_host() as lsh:
        for i in range(SIZE):
            if mh[i] != Float32(1):
                print("Mean mismatch at", i, ":", mh[i])
                raise Error("split_mean_log_std mismatch")
            if lsh[i] != Float32(-1):
                print("Log_std mismatch at", i, ":", lsh[i])
                raise Error("split_mean_log_std mismatch")
    print("  split_mean_log_std: PASSED")

    # Test clamp_log_std_kernel
    clamped = ctx.enqueue_create_buffer[dtype](SIZE)
    log_std_input = ctx.enqueue_create_buffer[dtype](SIZE)
    clamped.enqueue_fill(0)

    with log_std_input.map_to_host() as lsih:
        for i in range(SIZE):
            lsih[i] = Float32(i) - Float32(4)  # [-4, -3, -2, -1, 0, 1, 2, 3]

    clamped_t = LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin](clamped.unsafe_ptr())
    log_std_input_t = LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin](log_std_input.unsafe_ptr())

    ctx.enqueue_function_checked[
        clamp_log_std_kernel[dtype, SIZE, TPB],
        clamp_log_std_kernel[dtype, SIZE, TPB]
    ](
        clamped_t, log_std_input_t,
        Scalar[dtype](-2),  # log_std_min
        Scalar[dtype](2),   # log_std_max
        grid_dim=(num_blocks,),
        block_dim=(TPB,),
    )
    ctx.synchronize()

    with clamped.map_to_host() as ch:
        for i in range(SIZE):
            val = Float32(i) - Float32(4)
            expected_val = val
            if expected_val < Float32(-2):
                expected_val = Float32(-2)
            elif expected_val > Float32(2):
                expected_val = Float32(2)
            if ch[i] != expected_val:
                print("Clamp mismatch at", i, ":", ch[i], "vs", expected_val)
                raise Error("clamp_log_std mismatch")
    print("  clamp_log_std: PASSED")

    # Test squash_action_kernel
    squashed = ctx.enqueue_create_buffer[dtype](SIZE)
    pre_squash = ctx.enqueue_create_buffer[dtype](SIZE)
    squashed.enqueue_fill(0)

    with pre_squash.map_to_host() as psh:
        for i in range(SIZE):
            psh[i] = Float32(0.5)

    squashed_t = LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin](squashed.unsafe_ptr())
    pre_squash_t = LayoutTensor[dtype, Layout.row_major(SIZE), ImmutAnyOrigin](pre_squash.unsafe_ptr())

    ctx.enqueue_function_checked[
        squash_action_kernel[dtype, SIZE, TPB],
        squash_action_kernel[dtype, SIZE, TPB]
    ](
        squashed_t, pre_squash_t,
        Scalar[dtype](2),  # action_scale
        grid_dim=(num_blocks,),
        block_dim=(TPB,),
    )
    ctx.synchronize()

    expected_squash = math_tanh(Float32(0.5)) * Float32(2)
    with squashed.map_to_host() as sh:
        for i in range(SIZE):
            diff = abs(sh[i] - expected_squash)
            if diff > Float32(0.001):
                print("Squash mismatch at", i, ":", sh[i], "vs", expected_squash)
                raise Error("squash_action mismatch")
    print("  squash_action: PASSED (tanh(0.5) * 2.0 =", expected_squash, ")")


fn main() raises:
    print("=" * 60)
    print("GPU Actor-Critic Tests")
    print("=" * 60)
    print()

    with DeviceContext() as ctx:
        test_relu_grad(ctx)
        test_concat_obs_action(ctx)
        print()
        print("--- Actor Forward Pass ---")
        test_actor_forward(ctx)
        print()
        print("--- Critic Forward Pass ---")
        test_critic_forward(ctx)
        print()
        print("--- StochasticActor Kernels (for SAC) ---")
        test_stochastic_actor_kernels(ctx)

    print()
    print("=" * 60)
    print("All GPU Actor-Critic tests PASSED!")
    print("=" * 60)
