"""Minimal isolation test: LossGraph forward_gpu on Metal.

Tests whether Sequential[SplitApply[Gather, Slice], MSELoss].forward_gpu
crashes the Metal shader compiler. This is the exact graph used by
AutodiffQGradient in DQN GPU training.

We test each component individually, then the composed graph, to find
the minimal reproducer for the XPC_ERROR_CONNECTION_INTERRUPTED crash.

Run:
    pixi run -e apple mojo run -I . tests/deep_agents/test_lossgraph_metal.mojo
"""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.model import (
    Model,
    Sequential,
    Gather,
    Slice,
    MSELoss,
)
from mojo_rl.nn.autodiff.combinators import SplitApply


# Small batch/actions to minimize noise
comptime BATCH = 4
comptime ACTIONS = 2
comptime LOSS_IN = ACTIONS + 2  # [Q0, Q1, action_idx, target] = 4


fn fill_dummy_input(ctx: DeviceContext, buf: DeviceBuffer[dtype]) raises:
    """Fill with known values: Q=[1.0, 2.0], action=0, target=0.5 for each sample."""
    var host = List[Scalar[dtype]](capacity=BATCH * LOSS_IN)
    for b in range(BATCH):
        host.append(Scalar[dtype](1.0))   # Q[0]
        host.append(Scalar[dtype](2.0))   # Q[1]
        host.append(Scalar[dtype](0.0))   # action_idx = 0
        host.append(Scalar[dtype](0.5))   # target = 0.5
    ctx.enqueue_copy(buf, host.unsafe_ptr())
    ctx.synchronize()


fn test_gather_forward(ctx: DeviceContext) raises:
    """Test: Gather[ACTIONS].forward_gpu in isolation."""
    print("--- Test 1: Gather[ACTIONS].forward_gpu ---")

    comptime G = Gather[ACTIONS]
    # Gather input: [Q0, Q1, action_idx] = ACTIONS + 1 = 3
    comptime GIN = G.IN_DIM   # ACTIONS + 1
    comptime GOUT = G.OUT_DIM  # 1

    var in_buf = ctx.enqueue_create_buffer[dtype](BATCH * GIN)
    var out_buf = ctx.enqueue_create_buffer[dtype](BATCH * GOUT)
    var cache_buf = ctx.enqueue_create_buffer[dtype](max(1, BATCH * G.CACHE_SIZE))
    var params_buf = ctx.enqueue_create_buffer[dtype](max(1, G.PARAM_SIZE))

    # Fill input: [Q0=1.0, Q1=2.0, action_idx=0] for each sample
    var host_in = List[Scalar[dtype]](capacity=BATCH * GIN)
    for b in range(BATCH):
        host_in.append(Scalar[dtype](1.0))   # Q[0]
        host_in.append(Scalar[dtype](2.0))   # Q[1]
        host_in.append(Scalar[dtype](0.0))   # action_idx = 0
    ctx.enqueue_copy(in_buf, host_in.unsafe_ptr())
    ctx.synchronize()

    var in_t = LayoutTensor[dtype, Layout.row_major(BATCH, GIN), MutAnyOrigin](
        in_buf.unsafe_ptr()
    )
    var out_t = LayoutTensor[dtype, Layout.row_major(BATCH, GOUT), MutAnyOrigin](
        out_buf.unsafe_ptr()
    )
    var params_t = LayoutTensor[dtype, Layout.row_major(G.PARAM_SIZE), MutAnyOrigin](
        params_buf.unsafe_ptr()
    )
    var cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, G.CACHE_SIZE), MutAnyOrigin](
        cache_buf.unsafe_ptr()
    )

    # Use workspace
    var ws = ctx.enqueue_create_buffer[dtype](max(1, BATCH * G.WORKSPACE_SIZE_PER_SAMPLE))

    print("  Calling Gather.forward_gpu...")
    G.forward_gpu[BATCH](ctx, out_t, in_t, params_t, cache_t, ws)
    ctx.synchronize()
    print("  Gather.forward_gpu OK")

    # Read output
    var host_out = List[Scalar[dtype]](capacity=BATCH * GOUT)
    host_out.resize(BATCH * GOUT, Scalar[dtype](0))
    ctx.enqueue_copy(host_out.unsafe_ptr(), out_buf)
    ctx.synchronize()
    print("  Output[0] = " + String(host_out[0]) + " (expected 1.0)")
    print()


fn test_slice_forward(ctx: DeviceContext) raises:
    """Test: Slice[1,0,1].forward_gpu in isolation."""
    print("--- Test 2: Slice[1,0,1].forward_gpu ---")

    comptime S = Slice[1, 0, 1]
    comptime SIN = S.IN_DIM   # 1
    comptime SOUT = S.OUT_DIM  # 1

    var in_buf = ctx.enqueue_create_buffer[dtype](BATCH * SIN)
    var out_buf = ctx.enqueue_create_buffer[dtype](BATCH * SOUT)
    var cache_buf = ctx.enqueue_create_buffer[dtype](max(1, BATCH * S.CACHE_SIZE))
    var params_buf = ctx.enqueue_create_buffer[dtype](max(1, S.PARAM_SIZE))

    # Fill input: [0.5] for each sample (target value)
    var host_in = List[Scalar[dtype]](capacity=BATCH * SIN)
    for b in range(BATCH):
        host_in.append(Scalar[dtype](0.5))
    ctx.enqueue_copy(in_buf, host_in.unsafe_ptr())
    ctx.synchronize()

    var in_t = LayoutTensor[dtype, Layout.row_major(BATCH, SIN), MutAnyOrigin](
        in_buf.unsafe_ptr()
    )
    var out_t = LayoutTensor[dtype, Layout.row_major(BATCH, SOUT), MutAnyOrigin](
        out_buf.unsafe_ptr()
    )
    var params_t = LayoutTensor[dtype, Layout.row_major(S.PARAM_SIZE), MutAnyOrigin](
        params_buf.unsafe_ptr()
    )
    var cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, S.CACHE_SIZE), MutAnyOrigin](
        cache_buf.unsafe_ptr()
    )

    var ws = ctx.enqueue_create_buffer[dtype](max(1, BATCH * S.WORKSPACE_SIZE_PER_SAMPLE))

    print("  Calling Slice.forward_gpu...")
    S.forward_gpu[BATCH](ctx, out_t, in_t, params_t, cache_t, ws)
    ctx.synchronize()
    print("  Slice.forward_gpu OK")

    var host_out = List[Scalar[dtype]](capacity=BATCH * SOUT)
    host_out.resize(BATCH * SOUT, Scalar[dtype](0))
    ctx.enqueue_copy(host_out.unsafe_ptr(), out_buf)
    ctx.synchronize()
    print("  Output[0] = " + String(host_out[0]) + " (expected 0.5)")
    print()


fn test_mse_forward(ctx: DeviceContext) raises:
    """Test: MSELoss.forward_gpu in isolation."""
    print("--- Test 3: MSELoss.forward_gpu ---")

    comptime M = MSELoss
    comptime MIN = M.IN_DIM   # 2 (prediction, target)
    comptime MOUT = M.OUT_DIM  # 1

    var in_buf = ctx.enqueue_create_buffer[dtype](BATCH * MIN)
    var out_buf = ctx.enqueue_create_buffer[dtype](BATCH * MOUT)
    var cache_buf = ctx.enqueue_create_buffer[dtype](max(1, BATCH * M.CACHE_SIZE))
    var params_buf = ctx.enqueue_create_buffer[dtype](max(1, M.PARAM_SIZE))

    # Fill input: [prediction=1.0, target=0.5] -> MSE = (1.0-0.5)^2 = 0.25
    var host_in = List[Scalar[dtype]](capacity=BATCH * MIN)
    for b in range(BATCH):
        host_in.append(Scalar[dtype](1.0))   # prediction
        host_in.append(Scalar[dtype](0.5))   # target
    ctx.enqueue_copy(in_buf, host_in.unsafe_ptr())
    ctx.synchronize()

    var in_t = LayoutTensor[dtype, Layout.row_major(BATCH, MIN), MutAnyOrigin](
        in_buf.unsafe_ptr()
    )
    var out_t = LayoutTensor[dtype, Layout.row_major(BATCH, MOUT), MutAnyOrigin](
        out_buf.unsafe_ptr()
    )
    var params_t = LayoutTensor[dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin](
        params_buf.unsafe_ptr()
    )
    var cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin](
        cache_buf.unsafe_ptr()
    )

    var ws = ctx.enqueue_create_buffer[dtype](max(1, BATCH * M.WORKSPACE_SIZE_PER_SAMPLE))

    print("  Calling MSELoss.forward_gpu...")
    M.forward_gpu[BATCH](ctx, out_t, in_t, params_t, cache_t, ws)
    ctx.synchronize()
    print("  MSELoss.forward_gpu OK")

    var host_out = List[Scalar[dtype]](capacity=BATCH * MOUT)
    host_out.resize(BATCH * MOUT, Scalar[dtype](0))
    ctx.enqueue_copy(host_out.unsafe_ptr(), out_buf)
    ctx.synchronize()
    print("  Output[0] = " + String(host_out[0]) + " (expected 0.25)")
    print()


fn test_splitapply_forward(ctx: DeviceContext) raises:
    """Test: SplitApply[Gather[2], Slice[1,0,1], 3].forward_gpu."""
    print("--- Test 4: SplitApply[Gather[2], Slice[1,0,1], 3].forward_gpu ---")

    comptime SA = SplitApply[Gather[ACTIONS], Slice[1, 0, 1], ACTIONS + 1]
    comptime SAIN = SA.IN_DIM   # ACTIONS + 2 = 4
    comptime SAOUT = SA.OUT_DIM  # 2 (gathered Q + sliced target)

    var in_buf = ctx.enqueue_create_buffer[dtype](BATCH * SAIN)
    var out_buf = ctx.enqueue_create_buffer[dtype](BATCH * SAOUT)
    var cache_buf = ctx.enqueue_create_buffer[dtype](max(1, BATCH * SA.CACHE_SIZE))
    var params_buf = ctx.enqueue_create_buffer[dtype](max(1, SA.PARAM_SIZE))

    fill_dummy_input(ctx, in_buf)

    var in_t = LayoutTensor[dtype, Layout.row_major(BATCH, SAIN), MutAnyOrigin](
        in_buf.unsafe_ptr()
    )
    var out_t = LayoutTensor[dtype, Layout.row_major(BATCH, SAOUT), MutAnyOrigin](
        out_buf.unsafe_ptr()
    )
    var params_t = LayoutTensor[dtype, Layout.row_major(SA.PARAM_SIZE), MutAnyOrigin](
        params_buf.unsafe_ptr()
    )
    var cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, SA.CACHE_SIZE), MutAnyOrigin](
        cache_buf.unsafe_ptr()
    )

    var ws = ctx.enqueue_create_buffer[dtype](max(1, BATCH * SA.WORKSPACE_SIZE_PER_SAMPLE))

    print("  Calling SplitApply.forward_gpu...")
    SA.forward_gpu[BATCH](ctx, out_t, in_t, params_t, cache_t, ws)
    ctx.synchronize()
    print("  SplitApply.forward_gpu OK")

    var host_out = List[Scalar[dtype]](capacity=BATCH * SAOUT)
    host_out.resize(BATCH * SAOUT, Scalar[dtype](0))
    ctx.enqueue_copy(host_out.unsafe_ptr(), out_buf)
    ctx.synchronize()
    print(
        "  Output[0,0] = "
        + String(host_out[0])
        + " (expected 1.0 = Q[action=0])"
    )
    print(
        "  Output[0,1] = "
        + String(host_out[1])
        + " (expected 0.5 = target)"
    )
    print()


fn test_full_lossgraph_forward(ctx: DeviceContext) raises:
    """Test: Full LossGraph = Sequential[SplitApply[...], MSELoss].forward_gpu.

    This is the exact graph that crashes in AutodiffQGradient.
    """
    print("--- Test 5: Full LossGraph (Sequential[SplitApply, MSELoss]).forward_gpu ---")

    comptime LossGraph = Sequential[
        SplitApply[Gather[ACTIONS], Slice[1, 0, 1], ACTIONS + 1],
        MSELoss,
    ]
    comptime LIN = LossGraph.IN_DIM    # ACTIONS + 2 = 4
    comptime LOUT = LossGraph.OUT_DIM   # 1
    comptime LCS = LossGraph.CACHE_SIZE

    var in_buf = ctx.enqueue_create_buffer[dtype](BATCH * LIN)
    var out_buf = ctx.enqueue_create_buffer[dtype](BATCH * LOUT)
    var cache_buf = ctx.enqueue_create_buffer[dtype](max(1, BATCH * LCS))
    var params_buf = ctx.enqueue_create_buffer[dtype](max(1, LossGraph.PARAM_SIZE))

    fill_dummy_input(ctx, in_buf)

    var in_t = LayoutTensor[dtype, Layout.row_major(BATCH, LIN), MutAnyOrigin](
        in_buf.unsafe_ptr()
    )
    var out_t = LayoutTensor[dtype, Layout.row_major(BATCH, LOUT), MutAnyOrigin](
        out_buf.unsafe_ptr()
    )
    var params_t = LayoutTensor[dtype, Layout.row_major(LossGraph.PARAM_SIZE), MutAnyOrigin](
        params_buf.unsafe_ptr()
    )
    var cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, LCS), MutAnyOrigin](
        cache_buf.unsafe_ptr()
    )

    var ws = ctx.enqueue_create_buffer[dtype](
        max(1, BATCH * LossGraph.WORKSPACE_SIZE_PER_SAMPLE)
    )

    print("  Calling LossGraph.forward_gpu...")
    LossGraph.forward_gpu[BATCH](ctx, out_t, in_t, params_t, cache_t, ws)
    ctx.synchronize()
    print("  LossGraph.forward_gpu OK")

    var host_out = List[Scalar[dtype]](capacity=BATCH * LOUT)
    host_out.resize(BATCH * LOUT, Scalar[dtype](0))
    ctx.enqueue_copy(host_out.unsafe_ptr(), out_buf)
    ctx.synchronize()
    print(
        "  Output[0] = "
        + String(host_out[0])
        + " (expected 0.25 = (1.0-0.5)^2)"
    )
    print()


fn test_full_lossgraph_backward(ctx: DeviceContext) raises:
    """Test: Full LossGraph backward_gpu (if forward passed)."""
    print("--- Test 6: Full LossGraph backward_gpu ---")

    comptime LossGraph = Sequential[
        SplitApply[Gather[ACTIONS], Slice[1, 0, 1], ACTIONS + 1],
        MSELoss,
    ]
    comptime LIN = LossGraph.IN_DIM
    comptime LOUT = LossGraph.OUT_DIM
    comptime LCS = LossGraph.CACHE_SIZE
    comptime PS = max(1, LossGraph.PARAM_SIZE)

    var in_buf = ctx.enqueue_create_buffer[dtype](BATCH * LIN)
    var out_buf = ctx.enqueue_create_buffer[dtype](BATCH * LOUT)
    var cache_buf = ctx.enqueue_create_buffer[dtype](max(1, BATCH * LCS))
    var params_buf = ctx.enqueue_create_buffer[dtype](PS)
    var grad_in_buf = ctx.enqueue_create_buffer[dtype](BATCH * LIN)
    var grad_seed_buf = ctx.enqueue_create_buffer[dtype](BATCH * LOUT)
    var grads_buf = ctx.enqueue_create_buffer[dtype](PS)

    fill_dummy_input(ctx, in_buf)

    var in_t = LayoutTensor[dtype, Layout.row_major(BATCH, LIN), MutAnyOrigin](
        in_buf.unsafe_ptr()
    )
    var out_t = LayoutTensor[dtype, Layout.row_major(BATCH, LOUT), MutAnyOrigin](
        out_buf.unsafe_ptr()
    )
    var params_t = LayoutTensor[dtype, Layout.row_major(LossGraph.PARAM_SIZE), MutAnyOrigin](
        params_buf.unsafe_ptr()
    )
    var cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, LCS), MutAnyOrigin](
        cache_buf.unsafe_ptr()
    )

    var ws = ctx.enqueue_create_buffer[dtype](
        max(1, BATCH * LossGraph.WORKSPACE_SIZE_PER_SAMPLE)
    )

    # Forward first (populates cache)
    print("  Forward pass (for cache)...")
    LossGraph.forward_gpu[BATCH](ctx, out_t, in_t, params_t, cache_t, ws)
    ctx.synchronize()
    print("  Forward OK")

    # Fill grad seed = 1/BATCH
    var host_seed = List[Scalar[dtype]](capacity=BATCH * LOUT)
    for b in range(BATCH):
        host_seed.append(Scalar[dtype](1.0 / Float64(BATCH)))
    ctx.enqueue_copy(grad_seed_buf, host_seed.unsafe_ptr())
    ctx.synchronize()

    var grad_in_t = LayoutTensor[dtype, Layout.row_major(BATCH, LIN), MutAnyOrigin](
        grad_in_buf.unsafe_ptr()
    )
    var grad_seed_t = LayoutTensor[dtype, Layout.row_major(BATCH, LOUT), MutAnyOrigin](
        grad_seed_buf.unsafe_ptr()
    )
    var grads_t = LayoutTensor[dtype, Layout.row_major(LossGraph.PARAM_SIZE), MutAnyOrigin](
        grads_buf.unsafe_ptr()
    )

    print("  Calling LossGraph.backward_gpu...")
    LossGraph.backward_gpu[BATCH](
        ctx, grad_in_t, grad_seed_t, params_t, cache_t, grads_t, ws
    )
    ctx.synchronize()
    print("  LossGraph.backward_gpu OK")

    # Read grad_in
    var host_grad = List[Scalar[dtype]](capacity=BATCH * LIN)
    host_grad.resize(BATCH * LIN, Scalar[dtype](0))
    ctx.enqueue_copy(host_grad.unsafe_ptr(), grad_in_buf)
    ctx.synchronize()
    print(
        "  grad_in[0] = ["
        + String(host_grad[0])
        + ", "
        + String(host_grad[1])
        + ", "
        + String(host_grad[2])
        + ", "
        + String(host_grad[3])
        + "]"
    )
    print("  (expected: [0.25, 0.0, 0.0, ...] — sparse MSE grad at action=0)")
    print()


fn main() raises:
    print("=" * 60)
    print("LossGraph Metal Isolation Test")
    print("=" * 60)
    print()
    print(
        "Testing: Sequential[SplitApply[Gather["
        + String(ACTIONS)
        + "], Slice[1,0,1], "
        + String(ACTIONS + 1)
        + "], MSELoss]"
    )
    print("BATCH=" + String(BATCH) + ", ACTIONS=" + String(ACTIONS))
    print()

    with DeviceContext() as ctx:
        # Test components bottom-up: simplest first
        test_gather_forward(ctx)
        test_slice_forward(ctx)
        test_mse_forward(ctx)
        test_splitapply_forward(ctx)

        # The suspected crash point
        test_full_lossgraph_forward(ctx)

        # If forward passes, test backward too
        test_full_lossgraph_backward(ctx)

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
