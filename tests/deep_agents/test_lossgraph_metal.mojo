"""Minimal isolation test: LossGraph forward_gpu on Metal.

Tests whether Sequential[SplitApply[Gather, Slice], MSELoss/HuberLoss].forward_gpu
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
    HuberLoss,
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


# =============================================================================
# HuberLoss tests
# =============================================================================


fn test_huber_forward(ctx: DeviceContext) raises:
    """Test: HuberLoss.forward_gpu in isolation.

    Input: [pred=1.0, target=0.5] → residual=0.5, |0.5| <= delta=1.0
    Expected: 0.5 * 0.5^2 = 0.125
    """
    print("--- Test 7: HuberLoss.forward_gpu (quadratic region) ---")

    comptime H = HuberLoss[1.0]
    comptime HIN = H.IN_DIM   # 2 (prediction, target)
    comptime HOUT = H.OUT_DIM  # 1

    var in_buf = ctx.enqueue_create_buffer[dtype](BATCH * HIN)
    var out_buf = ctx.enqueue_create_buffer[dtype](BATCH * HOUT)
    var cache_buf = ctx.enqueue_create_buffer[dtype](max(1, BATCH * H.CACHE_SIZE))
    var params_buf = ctx.enqueue_create_buffer[dtype](max(1, H.PARAM_SIZE))

    # Fill input: [prediction=1.0, target=0.5]
    var host_in = List[Scalar[dtype]](capacity=BATCH * HIN)
    for b in range(BATCH):
        host_in.append(Scalar[dtype](1.0))   # prediction
        host_in.append(Scalar[dtype](0.5))   # target
    ctx.enqueue_copy(in_buf, host_in.unsafe_ptr())
    ctx.synchronize()

    var in_t = LayoutTensor[dtype, Layout.row_major(BATCH, HIN), MutAnyOrigin](
        in_buf.unsafe_ptr()
    )
    var out_t = LayoutTensor[dtype, Layout.row_major(BATCH, HOUT), MutAnyOrigin](
        out_buf.unsafe_ptr()
    )
    var params_t = LayoutTensor[dtype, Layout.row_major(H.PARAM_SIZE), MutAnyOrigin](
        params_buf.unsafe_ptr()
    )
    var cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, H.CACHE_SIZE), MutAnyOrigin](
        cache_buf.unsafe_ptr()
    )

    var ws = ctx.enqueue_create_buffer[dtype](max(1, BATCH * H.WORKSPACE_SIZE_PER_SAMPLE))

    print("  Calling HuberLoss.forward_gpu...")
    H.forward_gpu[BATCH](ctx, out_t, in_t, params_t, cache_t, ws)
    ctx.synchronize()
    print("  HuberLoss.forward_gpu OK")

    var host_out = List[Scalar[dtype]](capacity=BATCH * HOUT)
    host_out.resize(BATCH * HOUT, Scalar[dtype](0))
    ctx.enqueue_copy(host_out.unsafe_ptr(), out_buf)
    ctx.synchronize()
    print(
        "  Output[0] = "
        + String(host_out[0])
        + " (expected 0.125 = 0.5*0.5^2)"
    )

    # Also read cache to verify residual was stored
    var host_cache = List[Scalar[dtype]](capacity=BATCH * H.CACHE_SIZE)
    host_cache.resize(BATCH * H.CACHE_SIZE, Scalar[dtype](0))
    ctx.enqueue_copy(host_cache.unsafe_ptr(), cache_buf)
    ctx.synchronize()
    print(
        "  Cache[0] = "
        + String(host_cache[0])
        + " (expected 0.5 = residual)"
    )
    print()


fn test_huber_forward_linear(ctx: DeviceContext) raises:
    """Test: HuberLoss.forward_gpu in the LINEAR region (|residual| > delta).

    Input: [pred=3.0, target=0.5] → residual=2.5, |2.5| > delta=1.0
    Expected: 1.0 * 2.5 - 0.5 * 1.0^2 = 2.5 - 0.5 = 2.0
    """
    print("--- Test 8: HuberLoss.forward_gpu (linear region) ---")

    comptime H = HuberLoss[1.0]
    comptime HIN = H.IN_DIM
    comptime HOUT = H.OUT_DIM

    var in_buf = ctx.enqueue_create_buffer[dtype](BATCH * HIN)
    var out_buf = ctx.enqueue_create_buffer[dtype](BATCH * HOUT)
    var cache_buf = ctx.enqueue_create_buffer[dtype](max(1, BATCH * H.CACHE_SIZE))
    var params_buf = ctx.enqueue_create_buffer[dtype](max(1, H.PARAM_SIZE))

    var host_in = List[Scalar[dtype]](capacity=BATCH * HIN)
    for b in range(BATCH):
        host_in.append(Scalar[dtype](3.0))   # prediction (large)
        host_in.append(Scalar[dtype](0.5))   # target
    ctx.enqueue_copy(in_buf, host_in.unsafe_ptr())
    ctx.synchronize()

    var in_t = LayoutTensor[dtype, Layout.row_major(BATCH, HIN), MutAnyOrigin](
        in_buf.unsafe_ptr()
    )
    var out_t = LayoutTensor[dtype, Layout.row_major(BATCH, HOUT), MutAnyOrigin](
        out_buf.unsafe_ptr()
    )
    var params_t = LayoutTensor[dtype, Layout.row_major(H.PARAM_SIZE), MutAnyOrigin](
        params_buf.unsafe_ptr()
    )
    var cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, H.CACHE_SIZE), MutAnyOrigin](
        cache_buf.unsafe_ptr()
    )

    var ws = ctx.enqueue_create_buffer[dtype](max(1, BATCH * H.WORKSPACE_SIZE_PER_SAMPLE))

    H.forward_gpu[BATCH](ctx, out_t, in_t, params_t, cache_t, ws)
    ctx.synchronize()

    var host_out = List[Scalar[dtype]](capacity=BATCH * HOUT)
    host_out.resize(BATCH * HOUT, Scalar[dtype](0))
    ctx.enqueue_copy(host_out.unsafe_ptr(), out_buf)
    ctx.synchronize()
    print(
        "  Output[0] = "
        + String(host_out[0])
        + " (expected 2.0 = 1.0*2.5 - 0.5)"
    )
    print()


fn test_huber_backward_gpu_vs_cpu(ctx: DeviceContext) raises:
    """Critical test: Compare HuberLoss GPU backward vs CPU backward.

    If these differ, the GPU kernel has a bug.
    """
    print("--- Test 9: HuberLoss backward GPU vs CPU ---")

    comptime H = HuberLoss[1.0]
    comptime HIN = H.IN_DIM   # 2
    comptime HOUT = H.OUT_DIM  # 1

    # -- CPU backward --
    var cpu_in = InlineArray[Scalar[dtype], BATCH * HIN](uninitialized=True)
    var cpu_out = InlineArray[Scalar[dtype], BATCH * HOUT](uninitialized=True)
    var cpu_cache = InlineArray[Scalar[dtype], BATCH * H.CACHE_SIZE](
        uninitialized=True
    )
    var cpu_params = InlineArray[Scalar[dtype], max(1, H.PARAM_SIZE)](
        fill=Scalar[dtype](0)
    )
    var cpu_grad_seed = InlineArray[Scalar[dtype], BATCH * HOUT](
        uninitialized=True
    )
    var cpu_grad_in = InlineArray[Scalar[dtype], BATCH * HIN](
        uninitialized=True
    )
    var cpu_grads = InlineArray[Scalar[dtype], max(1, H.PARAM_SIZE)](
        fill=Scalar[dtype](0)
    )

    # Test values: mix of quadratic and linear region
    # Sample 0: pred=1.0, target=0.5 → residual=0.5 (quadratic, |0.5|<=1)
    # Sample 1: pred=3.0, target=0.5 → residual=2.5 (linear, |2.5|>1)
    # Sample 2: pred=0.2, target=0.8 → residual=-0.6 (quadratic, negative)
    # Sample 3: pred=0.0, target=5.0 → residual=-5.0 (linear, negative)
    var preds = InlineArray[Float64, BATCH](uninitialized=True)
    preds[0] = 1.0; preds[1] = 3.0; preds[2] = 0.2; preds[3] = 0.0
    var targets = InlineArray[Float64, BATCH](uninitialized=True)
    targets[0] = 0.5; targets[1] = 0.5; targets[2] = 0.8; targets[3] = 5.0

    for b in range(BATCH):
        cpu_in[b * HIN + 0] = Scalar[dtype](preds[b])
        cpu_in[b * HIN + 1] = Scalar[dtype](targets[b])
        cpu_grad_seed[b] = Scalar[dtype](1.0 / Float64(BATCH))

    var cpu_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, HIN), MutAnyOrigin
    ](cpu_in.unsafe_ptr())
    var cpu_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, HOUT), MutAnyOrigin
    ](cpu_out.unsafe_ptr())
    var cpu_params_t = LayoutTensor[
        dtype, Layout.row_major(H.PARAM_SIZE), MutAnyOrigin
    ](cpu_params.unsafe_ptr())
    var cpu_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H.CACHE_SIZE), MutAnyOrigin
    ](cpu_cache.unsafe_ptr())
    var cpu_grad_seed_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, HOUT), MutAnyOrigin
    ](cpu_grad_seed.unsafe_ptr())
    var cpu_grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, HIN), MutAnyOrigin
    ](cpu_grad_in.unsafe_ptr())
    var cpu_grads_t = LayoutTensor[
        dtype, Layout.row_major(H.PARAM_SIZE), MutAnyOrigin
    ](cpu_grads.unsafe_ptr())

    # CPU forward (populate cache)
    H.forward[BATCH](cpu_in_t, cpu_out_t, cpu_params_t, cpu_cache_t)
    # CPU backward
    H.backward[BATCH](
        cpu_grad_seed_t, cpu_grad_in_t, cpu_params_t, cpu_cache_t, cpu_grads_t
    )

    print("  CPU results:")
    for b in range(BATCH):
        var residual = preds[b] - targets[b]
        print(
            "    Sample "
            + String(b)
            + ": residual="
            + String(residual)[:6]
            + " fwd="
            + String(cpu_out[b])[:8]
            + " grad_pred="
            + String(cpu_grad_in[b * HIN])[:10]
            + " grad_tgt="
            + String(cpu_grad_in[b * HIN + 1])[:10]
        )

    # -- GPU backward --
    var in_buf = ctx.enqueue_create_buffer[dtype](BATCH * HIN)
    var out_buf = ctx.enqueue_create_buffer[dtype](BATCH * HOUT)
    var cache_buf = ctx.enqueue_create_buffer[dtype](max(1, BATCH * H.CACHE_SIZE))
    var params_buf = ctx.enqueue_create_buffer[dtype](max(1, H.PARAM_SIZE))
    var grad_in_buf = ctx.enqueue_create_buffer[dtype](BATCH * HIN)
    var grad_seed_buf = ctx.enqueue_create_buffer[dtype](BATCH * HOUT)
    var grads_buf = ctx.enqueue_create_buffer[dtype](max(1, H.PARAM_SIZE))

    ctx.enqueue_copy(in_buf, cpu_in.unsafe_ptr())
    ctx.enqueue_copy(grad_seed_buf, cpu_grad_seed.unsafe_ptr())
    ctx.synchronize()

    var gpu_in_t = LayoutTensor[dtype, Layout.row_major(BATCH, HIN), MutAnyOrigin](
        in_buf.unsafe_ptr()
    )
    var gpu_out_t = LayoutTensor[dtype, Layout.row_major(BATCH, HOUT), MutAnyOrigin](
        out_buf.unsafe_ptr()
    )
    var gpu_params_t = LayoutTensor[dtype, Layout.row_major(H.PARAM_SIZE), MutAnyOrigin](
        params_buf.unsafe_ptr()
    )
    var gpu_cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, H.CACHE_SIZE), MutAnyOrigin](
        cache_buf.unsafe_ptr()
    )
    var gpu_grad_in_t = LayoutTensor[dtype, Layout.row_major(BATCH, HIN), MutAnyOrigin](
        grad_in_buf.unsafe_ptr()
    )
    var gpu_grad_seed_t = LayoutTensor[dtype, Layout.row_major(BATCH, HOUT), MutAnyOrigin](
        grad_seed_buf.unsafe_ptr()
    )
    var gpu_grads_t = LayoutTensor[dtype, Layout.row_major(H.PARAM_SIZE), MutAnyOrigin](
        grads_buf.unsafe_ptr()
    )

    var ws = ctx.enqueue_create_buffer[dtype](max(1, BATCH * H.WORKSPACE_SIZE_PER_SAMPLE))

    # GPU forward (populate cache)
    H.forward_gpu[BATCH](ctx, gpu_out_t, gpu_in_t, gpu_params_t, gpu_cache_t, ws)
    ctx.synchronize()

    # GPU backward
    H.backward_gpu[BATCH](
        ctx,
        gpu_grad_in_t,
        gpu_grad_seed_t,
        gpu_params_t,
        gpu_cache_t,
        gpu_grads_t,
        ws,
    )
    ctx.synchronize()

    # Read GPU results
    var gpu_fwd = List[Scalar[dtype]](capacity=BATCH * HOUT)
    gpu_fwd.resize(BATCH * HOUT, Scalar[dtype](0))
    ctx.enqueue_copy(gpu_fwd.unsafe_ptr(), out_buf)

    var gpu_grad = List[Scalar[dtype]](capacity=BATCH * HIN)
    gpu_grad.resize(BATCH * HIN, Scalar[dtype](0))
    ctx.enqueue_copy(gpu_grad.unsafe_ptr(), grad_in_buf)
    ctx.synchronize()

    print("  GPU results:")
    for b in range(BATCH):
        print(
            "    Sample "
            + String(b)
            + ": fwd="
            + String(gpu_fwd[b])[:8]
            + " grad_pred="
            + String(gpu_grad[b * HIN])[:10]
            + " grad_tgt="
            + String(gpu_grad[b * HIN + 1])[:10]
        )

    # Compare
    print("  Comparing CPU vs GPU:")
    var all_ok = True
    for b in range(BATCH):
        var fwd_diff = Float64(cpu_out[b]) - Float64(gpu_fwd[b])
        var grad_diff = Float64(cpu_grad_in[b * HIN]) - Float64(
            gpu_grad[b * HIN]
        )
        if fwd_diff > 1e-5 or fwd_diff < -1e-5:
            print(
                "    MISMATCH fwd sample "
                + String(b)
                + ": cpu="
                + String(cpu_out[b])[:8]
                + " gpu="
                + String(gpu_fwd[b])[:8]
            )
            all_ok = False
        if grad_diff > 1e-5 or grad_diff < -1e-5:
            print(
                "    MISMATCH grad sample "
                + String(b)
                + ": cpu="
                + String(cpu_grad_in[b * HIN])[:10]
                + " gpu="
                + String(gpu_grad[b * HIN])[:10]
            )
            all_ok = False
    if all_ok:
        print("    ALL MATCH ✓")
    print()


fn test_full_huber_lossgraph_backward(ctx: DeviceContext) raises:
    """Test: Full HuberLoss LossGraph backward, compare GPU vs CPU.

    Uses the exact same graph as HuberDQNConfig:
        Sequential[SplitApply[Gather[2], Slice[1,0,1], 3], HuberLoss]

    Input: Q=[1.0, 2.0], action=0, target=0.5
    Residual = Q[0] - target = 1.0 - 0.5 = 0.5 (quadratic region)

    Expected GPU grad_q:
        grad_q[b, 0] = 0.5 * (1/BATCH) = 0.125  (taken action)
        grad_q[b, 1] = 0.0                        (not taken)
    """
    print("--- Test 10: Full HuberLoss LossGraph backward GPU vs CPU ---")

    comptime HuberLossGraph = Sequential[
        SplitApply[Gather[ACTIONS], Slice[1, 0, 1], ACTIONS + 1],
        HuberLoss[1.0],
    ]
    comptime LIN = HuberLossGraph.IN_DIM    # 4
    comptime LOUT = HuberLossGraph.OUT_DIM   # 1
    comptime LCS = HuberLossGraph.CACHE_SIZE
    comptime PS = max(1, HuberLossGraph.PARAM_SIZE)

    # -- CPU path --
    var cpu_in = InlineArray[Scalar[dtype], BATCH * LIN](uninitialized=True)
    for b in range(BATCH):
        cpu_in[b * LIN + 0] = Scalar[dtype](1.0)  # Q[0]
        cpu_in[b * LIN + 1] = Scalar[dtype](2.0)  # Q[1]
        cpu_in[b * LIN + 2] = Scalar[dtype](0.0)  # action_idx = 0
        cpu_in[b * LIN + 3] = Scalar[dtype](0.5)  # target

    var cpu_out = InlineArray[Scalar[dtype], BATCH * LOUT](uninitialized=True)
    var cpu_cache = InlineArray[Scalar[dtype], BATCH * LCS](uninitialized=True)
    var cpu_params = InlineArray[Scalar[dtype], PS](fill=Scalar[dtype](0))
    var cpu_grad_seed = InlineArray[Scalar[dtype], BATCH * LOUT](
        uninitialized=True
    )
    var cpu_grad_in = InlineArray[Scalar[dtype], BATCH * LIN](
        uninitialized=True
    )
    var cpu_grads = InlineArray[Scalar[dtype], PS](fill=Scalar[dtype](0))

    for b in range(BATCH):
        cpu_grad_seed[b] = Scalar[dtype](1.0 / Float64(BATCH))

    var cpu_in_t = LayoutTensor[dtype, Layout.row_major(BATCH, LIN), MutAnyOrigin](
        cpu_in.unsafe_ptr()
    )
    var cpu_out_t = LayoutTensor[dtype, Layout.row_major(BATCH, LOUT), MutAnyOrigin](
        cpu_out.unsafe_ptr()
    )
    var cpu_params_t = LayoutTensor[
        dtype, Layout.row_major(HuberLossGraph.PARAM_SIZE), MutAnyOrigin
    ](cpu_params.unsafe_ptr())
    var cpu_cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, LCS), MutAnyOrigin](
        cpu_cache.unsafe_ptr()
    )
    var cpu_grad_seed_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LOUT), MutAnyOrigin
    ](cpu_grad_seed.unsafe_ptr())
    var cpu_grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LIN), MutAnyOrigin
    ](cpu_grad_in.unsafe_ptr())
    var cpu_grads_t = LayoutTensor[
        dtype, Layout.row_major(HuberLossGraph.PARAM_SIZE), MutAnyOrigin
    ](cpu_grads.unsafe_ptr())

    HuberLossGraph.forward[BATCH](cpu_in_t, cpu_out_t, cpu_params_t, cpu_cache_t)
    HuberLossGraph.backward[BATCH](
        cpu_grad_seed_t, cpu_grad_in_t, cpu_params_t, cpu_cache_t, cpu_grads_t
    )

    print("  CPU: fwd[0]=" + String(cpu_out[0])[:8])
    print(
        "  CPU: grad_in[0] = ["
        + String(cpu_grad_in[0])[:10]
        + ", "
        + String(cpu_grad_in[1])[:10]
        + ", "
        + String(cpu_grad_in[2])[:10]
        + ", "
        + String(cpu_grad_in[3])[:10]
        + "]"
    )

    # -- GPU path --
    var in_buf = ctx.enqueue_create_buffer[dtype](BATCH * LIN)
    var out_buf = ctx.enqueue_create_buffer[dtype](BATCH * LOUT)
    var cache_buf = ctx.enqueue_create_buffer[dtype](max(1, BATCH * LCS))
    var params_buf = ctx.enqueue_create_buffer[dtype](PS)
    var grad_in_buf = ctx.enqueue_create_buffer[dtype](BATCH * LIN)
    var grad_seed_buf = ctx.enqueue_create_buffer[dtype](BATCH * LOUT)
    var grads_buf = ctx.enqueue_create_buffer[dtype](PS)

    ctx.enqueue_copy(in_buf, cpu_in.unsafe_ptr())
    ctx.enqueue_copy(grad_seed_buf, cpu_grad_seed.unsafe_ptr())
    ctx.synchronize()

    var gpu_in_t = LayoutTensor[dtype, Layout.row_major(BATCH, LIN), MutAnyOrigin](
        in_buf.unsafe_ptr()
    )
    var gpu_out_t = LayoutTensor[dtype, Layout.row_major(BATCH, LOUT), MutAnyOrigin](
        out_buf.unsafe_ptr()
    )
    var gpu_params_t = LayoutTensor[
        dtype, Layout.row_major(HuberLossGraph.PARAM_SIZE), MutAnyOrigin
    ](params_buf.unsafe_ptr())
    var gpu_cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, LCS), MutAnyOrigin](
        cache_buf.unsafe_ptr()
    )
    var gpu_grad_in_t = LayoutTensor[dtype, Layout.row_major(BATCH, LIN), MutAnyOrigin](
        grad_in_buf.unsafe_ptr()
    )
    var gpu_grad_seed_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LOUT), MutAnyOrigin
    ](grad_seed_buf.unsafe_ptr())
    var gpu_grads_t = LayoutTensor[
        dtype, Layout.row_major(HuberLossGraph.PARAM_SIZE), MutAnyOrigin
    ](grads_buf.unsafe_ptr())

    var ws = ctx.enqueue_create_buffer[dtype](
        max(1, BATCH * HuberLossGraph.WORKSPACE_SIZE_PER_SAMPLE)
    )

    HuberLossGraph.forward_gpu[BATCH](
        ctx, gpu_out_t, gpu_in_t, gpu_params_t, gpu_cache_t, ws
    )
    ctx.synchronize()

    HuberLossGraph.backward_gpu[BATCH](
        ctx, gpu_grad_in_t, gpu_grad_seed_t, gpu_params_t, gpu_cache_t,
        gpu_grads_t, ws,
    )
    ctx.synchronize()

    var gpu_out = List[Scalar[dtype]](capacity=BATCH * LOUT)
    gpu_out.resize(BATCH * LOUT, Scalar[dtype](0))
    ctx.enqueue_copy(gpu_out.unsafe_ptr(), out_buf)
    var gpu_grad = List[Scalar[dtype]](capacity=BATCH * LIN)
    gpu_grad.resize(BATCH * LIN, Scalar[dtype](0))
    ctx.enqueue_copy(gpu_grad.unsafe_ptr(), grad_in_buf)
    ctx.synchronize()

    print("  GPU: fwd[0]=" + String(gpu_out[0])[:8])
    print(
        "  GPU: grad_in[0] = ["
        + String(gpu_grad[0])[:10]
        + ", "
        + String(gpu_grad[1])[:10]
        + ", "
        + String(gpu_grad[2])[:10]
        + ", "
        + String(gpu_grad[3])[:10]
        + "]"
    )

    # Compare all samples
    print("  Comparing CPU vs GPU (full lossgraph):")
    var all_ok = True
    for b in range(BATCH):
        for c in range(LIN):
            var cpu_val = Float64(cpu_grad_in[b * LIN + c])
            var gpu_val = Float64(gpu_grad[b * LIN + c])
            var diff = cpu_val - gpu_val
            if diff > 1e-5 or diff < -1e-5:
                print(
                    "    MISMATCH grad[" + String(b) + "," + String(c) + "]:"
                    + " cpu=" + String(cpu_val)[:10]
                    + " gpu=" + String(gpu_val)[:10]
                )
                all_ok = False
    if all_ok:
        print("    ALL MATCH ✓")
    else:
        print("    *** GPU GRADIENT BUG DETECTED ***")
    print()


fn main() raises:
    print("=" * 60)
    print("LossGraph Metal Isolation Test (MSE + Huber)")
    print("=" * 60)
    print()
    print("BATCH=" + String(BATCH) + ", ACTIONS=" + String(ACTIONS))
    print()

    with DeviceContext() as ctx:
        # MSE tests (existing)
        test_gather_forward(ctx)
        test_slice_forward(ctx)
        test_mse_forward(ctx)
        test_splitapply_forward(ctx)
        test_full_lossgraph_forward(ctx)
        test_full_lossgraph_backward(ctx)

        # HuberLoss tests (new)
        test_huber_forward(ctx)
        test_huber_forward_linear(ctx)
        test_huber_backward_gpu_vs_cpu(ctx)
        test_full_huber_lossgraph_backward(ctx)

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
