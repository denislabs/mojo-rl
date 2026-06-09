"""Test ResBlockConv2DBN correctness vs decomposed ResBlockBN6x7.

Runs both forward+backward with identical inputs/params, checks outputs
and gradients match within tolerance.

Run with:
    pixi run -e nvidia mojo run -I . tests/nn/test_resblock_conv2d_bn.mojo
    pixi run -e apple mojo run -I . tests/nn/test_resblock_conv2d_bn.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.memory import UnsafePointer
from layout import Layout, LayoutTensor
from std.random import random_float64
from std.math import abs

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.model.resblock_conv2d_bn import ResBlockConv2DBN
from mojo_rl.nn.model.conv2d_bn_relu import Conv2DBatchNormReLU
from mojo_rl.nn.model.conv2d_layer import Conv2DLayer
from mojo_rl.nn.model.batch_norm_2d import BatchNorm2D
from mojo_rl.nn.model.sequential import Sequential
from mojo_rl.nn.model.relu import ReLU
from mojo_rl.nn.autodiff.combinators.residual import Residual


def main() raises:
    print("=== ResBlockConv2DBN Correctness Test ===\n")

    comptime F = 4  # Small channel count for testing
    comptime H = 3
    comptime W = 3
    comptime BATCH = 2
    comptime DIM = F * H * W

    comptime Fused = ResBlockConv2DBN[F, 3, 1, H, W]
    comptime Decomp = Sequential[
        Residual[Sequential[
            Conv2DBatchNormReLU[F, F, 3, 1, 1, H, W],
            Conv2DLayer[F, F, 3, 1, 1, H, W],
            BatchNorm2D[F, H, W],
        ]],
        ReLU[DIM],
    ]

    print("Fused  PARAM_SIZE:", Fused.PARAM_SIZE)
    print("Decomp PARAM_SIZE:", Decomp.PARAM_SIZE)

    if Fused.PARAM_SIZE != Decomp.PARAM_SIZE:
        print("ERROR: PARAM_SIZE mismatch!")
        return

    with DeviceContext() as ctx:
        # Allocate buffers for both
        var input_buf = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
        var grad_out_buf = ctx.enqueue_create_buffer[dtype](BATCH * DIM)

        var fused_out = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
        var fused_params = ctx.enqueue_create_buffer[dtype](Fused.PARAM_SIZE)
        var fused_cache = ctx.enqueue_create_buffer[dtype](BATCH * Fused.CACHE_SIZE)
        var fused_grads = ctx.enqueue_create_buffer[dtype](Fused.PARAM_SIZE)
        var fused_grad_in = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
        var fused_ws = ctx.enqueue_create_buffer[dtype](
            BATCH * Fused.WORKSPACE_SIZE_PER_SAMPLE if Fused.WORKSPACE_SIZE_PER_SAMPLE > 0 else 1
        )

        var decomp_out = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
        var decomp_params = ctx.enqueue_create_buffer[dtype](Decomp.PARAM_SIZE)
        var decomp_cache = ctx.enqueue_create_buffer[dtype](BATCH * Decomp.CACHE_SIZE)
        var decomp_grads = ctx.enqueue_create_buffer[dtype](Decomp.PARAM_SIZE)
        var decomp_grad_in = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
        var decomp_ws = ctx.enqueue_create_buffer[dtype](
            BATCH * Decomp.WORKSPACE_SIZE_PER_SAMPLE if Decomp.WORKSPACE_SIZE_PER_SAMPLE > 0 else 1
        )

        # Initialize with same random data
        with input_buf.map_to_host() as h:
            for i in range(BATCH * DIM):
                h[i] = Scalar[dtype](random_float64(-1.0, 1.0))

        with grad_out_buf.map_to_host() as h:
            for i in range(BATCH * DIM):
                h[i] = Scalar[dtype](random_float64(-1.0, 1.0))

        # Initialize params identically
        var params_host = ctx.enqueue_create_host_buffer[dtype](Fused.PARAM_SIZE)
        var fused_p = LayoutTensor[dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin](params_host.unsafe_ptr())
        Fused.initialize_params[Kaiming[]](fused_p)
        ctx.enqueue_copy(fused_params, params_host)
        ctx.enqueue_copy(decomp_params, params_host)  # Same params

        # Zero outputs, grads, caches
        fused_out.enqueue_fill(Scalar[dtype](0.0))
        decomp_out.enqueue_fill(Scalar[dtype](0.0))
        fused_cache.enqueue_fill(Scalar[dtype](0.0))
        decomp_cache.enqueue_fill(Scalar[dtype](0.0))
        fused_grads.enqueue_fill(Scalar[dtype](0.0))
        decomp_grads.enqueue_fill(Scalar[dtype](0.0))
        fused_grad_in.enqueue_fill(Scalar[dtype](0.0))
        decomp_grad_in.enqueue_fill(Scalar[dtype](0.0))
        fused_ws.enqueue_fill(Scalar[dtype](0.0))
        decomp_ws.enqueue_fill(Scalar[dtype](0.0))
        ctx.synchronize()

        # Create tensor views
        var in_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](input_buf)

        var f_out_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](fused_out)
        var f_params_t = LayoutTensor[dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin](fused_params)
        var f_cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, Fused.CACHE_SIZE), MutAnyOrigin](fused_cache)
        var f_grads_t = LayoutTensor[dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin](fused_grads)
        var f_gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](fused_grad_in)
        var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](grad_out_buf)

        var d_out_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](decomp_out)
        var d_params_t = LayoutTensor[dtype, Layout.row_major(Decomp.PARAM_SIZE), MutAnyOrigin](decomp_params)
        var d_cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, Decomp.CACHE_SIZE), MutAnyOrigin](decomp_cache)
        var d_grads_t = LayoutTensor[dtype, Layout.row_major(Decomp.PARAM_SIZE), MutAnyOrigin](decomp_grads)
        var d_gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](decomp_grad_in)

        # Forward
        print("Running forward...")
        var f_state_t = LayoutTensor[dtype, Layout.row_major(Fused.STATE_SIZE), MutAnyOrigin](
            UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(0))
        )
        var d_state_t = LayoutTensor[dtype, Layout.row_major(Decomp.STATE_SIZE), MutAnyOrigin](
            UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(0))
        )
        Fused.forward_gpu[BATCH](ctx, f_out_t, in_t, f_params_t, f_state_t, f_cache_t, fused_ws)
        Decomp.forward_gpu[BATCH](ctx, d_out_t, in_t, d_params_t, d_state_t, d_cache_t, decomp_ws)
        ctx.synchronize()

        # Compare forward outputs
        var f_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
        var d_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
        ctx.enqueue_copy(f_host, fused_out)
        ctx.enqueue_copy(d_host, decomp_out)
        ctx.synchronize()

        var max_fwd_diff = Scalar[dtype](0.0)
        for i in range(BATCH * DIM):
            var diff = abs(f_host[i] - d_host[i])
            if diff > max_fwd_diff:
                max_fwd_diff = diff

        print("Forward max diff:", max_fwd_diff)
        if max_fwd_diff > 1e-3:
            print("FORWARD MISMATCH!")
            for i in range(min(10, BATCH * DIM)):
                print("  [", i, "] fused=", f_host[i], " decomp=", d_host[i])
        else:
            print("Forward: PASS")

        # Backward
        print("\nRunning backward...")
        Fused.backward_gpu[BATCH](ctx, f_gi_t, go_t, f_params_t, f_state_t, f_cache_t, f_grads_t, fused_ws)
        Decomp.backward_gpu[BATCH](ctx, d_gi_t, go_t, d_params_t, d_state_t, d_cache_t, d_grads_t, decomp_ws)
        ctx.synchronize()

        # Compare grad_input
        var fgi_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
        var dgi_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
        ctx.enqueue_copy(fgi_host, fused_grad_in)
        ctx.enqueue_copy(dgi_host, decomp_grad_in)
        ctx.synchronize()

        var max_gi_diff = Scalar[dtype](0.0)
        for i in range(BATCH * DIM):
            var diff = abs(fgi_host[i] - dgi_host[i])
            if diff > max_gi_diff:
                max_gi_diff = diff

        print("grad_input max diff:", max_gi_diff)
        if max_gi_diff > 1e-3:
            print("GRAD_INPUT MISMATCH!")
            for i in range(min(10, BATCH * DIM)):
                print("  [", i, "] fused=", fgi_host[i], " decomp=", dgi_host[i])
        else:
            print("grad_input: PASS")

        # Compare param grads
        var fg_host = ctx.enqueue_create_host_buffer[dtype](Fused.PARAM_SIZE)
        var dg_host = ctx.enqueue_create_host_buffer[dtype](Decomp.PARAM_SIZE)
        ctx.enqueue_copy(fg_host, fused_grads)
        ctx.enqueue_copy(dg_host, decomp_grads)
        ctx.synchronize()

        var max_pg_diff = Scalar[dtype](0.0)
        var max_pg_idx = 0
        for i in range(Fused.PARAM_SIZE):
            var diff = abs(fg_host[i] - dg_host[i])
            if diff > max_pg_diff:
                max_pg_diff = diff
                max_pg_idx = i

        print("param_grads max diff:", max_pg_diff, "at index", max_pg_idx)
        if max_pg_diff > 1e-3:
            print("PARAM_GRADS MISMATCH!")
            # Show around the max diff
            var start = max(0, max_pg_idx - 3)
            var end = min(Fused.PARAM_SIZE, max_pg_idx + 4)
            for i in range(start, end):
                var marker = " <<<" if i == max_pg_idx else ""
                print("  [", i, "] fused=", fg_host[i], " decomp=", dg_host[i], marker)
        else:
            print("param_grads: PASS")

    print("\n=== Done ===")
