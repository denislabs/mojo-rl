"""Test ResBlockConv2D (no BN) correctness vs decomposed version."""

from std.gpu.host import DeviceContext
from std.memory import UnsafePointer
from layout import Layout, LayoutTensor
from std.random import random_float64
from std.math import abs

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.model.resblock_conv2d import ResBlockConv2D
from mojo_rl.nn.model.sequential import Sequential
from mojo_rl.nn.model.relu import ReLU
from mojo_rl.nn.autodiff.combinators.residual import Residual
from mojo_rl.nn.autodiff.fused import FusedConv2DActivation, ReLUActivation
from mojo_rl.nn.autodiff import AutoFused, Conv2D


def main() raises:
    print("=== ResBlockConv2D (no BN) Correctness Test ===\n")

    comptime F = 4
    comptime H = 3
    comptime W = 3
    comptime DIM = F * H * W
    comptime BATCH = 2

    comptime Fused = ResBlockConv2D[F, 3, 1, H, W]
    comptime Decomp = Sequential[
        Residual[Sequential[
            AutoFused[FusedConv2DActivation[F, F, 3, 1, 1, H, W, ReLUActivation]],
            AutoFused[Conv2D[F, F, 3, 1, 1, H, W]],
        ]],
        ReLU[DIM],
    ]

    print("Fused  PS:", Fused.PARAM_SIZE, " Decomp PS:", Decomp.PARAM_SIZE)

    with DeviceContext() as ctx:
        var input_buf = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
        var grad_out_buf = ctx.enqueue_create_buffer[dtype](BATCH * DIM)

        var f_out = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
        var f_params = ctx.enqueue_create_buffer[dtype](Fused.PARAM_SIZE)
        var f_cache = ctx.enqueue_create_buffer[dtype](BATCH * Fused.CACHE_SIZE)
        var f_grads = ctx.enqueue_create_buffer[dtype](Fused.PARAM_SIZE)
        var f_gi = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
        var f_ws = ctx.enqueue_create_buffer[dtype](
            BATCH * Fused.WORKSPACE_SIZE_PER_SAMPLE if Fused.WORKSPACE_SIZE_PER_SAMPLE > 0 else 1
        )

        var d_out = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
        var d_params = ctx.enqueue_create_buffer[dtype](Decomp.PARAM_SIZE)
        var d_cache = ctx.enqueue_create_buffer[dtype](BATCH * Decomp.CACHE_SIZE)
        var d_grads = ctx.enqueue_create_buffer[dtype](Decomp.PARAM_SIZE)
        var d_gi = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
        var d_ws = ctx.enqueue_create_buffer[dtype](
            BATCH * Decomp.WORKSPACE_SIZE_PER_SAMPLE if Decomp.WORKSPACE_SIZE_PER_SAMPLE > 0 else 1
        )

        with input_buf.map_to_host() as h:
            for i in range(BATCH * DIM):
                h[i] = Scalar[dtype](random_float64(-1.0, 1.0))
        with grad_out_buf.map_to_host() as h:
            for i in range(BATCH * DIM):
                h[i] = Scalar[dtype](random_float64(-1.0, 1.0))

        var ph = ctx.enqueue_create_host_buffer[dtype](Fused.PARAM_SIZE)
        var fp = LayoutTensor[dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin](ph.unsafe_ptr())
        Fused.initialize_params[Kaiming[]](fp)
        ctx.enqueue_copy(f_params, ph)
        ctx.enqueue_copy(d_params, ph)

        f_out.enqueue_fill(Scalar[dtype](0.0))
        d_out.enqueue_fill(Scalar[dtype](0.0))
        f_cache.enqueue_fill(Scalar[dtype](0.0))
        d_cache.enqueue_fill(Scalar[dtype](0.0))
        f_grads.enqueue_fill(Scalar[dtype](0.0))
        d_grads.enqueue_fill(Scalar[dtype](0.0))
        f_gi.enqueue_fill(Scalar[dtype](0.0))
        d_gi.enqueue_fill(Scalar[dtype](0.0))
        ctx.synchronize()

        var in_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](input_buf)
        var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](grad_out_buf)

        # Forward
        var f_out_t = LayoutTensor[dtype, Layout.row_major(BATCH, Fused.OUT_DIM), MutAnyOrigin](f_out)
        var f_in_t = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Fused.IN_DIM), MutAnyOrigin]](in_t)
        var f_p_t = LayoutTensor[dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin](f_params)
        var f_c_t = LayoutTensor[dtype, Layout.row_major(BATCH, Fused.CACHE_SIZE), MutAnyOrigin](f_cache)
        var f_s_t = LayoutTensor[dtype, Layout.row_major(Fused.STATE_SIZE), MutAnyOrigin](
            UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(0))
        )
        Fused.forward_gpu[BATCH](ctx, f_out_t, f_in_t, f_p_t, f_s_t, f_c_t, f_ws)

        var d_out_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](d_out)
        var d_p_t = LayoutTensor[dtype, Layout.row_major(Decomp.PARAM_SIZE), MutAnyOrigin](d_params)
        var d_c_t = LayoutTensor[dtype, Layout.row_major(BATCH, Decomp.CACHE_SIZE), MutAnyOrigin](d_cache)
        var d_s_t = LayoutTensor[dtype, Layout.row_major(Decomp.STATE_SIZE), MutAnyOrigin](
            UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(0))
        )
        Decomp.forward_gpu[BATCH](ctx, d_out_t, in_t, d_p_t, d_s_t, d_c_t, d_ws)
        ctx.synchronize()

        var fh = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
        var dh = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
        ctx.enqueue_copy(fh, f_out)
        ctx.enqueue_copy(dh, d_out)
        ctx.synchronize()

        var max_fwd = Scalar[dtype](0.0)
        for i in range(BATCH * DIM):
            var d = abs(fh[i] - dh[i])
            if d > max_fwd:
                max_fwd = d
        print("Forward max diff:", max_fwd)
        print("Forward:", "PASS" if max_fwd < 1e-3 else "FAIL")

        # Backward
        var f_gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, Fused.IN_DIM), MutAnyOrigin](f_gi)
        var f_go_t = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Fused.OUT_DIM), MutAnyOrigin]](go_t)
        var f_g_t = LayoutTensor[dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin](f_grads)
        Fused.backward_gpu[BATCH](ctx, f_gi_t, f_go_t, f_p_t, f_s_t, f_c_t, f_g_t, f_ws)

        var d_gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](d_gi)
        var d_g_t = LayoutTensor[dtype, Layout.row_major(Decomp.PARAM_SIZE), MutAnyOrigin](d_grads)
        Decomp.backward_gpu[BATCH](ctx, d_gi_t, go_t, d_p_t, d_s_t, d_c_t, d_g_t, d_ws)
        ctx.synchronize()

        var fgi_h = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
        var dgi_h = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
        ctx.enqueue_copy(fgi_h, f_gi)
        ctx.enqueue_copy(dgi_h, d_gi)
        ctx.synchronize()

        var max_gi = Scalar[dtype](0.0)
        for i in range(BATCH * DIM):
            var d = abs(fgi_h[i] - dgi_h[i])
            if d > max_gi:
                max_gi = d
        print("\ngrad_input max diff:", max_gi)
        print("grad_input:", "PASS" if max_gi < 1e-3 else "FAIL")

        var fg_h = ctx.enqueue_create_host_buffer[dtype](Fused.PARAM_SIZE)
        var dg_h = ctx.enqueue_create_host_buffer[dtype](Decomp.PARAM_SIZE)
        ctx.enqueue_copy(fg_h, f_grads)
        ctx.enqueue_copy(dg_h, d_grads)
        ctx.synchronize()

        var max_pg = Scalar[dtype](0.0)
        for i in range(Fused.PARAM_SIZE):
            var d = abs(fg_h[i] - dg_h[i])
            if d > max_pg:
                max_pg = d
        print("param_grads max diff:", max_pg)
        print("param_grads:", "PASS" if max_pg < 1e-3 else "FAIL")

    print("\n=== Done ===")
