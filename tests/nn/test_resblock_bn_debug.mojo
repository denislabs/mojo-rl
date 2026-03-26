"""Debug: compare intermediate values in ResBlockConv2DBN vs decomposed."""

from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor
from std.random import random_float64
from std.math import abs, ceildiv

from mojo_rl.nn.constants import dtype, TPB
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.model.resblock_conv2d_bn import ResBlockConv2DBN, _bn_skip_relu_bwd_kernel
from mojo_rl.nn.model.conv2d_bn_relu import Conv2DBatchNormReLU
from mojo_rl.nn.model.conv2d_layer import Conv2DLayer
from mojo_rl.nn.model.batch_norm_2d import BatchNorm2D
from mojo_rl.nn.model.sequential import Sequential
from mojo_rl.nn.model.relu import ReLU
from mojo_rl.nn.autodiff.combinators.residual import Residual


def main() raises:
    print("=== Debug: Isolate BN backward kernel ===\n")

    comptime F = 4
    comptime H = 3
    comptime W = 3
    comptime BATCH = 2
    comptime DIM = F * H * W
    comptime SPATIAL = H * W

    comptime Fused = ResBlockConv2DBN[F, 3, 1, H, W]
    comptime Decomp = Sequential[
        Residual[Sequential[
            Conv2DBatchNormReLU[F, F, 3, 1, 1, H, W],
            Conv2DLayer[F, F, 3, 1, 1, H, W],
            BatchNorm2D[F, H, W],
        ]],
        ReLU[DIM],
    ]

    with DeviceContext() as ctx:
        # Shared input and grad_output
        var input_buf = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
        var grad_out_buf = ctx.enqueue_create_buffer[dtype](BATCH * DIM)

        with input_buf.map_to_host() as h:
            for i in range(BATCH * DIM):
                h[i] = Scalar[dtype](random_float64(-1.0, 1.0))
        with grad_out_buf.map_to_host() as h:
            for i in range(BATCH * DIM):
                h[i] = Scalar[dtype](random_float64(-1.0, 1.0))

        # Shared params
        var ph = ctx.enqueue_create_host_buffer[dtype](Fused.PARAM_SIZE)
        var fp = LayoutTensor[dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin](ph.unsafe_ptr())
        Fused.initialize_params[Kaiming[]](fp)

        # ── Fused path: run forward, then ONLY step 1 of backward ──
        var f_params = ctx.enqueue_create_buffer[dtype](Fused.PARAM_SIZE)
        var f_out = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
        var f_cache = ctx.enqueue_create_buffer[dtype](BATCH * Fused.CACHE_SIZE)
        var f_ws = ctx.enqueue_create_buffer[dtype](BATCH * Fused.WORKSPACE_SIZE_PER_SAMPLE)
        var f_grads = ctx.enqueue_create_buffer[dtype](Fused.PARAM_SIZE)
        var f_grad_in = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
        var f_grad_conv2 = ctx.enqueue_create_buffer[dtype](BATCH * DIM)

        ctx.enqueue_copy(f_params, ph)
        f_out.enqueue_fill(Scalar[dtype](0.0))
        f_cache.enqueue_fill(Scalar[dtype](0.0))
        f_grads.enqueue_fill(Scalar[dtype](0.0))
        f_grad_in.enqueue_fill(Scalar[dtype](0.0))
        f_grad_conv2.enqueue_fill(Scalar[dtype](0.0))
        ctx.synchronize()

        var f_out_t = LayoutTensor[dtype, Layout.row_major(BATCH, Fused.OUT_DIM), MutAnyOrigin](f_out)
        var f_in_t = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Fused.IN_DIM), MutAnyOrigin]](
            LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](input_buf)
        )
        var f_p_t = LayoutTensor[dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin](f_params)
        var f_c_t = LayoutTensor[dtype, Layout.row_major(BATCH, Fused.CACHE_SIZE), MutAnyOrigin](f_cache)

        Fused.forward_gpu[BATCH](ctx, f_out_t, f_in_t, f_p_t, f_c_t, f_ws)
        ctx.synchronize()

        # Run ONLY the BN backward kernel (step 1)
        var bn2_params = LayoutTensor[dtype, Layout.row_major(Fused.BN2_PS), MutAnyOrigin](
            f_params.unsafe_ptr() + Fused.BN2_OFF
        )
        var bn2_cache = LayoutTensor[dtype, Layout.row_major(BATCH, Fused.BN2_CS), MutAnyOrigin](
            f_cache.unsafe_ptr() + BATCH * Fused.BN2_CACHE_OFF
        )
        var bn2_grads = LayoutTensor[dtype, Layout.row_major(Fused.BN2_PS), MutAnyOrigin](
            f_grads.unsafe_ptr() + Fused.BN2_OFF
        )
        var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](grad_out_buf)
        var gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](f_grad_in)
        var gc2_t = LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](f_grad_conv2)

        comptime bwd_k = _bn_skip_relu_bwd_kernel[
            BATCH, F, SPATIAL, Fused.BN2_PS, Fused.BN2_CS,
            Fused.BN2_GAMMA_OFF, Fused.BN2_BETA_OFF, Fused.BN2_XHAT_OFF, Fused.BN2_INVSTD_OFF,
            dtype,
        ]
        ctx.enqueue_function[bwd_k, bwd_k](
            gc2_t, go_t, gi_t, bn2_params, bn2_cache, bn2_grads,
            grid_dim=(F,), block_dim=(TPB,),
        )
        ctx.synchronize()

        # ── Decomposed path: run forward, extract ReLU-masked grad + BN backward ──
        var d_params = ctx.enqueue_create_buffer[dtype](Decomp.PARAM_SIZE)
        var d_out = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
        var d_cache = ctx.enqueue_create_buffer[dtype](BATCH * Decomp.CACHE_SIZE)
        var d_ws = ctx.enqueue_create_buffer[dtype](
            BATCH * Decomp.WORKSPACE_SIZE_PER_SAMPLE if Decomp.WORKSPACE_SIZE_PER_SAMPLE > 0 else 1
        )
        var d_grads = ctx.enqueue_create_buffer[dtype](Decomp.PARAM_SIZE)
        var d_grad_in = ctx.enqueue_create_buffer[dtype](BATCH * DIM)

        ctx.enqueue_copy(d_params, ph)
        d_out.enqueue_fill(Scalar[dtype](0.0))
        d_cache.enqueue_fill(Scalar[dtype](0.0))
        d_grads.enqueue_fill(Scalar[dtype](0.0))
        d_grad_in.enqueue_fill(Scalar[dtype](0.0))
        ctx.synchronize()

        var d_out_t = LayoutTensor[dtype, Layout.row_major(BATCH, Decomp.OUT_DIM), MutAnyOrigin](d_out)
        var d_in_t = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Decomp.IN_DIM), MutAnyOrigin]](
            LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](input_buf)
        )
        var d_p_t = LayoutTensor[dtype, Layout.row_major(Decomp.PARAM_SIZE), MutAnyOrigin](d_params)
        var d_c_t = LayoutTensor[dtype, Layout.row_major(BATCH, Decomp.CACHE_SIZE), MutAnyOrigin](d_cache)
        var d_g_t = LayoutTensor[dtype, Layout.row_major(Decomp.PARAM_SIZE), MutAnyOrigin](d_grads)
        var d_gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, Decomp.IN_DIM), MutAnyOrigin](d_grad_in)

        Decomp.forward_gpu[BATCH](ctx, d_out_t, d_in_t, d_p_t, d_c_t, d_ws)
        ctx.synchronize()

        # Full decomposed backward
        var d_go_t = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Decomp.OUT_DIM), MutAnyOrigin]](
            LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](grad_out_buf)
        )
        # IMPORTANT: make a copy of grad_out for decomposed since it may modify it
        var grad_out_copy = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
        ctx.enqueue_copy(grad_out_copy, grad_out_buf)
        ctx.synchronize()
        var d_go_t2 = rebind[LayoutTensor[dtype, Layout.row_major(BATCH, Decomp.OUT_DIM), MutAnyOrigin]](
            LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin](grad_out_copy)
        )
        Decomp.backward_gpu[BATCH](ctx, d_gi_t, d_go_t2, d_p_t, d_c_t, d_g_t, d_ws)
        ctx.synchronize()

        # Compare fused grad_conv2 vs decomposed grad_input (full backward result)
        print("Fused BN backward only (grad_conv2 + grad_skip):")
        var gc2_h = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
        ctx.enqueue_copy(gc2_h, f_grad_conv2)
        ctx.synchronize()
        print("  grad_conv2[0:8]:", end="")
        for i in range(8):
            print(" ", gc2_h[i], end="")
        print()

        var fgi_h = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
        ctx.enqueue_copy(fgi_h, f_grad_in)
        ctx.synchronize()
        print("  grad_skip[0:8]:", end="")
        for i in range(8):
            print(" ", fgi_h[i], end="")
        print()

        print("\nDecomposed full backward:")
        var dgi_h = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
        ctx.enqueue_copy(dgi_h, d_grad_in)
        ctx.synchronize()
        print("  grad_input[0:8]:", end="")
        for i in range(8):
            print(" ", dgi_h[i], end="")
        print()

        # Compare BN2 param grads (fused BN2 grads vs decomposed BN2 grads)
        # In decomposed, BN2 grads are at offset: Conv2DBatchNormReLU.PS + Conv2DLayer.PS
        comptime D_BN2_GOFF = Conv2DBatchNormReLU[F,F,3,1,1,H,W].PARAM_SIZE + Conv2DLayer[F,F,3,1,1,H,W].PARAM_SIZE
        var fg_h = ctx.enqueue_create_host_buffer[dtype](Fused.PARAM_SIZE)
        var dg_h = ctx.enqueue_create_host_buffer[dtype](Decomp.PARAM_SIZE)
        ctx.enqueue_copy(fg_h, f_grads)
        ctx.enqueue_copy(dg_h, d_grads)
        ctx.synchronize()

        print("\nBN2 param grads comparison (gamma grads, 4 channels):")
        print("  Fused BN2 gamma grads:", end="")
        for c in range(F):
            print(" ", fg_h[Fused.BN2_OFF + c], end="")
        print()
        print("  Decomp BN2 gamma grads:", end="")
        for c in range(F):
            print(" ", dg_h[D_BN2_GOFF + c], end="")
        print()

        print("\n  Fused BN2 beta grads:", end="")
        for c in range(F):
            print(" ", fg_h[Fused.BN2_OFF + F + c], end="")
        print()
        print("  Decomp BN2 beta grads:", end="")
        for c in range(F):
            print(" ", dg_h[D_BN2_GOFF + F + c], end="")
        print()

    print("\n=== Done ===")
