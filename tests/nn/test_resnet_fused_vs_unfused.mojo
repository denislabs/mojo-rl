"""Test fused vs unfused ResNet full PredModel for AlphaZero ConnectFour.

Instantiates both AlphaZeroConnectFourResNetConfig and
AlphaZeroConnectFourFusedResNetConfig PredModels with identical params,
runs forward+backward on GPU, and checks outputs match.

This catches divergence at production scale (F=128, 5 blocks, 6x7 board).

Run with:
    pixi run -e nvidia mojo run -I . tests/nn/test_resnet_fused_vs_unfused.mojo
    pixi run -e apple mojo run -I . tests/nn/test_resnet_fused_vs_unfused.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.memory import UnsafePointer
from layout import Layout, LayoutTensor
from std.random import random_float64
from std.math import abs

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.deep_agents.alphazero.configs import (
    AlphaZeroConnectFourResNetConfig,
    AlphaZeroConnectFourFusedResNetConfig,
)


def main() raises:
    print("=== Fused vs Unfused ResNet PredModel Test ===")
    print("(AlphaZero ConnectFour, F=128, 5 blocks, 6x7)\n")

    comptime Unfused = AlphaZeroConnectFourResNetConfig[]
    comptime Fused = AlphaZeroConnectFourFusedResNetConfig[]

    comptime UM = Unfused.PredModel
    comptime FM = Fused.PredModel

    comptime OBS = 126  # 3 channels × 6 × 7
    comptime OUT = UM.OUT_DIM  # 7 (policy) + 1 (value) = 8
    comptime BATCH = 4

    print("Unfused  IN_DIM:", UM.IN_DIM, " OUT_DIM:", UM.OUT_DIM)
    print("Fused    IN_DIM:", FM.IN_DIM, " OUT_DIM:", FM.OUT_DIM)
    print("Unfused  PARAM_SIZE:", UM.PARAM_SIZE)
    print("Fused    PARAM_SIZE:", FM.PARAM_SIZE)
    print("Unfused  CACHE_SIZE:", UM.CACHE_SIZE)
    print("Fused    CACHE_SIZE:", FM.CACHE_SIZE)
    print("Unfused  WORKSPACE:", UM.WORKSPACE_SIZE_PER_SAMPLE)
    print("Fused    WORKSPACE:", FM.WORKSPACE_SIZE_PER_SAMPLE)
    print()

    if UM.PARAM_SIZE != FM.PARAM_SIZE:
        print("WARNING: PARAM_SIZE mismatch! Unfused=", UM.PARAM_SIZE, " Fused=", FM.PARAM_SIZE)
        print("Cannot share params directly — test will init separately.")

    with DeviceContext() as ctx:
        # Input and grad_output
        var input_buf = ctx.enqueue_create_buffer[dtype](BATCH * OBS)
        var grad_out_buf = ctx.enqueue_create_buffer[dtype](BATCH * OUT)

        # Unfused buffers
        var u_out = ctx.enqueue_create_buffer[dtype](BATCH * OUT)
        var u_params = ctx.enqueue_create_buffer[dtype](UM.PARAM_SIZE)
        var u_cache = ctx.enqueue_create_buffer[dtype](BATCH * UM.CACHE_SIZE)
        var u_grads = ctx.enqueue_create_buffer[dtype](UM.PARAM_SIZE)
        var u_grad_in = ctx.enqueue_create_buffer[dtype](BATCH * OBS)
        var u_ws = ctx.enqueue_create_buffer[dtype](
            BATCH * UM.WORKSPACE_SIZE_PER_SAMPLE if UM.WORKSPACE_SIZE_PER_SAMPLE > 0 else 1
        )

        # Fused buffers
        var f_out = ctx.enqueue_create_buffer[dtype](BATCH * OUT)
        var f_params = ctx.enqueue_create_buffer[dtype](FM.PARAM_SIZE)
        var f_cache = ctx.enqueue_create_buffer[dtype](BATCH * FM.CACHE_SIZE)
        var f_grads = ctx.enqueue_create_buffer[dtype](FM.PARAM_SIZE)
        var f_grad_in = ctx.enqueue_create_buffer[dtype](BATCH * OBS)
        var f_ws = ctx.enqueue_create_buffer[dtype](
            BATCH * FM.WORKSPACE_SIZE_PER_SAMPLE if FM.WORKSPACE_SIZE_PER_SAMPLE > 0 else 1
        )

        # Random input
        with input_buf.map_to_host() as h:
            for i in range(BATCH * OBS):
                h[i] = Scalar[dtype](random_float64(-1.0, 1.0))

        # Random grad_output
        with grad_out_buf.map_to_host() as h:
            for i in range(BATCH * OUT):
                h[i] = Scalar[dtype](random_float64(-0.5, 0.5))

        # Initialize params — use same init for both
        var u_host = ctx.enqueue_create_host_buffer[dtype](UM.PARAM_SIZE)
        var u_p = LayoutTensor[dtype, Layout.row_major(UM.PARAM_SIZE), MutAnyOrigin](u_host.unsafe_ptr())
        UM.initialize_params[Kaiming[]](u_p)
        ctx.enqueue_copy(u_params, u_host)

        # Copy same params to fused (works if PARAM_SIZE matches)
        if UM.PARAM_SIZE == FM.PARAM_SIZE:
            ctx.enqueue_copy(f_params, u_host)
        else:
            var f_host = ctx.enqueue_create_host_buffer[dtype](FM.PARAM_SIZE)
            var f_p = LayoutTensor[dtype, Layout.row_major(FM.PARAM_SIZE), MutAnyOrigin](f_host.unsafe_ptr())
            FM.initialize_params[Kaiming[]](f_p)
            ctx.enqueue_copy(f_params, f_host)

        # Zero everything
        u_out.enqueue_fill(Scalar[dtype](0.0))
        f_out.enqueue_fill(Scalar[dtype](0.0))
        u_cache.enqueue_fill(Scalar[dtype](0.0))
        f_cache.enqueue_fill(Scalar[dtype](0.0))
        u_grads.enqueue_fill(Scalar[dtype](0.0))
        f_grads.enqueue_fill(Scalar[dtype](0.0))
        u_grad_in.enqueue_fill(Scalar[dtype](0.0))
        f_grad_in.enqueue_fill(Scalar[dtype](0.0))
        u_ws.enqueue_fill(Scalar[dtype](0.0))
        f_ws.enqueue_fill(Scalar[dtype](0.0))
        ctx.synchronize()

        # Tensor views
        var in_t = LayoutTensor[dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin](input_buf)
        var go_t = LayoutTensor[dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin](grad_out_buf)

        var u_out_t = LayoutTensor[dtype, Layout.row_major(BATCH, UM.OUT_DIM), MutAnyOrigin](u_out)
        var u_params_t = LayoutTensor[dtype, Layout.row_major(UM.PARAM_SIZE), MutAnyOrigin](u_params)
        var u_cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, UM.CACHE_SIZE), MutAnyOrigin](u_cache)
        var u_grads_t = LayoutTensor[dtype, Layout.row_major(UM.PARAM_SIZE), MutAnyOrigin](u_grads)
        var u_gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin](u_grad_in)

        var f_out_t = LayoutTensor[dtype, Layout.row_major(BATCH, FM.OUT_DIM), MutAnyOrigin](f_out)
        var f_params_t = LayoutTensor[dtype, Layout.row_major(FM.PARAM_SIZE), MutAnyOrigin](f_params)
        var f_cache_t = LayoutTensor[dtype, Layout.row_major(BATCH, FM.CACHE_SIZE), MutAnyOrigin](f_cache)
        var f_grads_t = LayoutTensor[dtype, Layout.row_major(FM.PARAM_SIZE), MutAnyOrigin](f_grads)
        var f_gi_t = LayoutTensor[dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin](f_grad_in)

        # ── Forward ──
        print("Running forward...")
        var u_state_t = LayoutTensor[dtype, Layout.row_major(UM.STATE_SIZE), MutAnyOrigin](
            UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(0))
        )
        var f_state_t = LayoutTensor[dtype, Layout.row_major(FM.STATE_SIZE), MutAnyOrigin](
            UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(0))
        )
        UM.forward_gpu[BATCH](ctx, u_out_t, in_t, u_params_t, u_state_t, u_cache_t, u_ws)
        FM.forward_gpu[BATCH](ctx, f_out_t, in_t, f_params_t, f_state_t, f_cache_t, f_ws)
        ctx.synchronize()

        # Compare forward outputs
        var u_fwd_host = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT)
        var f_fwd_host = ctx.enqueue_create_host_buffer[dtype](BATCH * OUT)
        ctx.enqueue_copy(u_fwd_host, u_out)
        ctx.enqueue_copy(f_fwd_host, f_out)
        ctx.synchronize()

        var max_fwd_diff = Scalar[dtype](0.0)
        var max_fwd_idx = 0
        for i in range(BATCH * OUT):
            var diff = abs(u_fwd_host[i] - f_fwd_host[i])
            if diff > max_fwd_diff:
                max_fwd_diff = diff
                max_fwd_idx = i

        print("Forward max diff:", max_fwd_diff, "at index", max_fwd_idx)
        if max_fwd_diff > 1e-2:
            print("FORWARD MISMATCH!")
            for b in range(BATCH):
                print("  Batch", b, ":")
                for i in range(OUT):
                    var idx = b * OUT + i
                    var label = "policy" if i < 7 else "value"
                    print(
                        "    [", label, i, "] unfused=", u_fwd_host[idx],
                        " fused=", f_fwd_host[idx],
                        " diff=", abs(u_fwd_host[idx] - f_fwd_host[idx]),
                    )
        else:
            print("Forward: PASS")

        # ── Backward ──
        print("\nRunning backward...")
        UM.backward_gpu[BATCH](ctx, u_gi_t, go_t, u_params_t, u_state_t, u_cache_t, u_grads_t, u_ws)
        FM.backward_gpu[BATCH](ctx, f_gi_t, go_t, f_params_t, f_state_t, f_cache_t, f_grads_t, f_ws)
        ctx.synchronize()

        # Compare grad_input
        var u_gi_host = ctx.enqueue_create_host_buffer[dtype](BATCH * OBS)
        var f_gi_host = ctx.enqueue_create_host_buffer[dtype](BATCH * OBS)
        ctx.enqueue_copy(u_gi_host, u_grad_in)
        ctx.enqueue_copy(f_gi_host, f_grad_in)
        ctx.synchronize()

        var max_gi_diff = Scalar[dtype](0.0)
        var max_gi_idx = 0
        for i in range(BATCH * OBS):
            var diff = abs(u_gi_host[i] - f_gi_host[i])
            if diff > max_gi_diff:
                max_gi_diff = diff
                max_gi_idx = i

        print("grad_input max diff:", max_gi_diff, "at index", max_gi_idx)
        if max_gi_diff > 1e-2:
            print("GRAD_INPUT MISMATCH!")
            for i in range(min(20, BATCH * OBS)):
                print("  [", i, "] unfused=", u_gi_host[i], " fused=", f_gi_host[i])
        else:
            print("grad_input: PASS")

        # Compare param grads
        var u_pg_host = ctx.enqueue_create_host_buffer[dtype](UM.PARAM_SIZE)
        var f_pg_host = ctx.enqueue_create_host_buffer[dtype](FM.PARAM_SIZE)
        ctx.enqueue_copy(u_pg_host, u_grads)
        ctx.enqueue_copy(f_pg_host, f_grads)
        ctx.synchronize()

        comptime MIN_PS = UM.PARAM_SIZE if UM.PARAM_SIZE < FM.PARAM_SIZE else FM.PARAM_SIZE
        var max_pg_diff = Scalar[dtype](0.0)
        var max_pg_idx = 0
        for i in range(MIN_PS):
            var diff = abs(u_pg_host[i] - f_pg_host[i])
            if diff > max_pg_diff:
                max_pg_diff = diff
                max_pg_idx = i

        print("param_grads max diff:", max_pg_diff, "at index", max_pg_idx)
        if max_pg_diff > 1e-2:
            print("PARAM_GRADS MISMATCH!")
            var start = max(0, max_pg_idx - 5)
            var end = min(MIN_PS, max_pg_idx + 6)
            for i in range(start, end):
                var marker = " <<<" if i == max_pg_idx else ""
                print("  [", i, "] unfused=", u_pg_host[i], " fused=", f_pg_host[i], marker)
        else:
            print("param_grads: PASS")

        # Summary
        print("\n" + "=" * 60)
        var all_pass = max_fwd_diff <= 1e-2 and max_gi_diff <= 1e-2 and max_pg_diff <= 1e-2
        if all_pass:
            print("ALL TESTS PASSED — fused and unfused ResNets are equivalent")
        else:
            print("SOME TESTS FAILED — fused and unfused produce different results")
            print("  Forward diff:     ", max_fwd_diff)
            print("  grad_input diff:  ", max_gi_diff)
            print("  param_grads diff: ", max_pg_diff)
        print("=" * 60)
