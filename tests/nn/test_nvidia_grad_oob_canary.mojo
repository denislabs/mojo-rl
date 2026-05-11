"""OOB-write canary test for backward kernels.

Hypothesis: the dW backward of MatMul/FusedMatMulBias writes past `grads_buf`
when out_dim is small (< MMA tile width). On NVIDIA the OOB writes corrupt
adjacent allocations because cuMemAlloc packs tightly; on Apple the
page-granular MTLBuffer absorbs the overrun, so the bug only manifests on
NVIDIA.

This test allocates one contiguous device buffer of size `PS + CANARY` and
treats the first PS elements as the gradients view. The remaining CANARY
elements are zeroed before the backward call. After backward, we read the
canary region back and report any non-zero entries — they prove the kernel
wrote past the `grads` LayoutTensor's declared extent.

Apple should report all-zero canary (i.e., test passes). NVIDIA is expected
to show non-zero canary entries on `Linear[8,4]` (and similarly small
out_dim cases) and zero on `Linear[128,1]` (different code path).

Run:
    pixi run -e nvidia mojo run -I . tests/nn/test_nvidia_grad_oob_canary.mojo
    pixi run -e apple  mojo run -I . tests/nn/test_nvidia_grad_oob_canary.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import abs
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import NetworkState, GPUNetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.model import Model, Linear, LinearReLU, Sequential


def canary_check[M: Model, BS: Int = 4, CANARY: Int = 2048](
    ctx: DeviceContext, name: String,
) raises:
    """Run M.backward_gpu over a [PS + CANARY] buffer and report any
    non-zero writes past PS.
    """
    comptime IN = M.IN_DIM
    comptime OUT = M.OUT_DIM
    comptime PS = M.PARAM_SIZE
    comptime CS = M.CACHE_SIZE
    comptime WS = M.WORKSPACE_SIZE_PER_SAMPLE

    print("Canary:", name, "(IN=", IN, "OUT=", OUT, "PS=", PS,
          "CANARY=", CANARY, ")")

    # ── Init model, params, optimizer state via the standard CPU path ───
    var cpu_state = NetworkState[M, Adam[]]()
    cpu_state.initialize[Xavier[]]()
    var gpu = GPUNetworkState[M, Adam[], dtype](ctx)
    gpu.upload_from(cpu_state, ctx)

    # ── Build a [PS + CANARY] grads buffer and zero it ──────────────────
    var grads_plus_canary = ctx.enqueue_create_buffer[dtype](PS + CANARY)
    ctx.enqueue_memset(grads_plus_canary, 0)

    # Grads view = first PS elements. The kernel expects exactly this
    # extent; anything written past index PS-1 is out-of-bounds.
    var grads_view = LayoutTensor[
        dtype, Layout.row_major(PS), MutAnyOrigin
    ](grads_plus_canary.unsafe_ptr())

    # ── Forward inputs / cache / output / grad_out ──────────────────────
    var input_host = ctx.enqueue_create_host_buffer[dtype](BS * IN)
    for i in range(BS * IN):
        input_host[i] = Scalar[dtype](0.1 + Float64(i % 13) / 13.0 * 0.8)
    var input_buf = ctx.enqueue_create_buffer[dtype](BS * IN)
    ctx.enqueue_copy(input_buf, input_host)

    var grad_out_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    for i in range(BS * OUT):
        grad_out_host[i] = Scalar[dtype](
            0.5 + Float64(i % 7) / 14.0 - Float64(i % 3) / 6.0
        )
    var grad_out_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    ctx.enqueue_copy(grad_out_buf, grad_out_host)

    var cache_buf = ctx.enqueue_create_buffer[dtype](BS * CS if CS > 0 else 1)
    var output_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    var workspace = ctx.enqueue_create_buffer[dtype](
        BS * WS if WS > 0 else 1
    )
    var grad_in_buf = ctx.enqueue_create_buffer[dtype](BS * IN)
    ctx.enqueue_memset(grad_in_buf, 0)

    var input_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](
        input_buf.unsafe_ptr()
    )
    var output_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](
        output_buf.unsafe_ptr()
    )
    var cache_t = LayoutTensor[dtype, Layout.row_major(BS, CS), MutAnyOrigin](
        cache_buf.unsafe_ptr()
    )
    var grad_out_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](
        grad_out_buf.unsafe_ptr()
    )
    var grad_in_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](
        grad_in_buf.unsafe_ptr()
    )

    # Forward populates cache.
    M.forward_gpu[BS](
        ctx, output_t, input_t, gpu.params_view(),
        gpu.model_state_view(), cache_t, workspace,
    )
    # Backward writes grads_view (and grad_in_t).
    M.backward_gpu[BS](
        ctx, grad_in_t, grad_out_t, gpu.params_view(),
        gpu.model_state_view(), cache_t, grads_view, workspace,
    )

    # ── Read back, scan canary slice ────────────────────────────────────
    var host = ctx.enqueue_create_host_buffer[dtype](PS + CANARY)
    ctx.enqueue_copy(host, grads_plus_canary)
    ctx.synchronize()

    var n_nonzero = 0
    var first_offsets = List[Int]()
    var max_abs_val = Float64(0.0)
    for j in range(CANARY):
        var v = Float64(host[PS + j])
        var av = abs(v)
        if av > 0.0:
            n_nonzero += 1
            if av > max_abs_val:
                max_abs_val = av
            if len(first_offsets) < 8:
                first_offsets.append(j)

    if n_nonzero == 0:
        print(
            "  [PASS] canary all zero (PS=", PS,
            ", scanned", CANARY, "elements past PS)",
        )
    else:
        print(
            "  [FAIL]", n_nonzero, "/", CANARY,
            "non-zero past PS — OOB write CONFIRMED",
        )
        print("        max |canary| =", max_abs_val)
        var s = String("        first offsets past PS: ")
        for k in range(len(first_offsets)):
            s += String(first_offsets[k])
            if k < len(first_offsets) - 1:
                s += ", "
        print(s)
    print()


def main() raises:
    print("=== NVIDIA backward OOB canary ===")
    print()
    var ctx = DeviceContext()

    # Failing levels in test_nvidia_gradcheck_isolate.mojo:
    canary_check[Linear[8, 4]](ctx, "Linear[8,4] (failing in isolate)")
    canary_check[LinearReLU[8, 4]](ctx, "LinearReLU[8,4] (failing in isolate)")

    # Passing levels — should report all-zero canary on every platform.
    canary_check[Linear[128, 1]](
        ctx, "Linear[128,1] (passing in isolate — OUT=1 path)"
    )
    canary_check[Sequential[LinearReLU[8, 6], Linear[6, 3]]](
        ctx, "Sequential[LinearReLU[8,6], Linear[6,3]] (passing in isolate)"
    )

    # Larger but small-out cases for sweep.
    canary_check[Linear[27, 9]](ctx, "Linear[27,9] (TTT-like head shape)")
    canary_check[Linear[128, 9]](
        ctx, "Linear[128,9] (TicTacToe MLP head)"
    )

    print("=== Done ===")
