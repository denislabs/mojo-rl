"""Quick bfloat16 GPU gradcheck to test SM100 native kernel consistency.

On RTX 5090, float32 matmul falls to a TF32 kernel. bfloat16 should use
the SM100 native UMMA kernel for forward (max_matmul). If the backward
also uses a bf16-compatible kernel, forward/backward should be self-consistent.

Usage:
    pixi run -e nvidia mojo run -I . tests/nn/test_bf16_gpu_gradcheck.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import abs, sqrt
from layout import Layout, LayoutTensor
from mojo_rl.nn.training import NetworkState, GPUNetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.model import (
    Model,
    Sequential,
    Parallel,
    Linear,
    LinearReLU,
)

comptime BF16 = DType.bfloat16


def gpu_gradcheck_bf16[M: Model, BS: Int = 4](
    ctx: DeviceContext,
    name: String,
    eps: Float64 = 1e-2,
    max_check: Int = 200,
    tol: Float64 = 0.05,
) raises:
    """GPU gradcheck using bfloat16.

    Uses larger eps (1e-2) because bf16 has less precision (~3 decimal digits).
    Tolerance is relaxed to 5% (bf16 finite-diff has inherent noise).
    """
    comptime IN = M.IN_DIM
    comptime OUT = M.OUT_DIM
    comptime PS = M.PARAM_SIZE
    comptime CS = M.CACHE_SIZE
    comptime WS = M.WORKSPACE_SIZE_PER_SAMPLE

    print("BF16 Gradcheck:", name, "(IN=", IN, "OUT=", OUT, "PS=", PS, ")")

    # Init in float32 on CPU, then convert to bf16 for GPU
    var cpu_f32 = NetworkState[M, Adam[]]()
    cpu_f32.initialize[Xavier[]]()

    # Create bf16 GPU state
    var gpu = GPUNetworkState[M, Adam[], BF16](ctx)

    # Upload: convert f32 params to bf16 via host buffer
    var params_bf16_host = ctx.enqueue_create_host_buffer[BF16](PS)
    for i in range(PS):
        params_bf16_host[i] = Scalar[BF16]((cpu_f32.params + i)[])
    ctx.enqueue_copy(gpu.params_buf, params_bf16_host)
    ctx.synchronize()

    var workspace = ctx.enqueue_create_buffer[BF16](BS * WS if WS > 0 else 1)

    # Input
    var input_host = ctx.enqueue_create_host_buffer[BF16](BS * IN)
    for i in range(BS * IN):
        input_host[i] = Scalar[BF16](0.1 + Float64(i % 13) / 13.0 * 0.8)
    var input_buf = ctx.enqueue_create_buffer[BF16](BS * IN)
    ctx.enqueue_copy(input_buf, input_host)

    # Grad output
    var grad_out_host = ctx.enqueue_create_host_buffer[BF16](BS * OUT)
    for i in range(BS * OUT):
        grad_out_host[i] = Scalar[BF16](
            0.5 + Float64(i % 7) / 14.0 - Float64(i % 3) / 6.0
        )
    var grad_out_buf = ctx.enqueue_create_buffer[BF16](BS * OUT)
    ctx.enqueue_copy(grad_out_buf, grad_out_host)

    # Forward + backward (analytical)
    var cache_buf = ctx.enqueue_create_buffer[BF16](BS * CS if CS > 0 else 1)
    var output_buf = ctx.enqueue_create_buffer[BF16](BS * OUT)

    var input_t = LayoutTensor[BF16, Layout.row_major(BS, IN), MutAnyOrigin](
        input_buf.unsafe_ptr()
    )
    var output_t = LayoutTensor[BF16, Layout.row_major(BS, OUT), MutAnyOrigin](
        output_buf.unsafe_ptr()
    )
    var cache_t = LayoutTensor[BF16, Layout.row_major(BS, CS), MutAnyOrigin](
        cache_buf.unsafe_ptr()
    )

    M.forward_gpu[BS, BF16](
        ctx, output_t, input_t, gpu.params_view(), cache_t, workspace
    )

    var grad_in_buf = ctx.enqueue_create_buffer[BF16](BS * IN)
    ctx.enqueue_memset(grad_in_buf, 0)
    var grad_out_t = LayoutTensor[BF16, Layout.row_major(BS, OUT), MutAnyOrigin](
        grad_out_buf.unsafe_ptr()
    )
    var grad_in_t = LayoutTensor[BF16, Layout.row_major(BS, IN), MutAnyOrigin](
        grad_in_buf.unsafe_ptr()
    )

    gpu.zero_grads(ctx)
    var grads = gpu.grads_view()
    M.backward_gpu[BS, BF16](
        ctx, grad_in_t, grad_out_t, gpu.params_view(), cache_t, grads, workspace,
    )

    var ana_grads_host = ctx.enqueue_create_host_buffer[BF16](PS)
    var params_host = ctx.enqueue_create_host_buffer[BF16](PS)
    var grad_out_host2 = ctx.enqueue_create_host_buffer[BF16](BS * OUT)
    ctx.enqueue_copy(ana_grads_host, gpu.grads_buf)
    ctx.enqueue_copy(params_host, gpu.params_buf)
    ctx.enqueue_copy(grad_out_host2, grad_out_buf)
    ctx.synchronize()

    # Finite differences via host-side perturbation
    var params_copy = ctx.enqueue_create_host_buffer[BF16](PS)
    var out_plus_buf = ctx.enqueue_create_buffer[BF16](BS * OUT)
    var out_minus_buf = ctx.enqueue_create_buffer[BF16](BS * OUT)
    var out_plus_host = ctx.enqueue_create_host_buffer[BF16](BS * OUT)
    var out_minus_host = ctx.enqueue_create_host_buffer[BF16](BS * OUT)
    var cache_tmp = ctx.enqueue_create_buffer[BF16](BS * CS if CS > 0 else 1)
    ctx.synchronize()

    var step = PS // max_check
    if step < 1:
        step = 1

    var max_abs: Float64 = 0.0
    var max_rel: Float64 = 0.0
    var num_checked = 0
    var num_fail = 0

    for p_idx in range(0, PS, step):
        for i in range(PS):
            params_copy[i] = params_host[i]
        var orig = Float64(params_host[p_idx])

        # f(p + eps)
        params_copy[p_idx] = Scalar[BF16](orig + eps)
        ctx.enqueue_copy(gpu.params_buf, params_copy)
        var out_plus_t = LayoutTensor[
            BF16, Layout.row_major(BS, OUT), MutAnyOrigin
        ](out_plus_buf.unsafe_ptr())
        var cache_tmp_t = LayoutTensor[
            BF16, Layout.row_major(BS, CS), MutAnyOrigin
        ](cache_tmp.unsafe_ptr())
        M.forward_gpu[BS, BF16](
            ctx, out_plus_t, input_t, gpu.params_view(), cache_tmp_t, workspace
        )
        ctx.enqueue_copy(out_plus_host, out_plus_buf)

        # f(p - eps)
        params_copy[p_idx] = Scalar[BF16](orig - eps)
        ctx.enqueue_copy(gpu.params_buf, params_copy)
        var out_minus_t = LayoutTensor[
            BF16, Layout.row_major(BS, OUT), MutAnyOrigin
        ](out_minus_buf.unsafe_ptr())
        var cache_m_t = LayoutTensor[
            BF16, Layout.row_major(BS, CS), MutAnyOrigin
        ](cache_tmp.unsafe_ptr())
        M.forward_gpu[BS, BF16](
            ctx, out_minus_t, input_t, gpu.params_view(), cache_m_t, workspace
        )
        ctx.enqueue_copy(out_minus_host, out_minus_buf)

        # Restore
        params_copy[p_idx] = Scalar[BF16](orig)
        ctx.enqueue_copy(gpu.params_buf, params_copy)
        ctx.synchronize()

        var num_grad: Float64 = 0.0
        for j in range(BS * OUT):
            var go = Float64(grad_out_host2[j])
            var fp = Float64(out_plus_host[j])
            var fm = Float64(out_minus_host[j])
            num_grad += go * (fp - fm) / (2.0 * eps)

        var ana_grad = Float64(ana_grads_host[p_idx])
        var err = abs(ana_grad - num_grad)
        var denom = abs(ana_grad) + abs(num_grad)
        var rel: Float64 = 0.0
        if denom > 1e-3:
            rel = err / denom

        if err > max_abs:
            max_abs = err
        if rel > max_rel:
            max_rel = rel
        if rel > tol and denom > 1e-2:
            num_fail += 1
            if num_fail <= 5:
                print(
                    "    MISMATCH p[", p_idx, "]: ana=",
                    ana_grad, "num=", num_grad, "rel=", rel,
                )
        num_checked += 1

    if num_fail == 0:
        print("  [PASS] max_rel=", max_rel, "(", num_checked, "checked)")
    else:
        print(
            "  [FAIL]", num_fail, "/", num_checked,
            "max_rel=", max_rel,
        )
    print()


def main() raises:
    print("=== BF16 GPU Gradcheck (SM100 native kernel test) ===")
    print()

    var ctx = DeviceContext()

    # Same models as the float32 isolate test
    gpu_gradcheck_bf16[Linear[8, 4]](ctx, "Linear[8,4]")
    gpu_gradcheck_bf16[LinearReLU[8, 4]](ctx, "LinearReLU[8,4]")
    gpu_gradcheck_bf16[Linear[128, 1]](ctx, "Linear[128,1]")

    gpu_gradcheck_bf16[Sequential[LinearReLU[8, 6], Linear[6, 4]]](
        ctx, "Sequential[LinearReLU[8,6], Linear[6,4]]"
    )

    gpu_gradcheck_bf16[Parallel[Linear[8, 4], Linear[8, 4]]](
        ctx, "Parallel[Linear[8,4], Linear[8,4]]"
    )

    gpu_gradcheck_bf16[Parallel[Linear[8, 4], Linear[8, 1]]](
        ctx, "Parallel[Linear[8,4], Linear[8,1]]"
    )

    # TicTacToe-like
    comptime MLP = Sequential[
        LinearReLU[27, 128],
        LinearReLU[128, 128],
        Parallel[Linear[128, 9], Linear[128, 1]],
    ]
    gpu_gradcheck_bf16[MLP](ctx, "TicTacToe MLP")

    print("=== Done ===")
