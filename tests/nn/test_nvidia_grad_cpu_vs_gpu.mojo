"""CPU-vs-GPU analytical gradient parity test.

Both `tests/nn/test_alphazero_architecture.mojo` and
`tests/nn/test_nvidia_gradcheck_isolate.mojo` compare GPU analytical
gradients against GPU finite-difference (numerical) gradients. NVIDIA
fails catastrophically on small layers; the OOB-write canary
(`tests/nn/test_nvidia_grad_oob_canary.mojo`) returned all-zero, so
the kernels write within bounds. That leaves two possibilities:

  (1) GPU backward kernel produces wrong values within bounded writes.
  (2) GPU forward (used by finite-diff) is unstable per perturbation,
      so the numerical reference itself is bad even when backward is
      correct.

This test discriminates by **bypassing the finite-difference reference
entirely**: with identical params + inputs + grad_output, run
`M.backward` on CPU and `M.backward_gpu` on GPU, then compare the two
analytical gradient buffers element-wise.

Apple should report ~zero diff (CPU and GPU paths produce the same
math). NVIDIA, if (1) is the bug, will show large per-element diffs
on the failing shapes (`Linear[8,4]`, `Linear[128,9]`, etc.) — and
the diff pattern will localize *which* gradient elements are wrong.
NVIDIA, if (2) is the bug, will show ~zero diff on every shape.

Run:
    pixi run -e nvidia mojo run -I . tests/nn/test_nvidia_grad_cpu_vs_gpu.mojo
    pixi run -e apple  mojo run -I . tests/nn/test_nvidia_grad_cpu_vs_gpu.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import abs
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import NetworkState, GPUNetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.model import Model, Linear, LinearReLU, Sequential


def parity_check[M: Model, BS: Int = 4](
    ctx: DeviceContext, name: String,
) raises:
    """Compare CPU and GPU analytical gradients element-wise."""
    comptime IN = M.IN_DIM
    comptime OUT = M.OUT_DIM
    comptime PS = M.PARAM_SIZE
    comptime CS = M.CACHE_SIZE
    comptime SS = M.STATE_SIZE
    comptime WS = M.WORKSPACE_SIZE_PER_SAMPLE

    print("Parity:", name, "(IN=", IN, "OUT=", OUT, "PS=", PS,
          "BS=", BS, ")")

    # ── Init shared params via CPU + Xavier (deterministic) ─────────────
    var cpu_state = NetworkState[M, Adam[]]()
    cpu_state.initialize[Xavier[]]()
    var gpu = GPUNetworkState[M, Adam[], dtype](ctx)
    gpu.upload_from(cpu_state, ctx)

    # ── Deterministic input + grad_output (same on CPU and GPU) ─────────
    var input_arr = List[Scalar[dtype]](capacity=BS * IN)
    for i in range(BS * IN):
        input_arr.append(
            Scalar[dtype](0.1 + Float64(i % 13) / 13.0 * 0.8)
        )
    var grad_out_arr = List[Scalar[dtype]](capacity=BS * OUT)
    for i in range(BS * OUT):
        grad_out_arr.append(
            Scalar[dtype](0.5 + Float64(i % 7) / 14.0 - Float64(i % 3) / 6.0)
        )

    # ── CPU forward + backward ──────────────────────────────────────────
    var cpu_out = List[Scalar[dtype]](capacity=BS * OUT)
    for _ in range(BS * OUT):
        cpu_out.append(Scalar[dtype](0.0))
    var cpu_cache = List[Scalar[dtype]](capacity=BS * CS + 1)
    for _ in range(BS * CS + 1):
        cpu_cache.append(Scalar[dtype](0.0))
    var cpu_grad_in = List[Scalar[dtype]](capacity=BS * IN)
    for _ in range(BS * IN):
        cpu_grad_in.append(Scalar[dtype](0.0))

    var cpu_input_t = LayoutTensor[
        dtype, Layout.row_major(BS, IN), MutAnyOrigin
    ](input_arr.unsafe_ptr())
    var cpu_out_t = LayoutTensor[
        dtype, Layout.row_major(BS, OUT), MutAnyOrigin
    ](cpu_out.unsafe_ptr())
    var cpu_cache_t = LayoutTensor[
        dtype, Layout.row_major(BS, CS), MutAnyOrigin
    ](cpu_cache.unsafe_ptr())
    var cpu_grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BS, OUT), MutAnyOrigin
    ](grad_out_arr.unsafe_ptr())
    var cpu_grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BS, IN), MutAnyOrigin
    ](cpu_grad_in.unsafe_ptr())

    M.forward[BS](
        cpu_input_t, cpu_out_t,
        cpu_state.params_view(), cpu_state.model_state_view(),
        cpu_cache_t,
    )
    cpu_state.zero_grads()
    var cpu_params_view = cpu_state.params_view()
    var cpu_state_view = cpu_state.model_state_view()
    var cpu_grads_view = cpu_state.grads_view()
    M.backward[BS](
        cpu_grad_out_t, cpu_grad_in_t,
        cpu_params_view, cpu_state_view,
        cpu_cache_t, cpu_grads_view,
    )

    # Snapshot CPU grads via the underlying pointer (avoids LayoutTensor
    # SIMD-element-type indexing quirks).
    var cpu_grads = List[Scalar[dtype]](capacity=PS)
    for i in range(PS):
        cpu_grads.append((cpu_state.grads + i)[])

    # ── GPU forward + backward ──────────────────────────────────────────
    var input_host = ctx.enqueue_create_host_buffer[dtype](BS * IN)
    for i in range(BS * IN):
        input_host[i] = input_arr[i]
    var grad_out_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
    for i in range(BS * OUT):
        grad_out_host[i] = grad_out_arr[i]

    var input_buf = ctx.enqueue_create_buffer[dtype](BS * IN)
    ctx.enqueue_copy(input_buf, input_host)
    var grad_out_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    ctx.enqueue_copy(grad_out_buf, grad_out_host)
    var output_buf = ctx.enqueue_create_buffer[dtype](BS * OUT)
    var cache_buf = ctx.enqueue_create_buffer[dtype](BS * CS if CS > 0 else 1)
    var grad_in_buf = ctx.enqueue_create_buffer[dtype](BS * IN)
    ctx.enqueue_memset(grad_in_buf, 0)
    var workspace = ctx.enqueue_create_buffer[dtype](
        BS * WS if WS > 0 else 1
    )

    var gpu_input_t = LayoutTensor[
        dtype, Layout.row_major(BS, IN), MutAnyOrigin
    ](input_buf.unsafe_ptr())
    var gpu_out_t = LayoutTensor[
        dtype, Layout.row_major(BS, OUT), MutAnyOrigin
    ](output_buf.unsafe_ptr())
    var gpu_cache_t = LayoutTensor[
        dtype, Layout.row_major(BS, CS), MutAnyOrigin
    ](cache_buf.unsafe_ptr())
    var gpu_grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BS, OUT), MutAnyOrigin
    ](grad_out_buf.unsafe_ptr())
    var gpu_grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BS, IN), MutAnyOrigin
    ](grad_in_buf.unsafe_ptr())

    M.forward_gpu[BS](
        ctx, gpu_out_t, gpu_input_t,
        gpu.params_view(), gpu.model_state_view(),
        gpu_cache_t, workspace,
    )
    gpu.zero_grads(ctx)
    var gpu_grads_view = gpu.grads_view()
    M.backward_gpu[BS](
        ctx, gpu_grad_in_t, gpu_grad_out_t,
        gpu.params_view(), gpu.model_state_view(),
        gpu_cache_t, gpu_grads_view, workspace,
    )

    var gpu_grads_host = ctx.enqueue_create_host_buffer[dtype](PS)
    ctx.enqueue_copy(gpu_grads_host, gpu.grads_buf)
    ctx.synchronize()

    # ── Element-wise comparison ─────────────────────────────────────────
    var max_abs = Float64(0.0)
    var max_rel = Float64(0.0)
    var n_mismatch = 0
    var first_mismatches = List[Int]()
    for i in range(PS):
        var c = Float64(cpu_grads[i])
        var g = Float64(gpu_grads_host[i])
        var err = abs(c - g)
        var denom = abs(c) + abs(g)
        var rel = Float64(0.0)
        if denom > 1e-8:
            rel = err / denom
        if err > max_abs:
            max_abs = err
        if rel > max_rel:
            max_rel = rel
        if rel > 0.01 and denom > 1e-6:
            n_mismatch += 1
            if len(first_mismatches) < 5:
                first_mismatches.append(i)

    if n_mismatch == 0:
        print(
            "  [PASS] max_rel=", max_rel,
            "max_abs=", max_abs,
            "(", PS, "params)",
        )
    else:
        print(
            "  [FAIL]", n_mismatch, "/", PS,
            "params differ — GPU backward computes wrong values",
        )
        print(
            "        max_rel=", max_rel, "max_abs=", max_abs,
        )
        for k in range(len(first_mismatches)):
            var i = first_mismatches[k]
            var c = Float64(cpu_grads[i])
            var g = Float64(gpu_grads_host[i])
            print(
                "        p[", i, "] cpu=", c, "gpu=", g,
                "diff=", g - c,
            )
    print()


def main() raises:
    print("=== CPU vs GPU backward analytical-gradient parity ===")
    print()
    var ctx = DeviceContext()

    # Shapes failing in gradcheck isolate.
    parity_check[Linear[8, 4]](ctx, "Linear[8,4] (gradcheck FAIL)")
    parity_check[LinearReLU[8, 4]](ctx, "LinearReLU[8,4] (gradcheck FAIL)")

    # Shapes passing in gradcheck isolate.
    parity_check[Linear[128, 1]](
        ctx, "Linear[128,1] (gradcheck PASS — OUT=1 path)"
    )
    parity_check[Sequential[LinearReLU[8, 6], Linear[6, 3]]](
        ctx, "Sequential[LinearReLU[8,6], Linear[6,3]] (gradcheck PASS)"
    )

    # TTT-MLP-like head shapes.
    parity_check[Linear[27, 9]](ctx, "Linear[27,9] (TTT-like)")
    parity_check[Linear[128, 9]](ctx, "Linear[128,9] (TicTacToe MLP head)")

    print("=== Done ===")
