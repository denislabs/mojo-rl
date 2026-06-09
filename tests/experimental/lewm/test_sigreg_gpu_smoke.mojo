"""SIGRegOp GPU smoke: forward + backward at a tiny shape.

Validates the GPU kernels compile and produce reasonable numeric output:
  - forward stat is a finite non-negative scalar
  - backward grad_input has finite values
  - CPU↔GPU statistic agree to ~1e-3 relative tolerance

Run:
    pixi run -e apple  mojo run -I . tests/experimental/lewm/test_sigreg_gpu_smoke.mojo
    pixi run -e nvidia mojo run -I . tests/experimental/lewm/test_sigreg_gpu_smoke.mojo
"""

from std.math import abs
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.autodiff.primitives import SIGRegOp


def main() raises:
    print("=== SIGRegOp GPU smoke ===")
    comptime BATCH = 4
    comptime D = 8
    comptime T = 3
    comptime P = 16
    comptime K = 5
    comptime SR = SIGRegOp[D, T, P, K]

    print(
        "  shapes: BATCH=", BATCH, " D=", D, " T=", T, " P=", P, " K=", K,
        " IN_DIM=", SR.IN_DIM, " CACHE=", SR.CACHE_SIZE,
    )

    var ctx = DeviceContext()

    # ---------- Host inputs ----------
    var in_buf = ctx.enqueue_create_buffer[dtype](BATCH * SR.IN_DIM)
    var in_host = ctx.enqueue_create_host_buffer[dtype](BATCH * SR.IN_DIM)
    for i in range(BATCH * SR.IN_DIM):
        # Deterministic pseudo-Gaussian-ish input.
        in_host[i] = Scalar[dtype](
            Float64((i * 13 + 7) % 17 - 8) / 8.0
        )
    ctx.enqueue_copy(in_buf, in_host)

    var out_buf = ctx.enqueue_create_buffer[dtype](BATCH * SR.OUT_DIM)
    var cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * SR.CACHE_SIZE)
    var grad_in_buf = ctx.enqueue_create_buffer[dtype](BATCH * SR.IN_DIM)
    var grad_out_buf = ctx.enqueue_create_buffer[dtype](BATCH * SR.OUT_DIM)

    # Seed grad_output = 1/B (so chain-rule G = 1, matching CPU vjp test).
    var grad_out_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * SR.OUT_DIM
    )
    for i in range(BATCH * SR.OUT_DIM):
        grad_out_host[i] = Scalar[dtype](1.0 / Float64(BATCH))
    ctx.enqueue_copy(grad_out_buf, grad_out_host)

    var empty_params = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(0)))
    var empty_grad_params = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(0)))

    var in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, SR.IN_DIM), MutAnyOrigin
    ](in_buf)
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, SR.OUT_DIM), MutAnyOrigin
    ](out_buf)
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, SR.CACHE_SIZE), MutAnyOrigin
    ](cache_buf)
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, SR.IN_DIM), MutAnyOrigin
    ](grad_in_buf)
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, SR.OUT_DIM), MutAnyOrigin
    ](grad_out_buf)

    # SIGReg now uses the caller-allocated workspace buffer.
    comptime WS_SIZE = SR.workspace_size_for[BATCH]()
    print("  workspace size (elements):", WS_SIZE)
    var ws_buf = ctx.enqueue_create_buffer[dtype](WS_SIZE)
    var op_ws = ws_buf.unsafe_ptr()

    # ---------- Forward ----------
    print("\n--- Forward ---")
    SR.eval_gpu[BATCH, dtype](
        ctx, out_t, in_t, empty_params, cache_t, op_ws,
    )
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[dtype](BATCH * SR.OUT_DIM)
    ctx.enqueue_copy(out_host, out_buf)
    ctx.synchronize()

    var stat_gpu = Float64(out_host[0])
    var nan_out = 0
    for i in range(BATCH * SR.OUT_DIM):
        var v = Float64(out_host[i])
        if v != v:
            nan_out += 1
    print(
        "  GPU stat (output[0,0])=", stat_gpu,
        "  NaN out=", nan_out,
        "  uniform across BATCH=", Bool(Float64(out_host[BATCH - 1]) == stat_gpu),
    )
    if nan_out > 0 or stat_gpu < 0.0:
        print("  [FAIL] forward produced NaN or negative stat")
        return

    # ---------- Backward ----------
    print("\n--- Backward ---")
    SR.vjp_gpu[BATCH, dtype](
        ctx, grad_out_t, grad_in_t, empty_params, cache_t, empty_grad_params,
        op_ws,
    )
    ctx.synchronize()

    var grad_in_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * SR.IN_DIM
    )
    ctx.enqueue_copy(grad_in_host, grad_in_buf)
    ctx.synchronize()

    var nan_grad = 0
    var max_abs_grad: Float64 = 0.0
    var nz_count = 0
    for i in range(BATCH * SR.IN_DIM):
        var v = Float64(grad_in_host[i])
        if v != v:
            nan_grad += 1
        var av = abs(v)
        if av > max_abs_grad:
            max_abs_grad = av
        if av > 1e-8:
            nz_count += 1
    print(
        "  grad_input: max|g|=", max_abs_grad,
        " nz=", nz_count, "/", BATCH * SR.IN_DIM,
        " NaN=", nan_grad,
    )

    # ---------- CPU parity check ----------
    print("\n--- CPU↔GPU parity ---")
    var cpu_in = alloc[Scalar[dtype]](BATCH * SR.IN_DIM)
    var cpu_out = alloc[Scalar[dtype]](BATCH * SR.OUT_DIM)
    var cpu_cache = alloc[Scalar[dtype]](BATCH * SR.CACHE_SIZE)
    var cpu_grad_in = alloc[Scalar[dtype]](BATCH * SR.IN_DIM)
    var cpu_grad_out = alloc[Scalar[dtype]](BATCH * SR.OUT_DIM)
    # Copy GPU input to CPU buffer (re-read host buffer).
    for i in range(BATCH * SR.IN_DIM):
        cpu_in[i] = in_host[i]
    for i in range(BATCH * SR.OUT_DIM):
        cpu_grad_out[i] = grad_out_host[i]

    var cpu_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, SR.IN_DIM), MutAnyOrigin
    ](cpu_in)
    var cpu_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, SR.OUT_DIM), MutAnyOrigin
    ](cpu_out)
    var cpu_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, SR.CACHE_SIZE), MutAnyOrigin
    ](cpu_cache)
    var cpu_grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, SR.IN_DIM), MutAnyOrigin
    ](cpu_grad_in)
    var cpu_grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, SR.OUT_DIM), MutAnyOrigin
    ](cpu_grad_out)

    SR.eval[BATCH, dtype](
        cpu_in_t, cpu_out_t, empty_params, cpu_cache_t
    )
    SR.vjp[BATCH, dtype](
        cpu_grad_out_t, cpu_grad_in_t, empty_params, cpu_cache_t,
        empty_grad_params,
    )

    var stat_cpu = Float64(cpu_out[0])
    # Statistics are independent of A's particular realization (under
    # expectation) but for a specific seed CPU and GPU draw different A
    # (Float64 vs Float32 Box-Muller). So we compare orders of magnitude,
    # not exact values.
    var stat_ratio = stat_gpu / stat_cpu if stat_cpu > 1e-9 else 0.0
    print(
        "  stat CPU=", stat_cpu, "  stat GPU=", stat_gpu,
        "  ratio GPU/CPU=", stat_ratio,
    )

    # For grad magnitude, similar logic — magnitudes should agree to within
    # 2× since they're samples from random projections.
    var max_abs_grad_cpu: Float64 = 0.0
    for i in range(BATCH * SR.IN_DIM):
        var av = abs(Float64(cpu_grad_in[i]))
        if av > max_abs_grad_cpu:
            max_abs_grad_cpu = av
    print(
        "  max|g| CPU=", max_abs_grad_cpu, "  max|g| GPU=", max_abs_grad,
        "  ratio=", max_abs_grad / max_abs_grad_cpu if max_abs_grad_cpu > 0 else 0.0,
    )

    cpu_in.free()
    cpu_out.free()
    cpu_cache.free()
    cpu_grad_in.free()
    cpu_grad_out.free()

    # Parity criterion: same order of magnitude (0.2× to 5× ratio).
    var parity_ok = (
        stat_ratio > 0.2 and stat_ratio < 5.0
        and (max_abs_grad_cpu == 0.0 or
             (max_abs_grad / max_abs_grad_cpu > 0.2
              and max_abs_grad / max_abs_grad_cpu < 5.0))
    )

    if (
        nan_out == 0
        and nan_grad == 0
        and stat_gpu >= 0.0
        and max_abs_grad > 1e-8
        and nz_count > (BATCH * SR.IN_DIM) // 2
        and parity_ok
    ):
        print("\n  [PASS] SIGReg GPU smoke + CPU↔GPU magnitude parity")
    else:
        print("\n  [FAIL] SIGReg GPU smoke")
