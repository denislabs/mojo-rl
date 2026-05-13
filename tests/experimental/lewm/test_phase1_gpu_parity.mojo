"""Phase 1 GPU parity — Modulate + Gate.

Validates that GPU eval/vjp produce the same results as CPU within tolerance.

SIGRegOp is intentionally skipped (Phase 1 ships CPU only — see
`mojo_rl/nn/autodiff/primitives/sigreg.mojo` for the rationale).

Run on Apple Metal:
    pixi run -e apple mojo run -I . tests/experimental/lewm/test_phase1_gpu_parity.mojo
Run on NVIDIA:
    pixi run -e nvidia mojo run -I . tests/experimental/lewm/test_phase1_gpu_parity.mojo
"""

from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.memory import alloc
from std.math import abs
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.autodiff.primitives import ModulateOp, GateOp


def _max_diff(
    a_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
) -> Float64:
    var m = Float64(0.0)
    for i in range(n):
        var d = abs(Float64(a_ptr[i]) - Float64(b_ptr[i]))
        if d > m:
            m = d
    return m


# =============================================================================
# Modulate parity
# =============================================================================

def test_modulate_gpu_parity(ctx: DeviceContext) raises:
    comptime BATCH = 4
    comptime DIM = 8
    comptime IN = 3 * DIM
    comptime CACHE = 2 * DIM

    # ---------- CPU side ----------
    var input_ptr = alloc[Scalar[dtype]](BATCH * IN)
    for i in range(BATCH * IN):
        input_ptr[i] = Scalar[dtype](0.21 * Float64(i % 17) - 0.7)

    var cpu_output_ptr = alloc[Scalar[dtype]](BATCH * DIM)
    var cpu_cache_ptr = alloc[Scalar[dtype]](BATCH * CACHE)
    var params_ptr = alloc[Scalar[dtype]](1)

    var cpu_input_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](input_ptr)
    var cpu_output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](cpu_output_ptr)
    var cpu_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, CACHE), MutAnyOrigin
    ](cpu_cache_ptr)
    var params_t = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](params_ptr)

    ModulateOp[DIM].eval[BATCH](
        cpu_input_t, cpu_output_t, params_t, cpu_cache_t
    )

    # ---------- GPU side: upload input, run kernel, download output ----------
    var gpu_input_host = ctx.enqueue_create_host_buffer[dtype](BATCH * IN)
    for i in range(BATCH * IN):
        gpu_input_host[i] = input_ptr[i]
    var gpu_input_buf = ctx.enqueue_create_buffer[dtype](BATCH * IN)
    ctx.enqueue_copy(gpu_input_buf, gpu_input_host)

    var gpu_output_buf = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    var gpu_cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * CACHE)
    var gpu_params_buf = ctx.enqueue_create_buffer[dtype](1)

    var gpu_input_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](gpu_input_buf.unsafe_ptr())
    var gpu_output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](gpu_output_buf.unsafe_ptr())
    var gpu_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, CACHE), MutAnyOrigin
    ](gpu_cache_buf.unsafe_ptr())
    var gpu_params_t = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](gpu_params_buf.unsafe_ptr())

    var ws = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=0
    )
    ModulateOp[DIM].eval_gpu[BATCH](
        ctx, gpu_output_t, gpu_input_t, gpu_params_t, gpu_cache_t, ws
    )

    var gpu_output_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
    ctx.enqueue_copy(gpu_output_host, gpu_output_buf)
    ctx.synchronize()

    var fwd_diff = _max_diff(
        cpu_output_ptr, gpu_output_host.unsafe_ptr(), BATCH * DIM
    )

    # ---------- Backward parity ----------
    var grad_out_ptr = alloc[Scalar[dtype]](BATCH * DIM)
    for i in range(BATCH * DIM):
        grad_out_ptr[i] = Scalar[dtype](1.0)

    var cpu_grad_in_ptr = alloc[Scalar[dtype]](BATCH * IN)
    var cpu_grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](cpu_grad_in_ptr)
    var cpu_grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](grad_out_ptr)
    var grad_params_ptr = alloc[Scalar[dtype]](1)
    var grad_params_t = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](grad_params_ptr)
    ModulateOp[DIM].vjp[BATCH](
        cpu_grad_out_t, cpu_grad_in_t, params_t, cpu_cache_t, grad_params_t
    )

    var gpu_grad_out_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
    for i in range(BATCH * DIM):
        gpu_grad_out_host[i] = grad_out_ptr[i]
    var gpu_grad_out_buf = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    ctx.enqueue_copy(gpu_grad_out_buf, gpu_grad_out_host)

    var gpu_grad_in_buf = ctx.enqueue_create_buffer[dtype](BATCH * IN)
    var gpu_grad_params_buf = ctx.enqueue_create_buffer[dtype](1)

    var gpu_grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](gpu_grad_out_buf.unsafe_ptr())
    var gpu_grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](gpu_grad_in_buf.unsafe_ptr())
    var gpu_grad_params_t = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](gpu_grad_params_buf.unsafe_ptr())

    ModulateOp[DIM].vjp_gpu[BATCH](
        ctx,
        gpu_grad_out_t,
        gpu_grad_in_t,
        gpu_params_t,
        gpu_cache_t,
        gpu_grad_params_t,
        ws,
    )

    var gpu_grad_in_host = ctx.enqueue_create_host_buffer[dtype](BATCH * IN)
    ctx.enqueue_copy(gpu_grad_in_host, gpu_grad_in_buf)
    ctx.synchronize()

    var bwd_diff = _max_diff(
        cpu_grad_in_ptr, gpu_grad_in_host.unsafe_ptr(), BATCH * IN
    )

    if fwd_diff < 1e-4 and bwd_diff < 1e-4:
        print(
            "  [PASS] Modulate CPU/GPU parity: fwd_diff =",
            fwd_diff,
            "bwd_diff =",
            bwd_diff,
        )
    else:
        print(
            "  [FAIL] Modulate CPU/GPU parity: fwd_diff =",
            fwd_diff,
            "bwd_diff =",
            bwd_diff,
        )


# =============================================================================
# Gate parity
# =============================================================================

def test_gate_gpu_parity(ctx: DeviceContext) raises:
    comptime BATCH = 4
    comptime DIM = 8
    comptime IN = 3 * DIM
    comptime CACHE = 2 * DIM

    var input_ptr = alloc[Scalar[dtype]](BATCH * IN)
    for i in range(BATCH * IN):
        input_ptr[i] = Scalar[dtype](0.19 * Float64(i % 11) - 0.55)

    var cpu_output_ptr = alloc[Scalar[dtype]](BATCH * DIM)
    var cpu_cache_ptr = alloc[Scalar[dtype]](BATCH * CACHE)
    var params_ptr = alloc[Scalar[dtype]](1)

    var cpu_input_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](input_ptr)
    var cpu_output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](cpu_output_ptr)
    var cpu_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, CACHE), MutAnyOrigin
    ](cpu_cache_ptr)
    var params_t = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](params_ptr)

    GateOp[DIM].eval[BATCH](cpu_input_t, cpu_output_t, params_t, cpu_cache_t)

    var gpu_input_host = ctx.enqueue_create_host_buffer[dtype](BATCH * IN)
    for i in range(BATCH * IN):
        gpu_input_host[i] = input_ptr[i]
    var gpu_input_buf = ctx.enqueue_create_buffer[dtype](BATCH * IN)
    ctx.enqueue_copy(gpu_input_buf, gpu_input_host)

    var gpu_output_buf = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    var gpu_cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * CACHE)
    var gpu_params_buf = ctx.enqueue_create_buffer[dtype](1)

    var gpu_input_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](gpu_input_buf.unsafe_ptr())
    var gpu_output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](gpu_output_buf.unsafe_ptr())
    var gpu_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, CACHE), MutAnyOrigin
    ](gpu_cache_buf.unsafe_ptr())
    var gpu_params_t = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](gpu_params_buf.unsafe_ptr())

    var ws = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=0
    )
    GateOp[DIM].eval_gpu[BATCH](
        ctx, gpu_output_t, gpu_input_t, gpu_params_t, gpu_cache_t, ws
    )

    var gpu_output_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
    ctx.enqueue_copy(gpu_output_host, gpu_output_buf)
    ctx.synchronize()

    var fwd_diff = _max_diff(
        cpu_output_ptr, gpu_output_host.unsafe_ptr(), BATCH * DIM
    )

    var grad_out_ptr = alloc[Scalar[dtype]](BATCH * DIM)
    for i in range(BATCH * DIM):
        grad_out_ptr[i] = Scalar[dtype](1.0)

    var cpu_grad_in_ptr = alloc[Scalar[dtype]](BATCH * IN)
    var cpu_grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](cpu_grad_in_ptr)
    var cpu_grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](grad_out_ptr)
    var grad_params_ptr = alloc[Scalar[dtype]](1)
    var grad_params_t = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](grad_params_ptr)
    GateOp[DIM].vjp[BATCH](
        cpu_grad_out_t, cpu_grad_in_t, params_t, cpu_cache_t, grad_params_t
    )

    var gpu_grad_out_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
    for i in range(BATCH * DIM):
        gpu_grad_out_host[i] = grad_out_ptr[i]
    var gpu_grad_out_buf = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    ctx.enqueue_copy(gpu_grad_out_buf, gpu_grad_out_host)

    var gpu_grad_in_buf = ctx.enqueue_create_buffer[dtype](BATCH * IN)
    var gpu_grad_params_buf = ctx.enqueue_create_buffer[dtype](1)

    var gpu_grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](gpu_grad_out_buf.unsafe_ptr())
    var gpu_grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](gpu_grad_in_buf.unsafe_ptr())
    var gpu_grad_params_t = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](gpu_grad_params_buf.unsafe_ptr())

    GateOp[DIM].vjp_gpu[BATCH](
        ctx,
        gpu_grad_out_t,
        gpu_grad_in_t,
        gpu_params_t,
        gpu_cache_t,
        gpu_grad_params_t,
        ws,
    )

    var gpu_grad_in_host = ctx.enqueue_create_host_buffer[dtype](BATCH * IN)
    ctx.enqueue_copy(gpu_grad_in_host, gpu_grad_in_buf)
    ctx.synchronize()

    var bwd_diff = _max_diff(
        cpu_grad_in_ptr, gpu_grad_in_host.unsafe_ptr(), BATCH * IN
    )

    if fwd_diff < 1e-4 and bwd_diff < 1e-4:
        print(
            "  [PASS] Gate CPU/GPU parity: fwd_diff =",
            fwd_diff,
            "bwd_diff =",
            bwd_diff,
        )
    else:
        print(
            "  [FAIL] Gate CPU/GPU parity: fwd_diff =",
            fwd_diff,
            "bwd_diff =",
            bwd_diff,
        )


def main() raises:
    print("=== LeWM Phase 1 CPU/GPU Parity ===")
    print()
    comptime if not has_accelerator():
        print("No GPU available — skipping parity tests.")
        return

    var ctx = DeviceContext()
    print("--- ModulateOp ---")
    test_modulate_gpu_parity(ctx)
    print()
    print("--- GateOp ---")
    test_gate_gpu_parity(ctx)
    print()
    print("=== Phase 1 GPU parity done ===")
