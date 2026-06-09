"""u8_hwc_to_chw_norm CPU↔GPU parity (Phase E real-pixel path).

Checks the HWC uint8 → CHW fp32 ÷255 conversion is bitwise-identical on
CPU and GPU, and that a hand-decoded reference matches both.

Run:  pixi run -e apple mojo run -I . tests/experimental/lewm2/test_pixel_convert_gpu.mojo
"""

from std.memory import alloc
from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.experimental.lewm2.pixel_convert import (
    u8_hwc_to_chw_norm,
    u8_to_fp32_norm,
)


comptime C = 4
comptime FH = 8
comptime FW = 6
comptime BATCH = 3
comptime N = BATCH * C * FH * FW


def _u8p(b: DeviceBuffer[DType.uint8]) -> UnsafePointer[
    Scalar[DType.uint8], MutAnyOrigin
]:
    return rebind[UnsafePointer[Scalar[DType.uint8], MutAnyOrigin]](
        b.unsafe_ptr()
    )


def _fp(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def main() raises:
    print("=" * 70)
    print("u8_hwc_to_chw_norm CPU↔GPU parity")
    print("=" * 70)
    var ctx = DeviceContext()

    # host source frame (uint8 HWC)
    var src_h = ctx.enqueue_create_host_buffer[DType.uint8](N)
    ctx.synchronize()
    for k in range(N):
        src_h.unsafe_ptr()[k] = UInt8((k * 37 + 11) % 256)

    # ── CPU reference
    var cpu_dst: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var src_host_ptr = rebind[UnsafePointer[Scalar[DType.uint8], MutAnyOrigin]](
        src_h.unsafe_ptr()
    )
    u8_hwc_to_chw_norm["cpu", C, FH, FW, BATCH](src_host_ptr, cpu_dst)

    # ── hand-decoded oracle (one element)
    # dst[(b,c,h,w)] == src[(b, h,w,c)] / 255
    var b0 = 1; var c0 = 2; var h0 = 3; var w0 = 4
    comptime HWC = C * FH * FW
    var d_idx = b0 * HWC + (c0 * FH + h0) * FW + w0
    var s_idx = b0 * HWC + (h0 * FW + w0) * C + c0
    var oracle = Scalar[DT](Float64(Int(src_h.unsafe_ptr()[s_idx]))) / 255.0
    print("   oracle dst[",d_idx,"]=", oracle, " cpu=", cpu_dst[d_idx])
    assert_true((cpu_dst[d_idx] - oracle).__abs__() < Scalar[DT](1e-7),
                "CPU matches hand-decoded oracle")

    # ── GPU
    var src_d = ctx.enqueue_create_buffer[DType.uint8](N)
    var dst_d = ctx.enqueue_create_buffer[DT](N)
    ctx.enqueue_copy(src_d, src_h)
    ctx.synchronize()
    u8_hwc_to_chw_norm["gpu", C, FH, FW, BATCH](
        _u8p(src_d), _fp(dst_d), ctx=ctx
    )
    var gpu_h = ctx.enqueue_create_host_buffer[DT](N)
    ctx.enqueue_copy(gpu_h, dst_d)
    ctx.synchronize()

    var maxd: Scalar[DT] = 0.0
    for k in range(N):
        var d = (gpu_h.unsafe_ptr()[k] - cpu_dst[k]).__abs__()
        if d > maxd:
            maxd = d
    print("   max|gpu - cpu| =", maxd)
    assert_true(maxd < Scalar[DT](1e-7), "GPU bitwise-matches CPU")

    cpu_dst.free()

    # ── layout-preserving u8_to_fp32_norm (the CHW Pong path) ──────────
    print("u8_to_fp32_norm (layout-preserving) CPU↔GPU ...")
    var lp_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    u8_to_fp32_norm["cpu", N](src_host_ptr, lp_cpu)
    var lp_dst_d = ctx.enqueue_create_buffer[DT](N)
    u8_to_fp32_norm["gpu", N](_u8p(src_d), _fp(lp_dst_d), ctx=ctx)
    var lp_gpu_h = ctx.enqueue_create_host_buffer[DT](N)
    ctx.enqueue_copy(lp_gpu_h, lp_dst_d)
    ctx.synchronize()
    var lp_maxd: Scalar[DT] = 0.0
    var lp_oracle_ok = True
    for k in range(N):
        var d = (lp_gpu_h.unsafe_ptr()[k] - lp_cpu[k]).__abs__()
        if d > lp_maxd:
            lp_maxd = d
        var want = Scalar[DT](Float64(Int(src_h.unsafe_ptr()[k]))) / 255.0
        if (lp_cpu[k] - want).__abs__() > Scalar[DT](1e-7):
            lp_oracle_ok = False
    print("   max|gpu - cpu| =", lp_maxd)
    assert_true(lp_maxd < Scalar[DT](1e-7), "u8_to_fp32_norm GPU==CPU")
    assert_true(lp_oracle_ok, "u8_to_fp32_norm == x/255 elementwise")
    lp_cpu.free()

    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
