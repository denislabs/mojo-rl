"""Dreamer 4 block primitives — CPU↔GPU parity (Phase 1).

    pixi run -e apple  mojo run -I . tests/nn2/test_dreamer4_prims_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/nn2/test_dreamer4_prims_gpu.mojo

SwiGLU (fwd+vjp) and SpaceTimeTranspose (fwd+vjp) vs the CPU paths.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import alloc
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.swiglu import SwiGLU
from mojo_rl.nn2.primitives.space_time_transpose import SpaceTimeTranspose


comptime TOL: Float64 = 1e-4


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def _mao(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _spread(i: Int, seed: Float64) -> Scalar[DT]:
    var x = seed + 0.7 * Float64(i)
    var t = x - 6.2831853 * Float64(Int(x / 6.2831853))
    return Scalar[DT](0.5 * (t - (t * t * t) / 6.0))


def _maxdiff(
    a: UnsafePointer[Scalar[DT], MutAnyOrigin],
    b: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n: Int,
) -> Float64:
    var m: Float64 = 0.0
    for i in range(n):
        var d = abs(Float64(a[i]) - Float64(b[i]))
        if d > m:
            m = d
    return m


def test_swiglu_parity(ctx: DeviceContext) raises:
    print("swiglu_parity ...")
    comptime HIDDEN = 6
    comptime BATCH = 3
    comptime IN_N = BATCH * 2 * HIDDEN
    comptime OUT_N = BATCH * HIDDEN

    var cpu = SwiGLU[HIDDEN].make[target="cpu", INIT=Zero]()
    var gpu = SwiGLU[HIDDEN].make[target="gpu", INIT=Zero](ctx)

    var xh = ctx.enqueue_create_host_buffer[DT](IN_N)
    var goh = ctx.enqueue_create_host_buffer[DT](OUT_N)
    var yh = ctx.enqueue_create_host_buffer[DT](OUT_N)
    var gih = ctx.enqueue_create_host_buffer[DT](IN_N)
    ctx.synchronize()
    for i in range(IN_N):
        xh.unsafe_ptr()[i] = _spread(i, 1.3)
    for i in range(OUT_N):
        goh.unsafe_ptr()[i] = _spread(i, 4.1)

    var ycpu = _alloc(OUT_N)
    var gicpu = _alloc(IN_N)
    var xc = TileTensor(xh.unsafe_ptr(), row_major[BATCH, 2 * HIDDEN]())
    var yc = TileTensor(ycpu, row_major[BATCH, HIDDEN]())
    cpu.forward["cpu", BATCH](xc, output=yc)
    var goc = TileTensor(goh.unsafe_ptr(), row_major[BATCH, HIDDEN]())
    var gic = TileTensor(gicpu, row_major[BATCH, 2 * HIDDEN]())
    cpu.vjp["cpu", BATCH](goc, gic)

    var xd = ctx.enqueue_create_buffer[DT](IN_N)
    var yd = ctx.enqueue_create_buffer[DT](OUT_N)
    var god = ctx.enqueue_create_buffer[DT](OUT_N)
    var gid = ctx.enqueue_create_buffer[DT](IN_N)
    ctx.enqueue_copy(xd, xh)
    ctx.enqueue_copy(god, goh)
    ctx.synchronize()
    var xt = TileTensor(_mao(xd), row_major[BATCH, 2 * HIDDEN]())
    var yt = TileTensor(_mao(yd), row_major[BATCH, HIDDEN]())
    gpu.forward["gpu", BATCH](xt, output=yt)
    var got = TileTensor(_mao(god), row_major[BATCH, HIDDEN]())
    var git = TileTensor(_mao(gid), row_major[BATCH, 2 * HIDDEN]())
    gpu.vjp["gpu", BATCH](got, git)
    ctx.enqueue_copy(yh, yd)
    ctx.enqueue_copy(gih, gid)
    ctx.synchronize()

    var mf = _maxdiff(yh.unsafe_ptr(), ycpu, OUT_N)
    var mb = _maxdiff(gih.unsafe_ptr(), gicpu, IN_N)
    print("   fwd diff =", mf, "  grad diff =", mb)
    assert_true(mf < TOL and mb < TOL, "swiglu parity")
    print("  ok")


def test_stt_parity(ctx: DeviceContext) raises:
    print("stt_parity ...")
    comptime T = 3
    comptime S = 5
    comptime D = 4
    comptime BATCH = 2
    comptime N = BATCH * T * S * D

    var cpu = SpaceTimeTranspose[T, S, D].make[target="cpu", INIT=Zero]()
    var gpu = SpaceTimeTranspose[T, S, D].make[target="gpu", INIT=Zero](ctx)

    var xh = ctx.enqueue_create_host_buffer[DT](N)
    var goh = ctx.enqueue_create_host_buffer[DT](N)
    var yh = ctx.enqueue_create_host_buffer[DT](N)
    var gih = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()
    for i in range(N):
        xh.unsafe_ptr()[i] = _spread(i, 0.9)
        goh.unsafe_ptr()[i] = _spread(i, 2.2)

    var ycpu = _alloc(N)
    var gicpu = _alloc(N)
    var xc = TileTensor(xh.unsafe_ptr(), row_major[BATCH, T * S * D]())
    var yc = TileTensor(ycpu, row_major[BATCH, T * S * D]())
    cpu.forward["cpu", BATCH](xc, output=yc)
    var goc = TileTensor(goh.unsafe_ptr(), row_major[BATCH, T * S * D]())
    var gic = TileTensor(gicpu, row_major[BATCH, T * S * D]())
    cpu.vjp["cpu", BATCH](goc, gic)

    var xd = ctx.enqueue_create_buffer[DT](N)
    var yd = ctx.enqueue_create_buffer[DT](N)
    var god = ctx.enqueue_create_buffer[DT](N)
    var gid = ctx.enqueue_create_buffer[DT](N)
    ctx.enqueue_copy(xd, xh)
    ctx.enqueue_copy(god, goh)
    ctx.synchronize()
    var xt = TileTensor(_mao(xd), row_major[BATCH, T * S * D]())
    var yt = TileTensor(_mao(yd), row_major[BATCH, T * S * D]())
    gpu.forward["gpu", BATCH](xt, output=yt)
    var got = TileTensor(_mao(god), row_major[BATCH, T * S * D]())
    var git = TileTensor(_mao(gid), row_major[BATCH, T * S * D]())
    gpu.vjp["gpu", BATCH](got, git)
    ctx.enqueue_copy(yh, yd)
    ctx.enqueue_copy(gih, gid)
    ctx.synchronize()

    var mf = _maxdiff(yh.unsafe_ptr(), ycpu, N)
    var mb = _maxdiff(gih.unsafe_ptr(), gicpu, N)
    print("   fwd diff =", mf, "  grad diff =", mb)
    assert_true(mf < TOL and mb < TOL, "stt parity")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Dreamer 4 block primitives CPU↔GPU parity (Phase 1)")
    print("=" * 70)
    var ctx = DeviceContext()
    test_swiglu_parity(ctx)
    test_stt_parity(ctx)
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
