"""SinusoidalPosAdd — CPU reference + GPU parity (Phase 1).

Verifies the precomputed bias matches the reference separable sinusoid
(pos_t[t] + pos_s[s]), the forward add, identity vjp, and CPU↔GPU parity.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import alloc
from std.math import abs, exp, sin, cos, log, sqrt
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.primitives.sinusoidal_pos import SinusoidalPosAdd


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def _mao(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _spread(i: Int, seed: Float64) -> Scalar[DT]:
    var x = seed + 0.7 * Float64(i)
    var t = x - 6.2831853 * Float64(Int(x / 6.2831853))
    return Scalar[DT](0.5 * (t - (t * t * t) / 6.0))


def _ref_val(pos: Int, j: Int, d: Int) -> Float64:
    var k = Float64(j // 2)
    var div = exp(-(2.0 * k) / Float64(d) * log(10000.0))
    var ang = Float64(pos) * div
    return sin(ang) if (j % 2) == 0 else cos(ang)


def test_forward_reference() raises:
    print("test_forward_reference ...")
    comptime T = 3
    comptime S = 4
    comptime D = 6
    comptime BATCH = 2
    comptime N = T * S * D
    var op = SinusoidalPosAdd[T, S, D, False].make[target="cpu", INIT=Zero]()
    var x = _alloc(BATCH * N)
    var y = _alloc(BATCH * N)
    for i in range(BATCH * N):
        x[i] = _spread(i, 1.0)
    var xt = TileTensor(x, row_major[BATCH, N]())
    var yt = TileTensor(y, row_major[BATCH, N]())
    op.forward["cpu", BATCH](xt, output=yt)

    var max_err: Float64 = 0.0
    for b in range(BATCH):
        for t in range(T):
            for s in range(S):
                for j in range(D):
                    var off = (t * S + s) * D + j
                    var bias = _ref_val(t, j, D) + _ref_val(s, j, D)
                    var want = Float64(x[b * N + off]) + bias
                    var e = abs(Float64(y[b * N + off]) - want)
                    if e > max_err:
                        max_err = e
    print("   max fwd err =", max_err)
    assert_true(max_err < 1e-5, "sinusoid forward reference")
    print("  ok")


def test_identity_vjp() raises:
    print("test_identity_vjp ...")
    comptime T = 2
    comptime S = 3
    comptime D = 4
    comptime BATCH = 2
    comptime N = T * S * D
    var op = SinusoidalPosAdd[T, S, D, True].make[target="cpu", INIT=Zero]()
    var go = _alloc(BATCH * N)
    var gi = _alloc(BATCH * N)
    for i in range(BATCH * N):
        go[i] = _spread(i, 2.5)
    var got = TileTensor(go, row_major[BATCH, N]())
    var git = TileTensor(gi, row_major[BATCH, N]())
    op.vjp["cpu", BATCH](got, git)
    var max_err: Float64 = 0.0
    for i in range(BATCH * N):
        var e = abs(Float64(gi[i]) - Float64(go[i]))
        if e > max_err:
            max_err = e
    print("   max vjp err =", max_err)
    assert_true(max_err < 1e-7, "sinusoid identity vjp")
    print("  ok")


def test_gpu_parity() raises:
    print("test_gpu_parity ...")
    comptime T = 3
    comptime S = 5
    comptime D = 8
    comptime BATCH = 2
    comptime N = T * S * D
    var ctx = DeviceContext()
    var cpu = SinusoidalPosAdd[T, S, D, True].make[target="cpu", INIT=Zero]()
    var gpu = SinusoidalPosAdd[T, S, D, True].make[target="gpu", INIT=Zero](ctx)

    var xh = ctx.enqueue_create_host_buffer[DT](BATCH * N)
    var yh = ctx.enqueue_create_host_buffer[DT](BATCH * N)
    ctx.synchronize()
    for i in range(BATCH * N):
        xh.unsafe_ptr()[i] = _spread(i, 0.3)

    var ycpu = _alloc(BATCH * N)
    var xc = TileTensor(xh.unsafe_ptr(), row_major[BATCH, N]())
    var yc = TileTensor(ycpu, row_major[BATCH, N]())
    cpu.forward["cpu", BATCH](xc, output=yc)

    var xd = ctx.enqueue_create_buffer[DT](BATCH * N)
    var yd = ctx.enqueue_create_buffer[DT](BATCH * N)
    ctx.enqueue_copy(xd, xh)
    ctx.synchronize()
    var xt = TileTensor(_mao(xd), row_major[BATCH, N]())
    var yt = TileTensor(_mao(yd), row_major[BATCH, N]())
    gpu.forward["gpu", BATCH](xt, output=yt)
    ctx.enqueue_copy(yh, yd)
    ctx.synchronize()

    var max_err: Float64 = 0.0
    for i in range(BATCH * N):
        var e = abs(Float64(yh.unsafe_ptr()[i]) - Float64(ycpu[i]))
        if e > max_err:
            max_err = e
    print("   max parity err =", max_err)
    assert_true(max_err < 1e-4, "sinusoid CPU<->GPU parity")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("SinusoidalPosAdd (Phase 1)")
    print("=" * 70)
    test_forward_reference()
    test_identity_vjp()
    test_gpu_parity()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
