"""Dreamer 4 encoder/decoder I/O leaves — CPU (+GPU for pos) (Phase 1).

SinusoidalPosAddBT: B·T-layout position reference + GPU parity.
LearnedTokens: forward placement, input gradcheck, and param gradcheck (FD
over the learned-token parameter), for both prepend (encoder) and append
(decoder) modes.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import alloc
from std.math import abs, exp, sin, cos, log
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Zero, Xavier
from mojo_rl.nn.primitives.sinusoidal_pos_bt import SinusoidalPosAddBT
from mojo_rl.nn.primitives.learned_tokens import LearnedTokens


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def _mao(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _spread(i: Int, seed: Float64) -> Scalar[DT]:
    var x = seed + 0.7 * Float64(i)
    var t = x - 6.2831853 * Float64(Int(x / 6.2831853))
    return Scalar[DT](0.5 * (t - (t * t * t) / 6.0))


def _refpos(pos: Int, j: Int, d: Int) -> Float64:
    var k = Float64(j // 2)
    var div = exp(-(2.0 * k) / Float64(d) * log(10000.0))
    var ang = Float64(pos) * div
    return sin(ang) if (j % 2) == 0 else cos(ang)


def test_pos_bt_reference() raises:
    print("pos_bt reference + gpu parity ...")
    comptime T = 3
    comptime S = 4
    comptime D = 6
    comptime B = 2
    comptime BATCH = B * T
    comptime SD = S * D
    var ctx = DeviceContext()
    var cpu = SinusoidalPosAddBT[T, S, D, False].make[target="cpu", INIT=Zero]()
    var gpu = SinusoidalPosAddBT[T, S, D, False].make[target="gpu", INIT=Zero](ctx)

    var xh = ctx.enqueue_create_host_buffer[DT](BATCH * SD)
    var yh = ctx.enqueue_create_host_buffer[DT](BATCH * SD)
    ctx.synchronize()
    for i in range(BATCH * SD):
        xh.unsafe_ptr()[i] = _spread(i, 1.0)

    var yc = _alloc(BATCH * SD)
    var xc = TileTensor(xh.unsafe_ptr(), row_major[BATCH, SD]())
    var yct = TileTensor(yc, row_major[BATCH, SD]())
    cpu.forward["cpu", BATCH](xc, output=yct)

    var max_ref: Float64 = 0.0
    for bt in range(BATCH):
        var t = bt % T
        for s in range(S):
            for j in range(D):
                var want = Float64(xh.unsafe_ptr()[bt * SD + s * D + j]) + (
                    _refpos(t, j, D) + _refpos(s, j, D)
                )
                var e = abs(Float64(yc[bt * SD + s * D + j]) - want)
                if e > max_ref:
                    max_ref = e
    assert_true(max_ref < 1e-5, "pos_bt reference")

    var xd = ctx.enqueue_create_buffer[DT](BATCH * SD)
    var yd = ctx.enqueue_create_buffer[DT](BATCH * SD)
    ctx.enqueue_copy(xd, xh)
    ctx.synchronize()
    var xt = TileTensor(_mao(xd), row_major[BATCH, SD]())
    var yt = TileTensor(_mao(yd), row_major[BATCH, SD]())
    gpu.forward["gpu", BATCH](xt, output=yt)
    ctx.enqueue_copy(yh, yd)
    ctx.synchronize()
    var max_par: Float64 = 0.0
    for i in range(BATCH * SD):
        var e = abs(Float64(yh.unsafe_ptr()[i]) - Float64(yc[i]))
        if e > max_par:
            max_par = e
    print("   ref err =", max_ref, "  gpu parity =", max_par)
    assert_true(max_par < 1e-4, "pos_bt gpu parity")
    print("  ok")


def _test_learned_tokens[PREPEND: Bool](name: String) raises:
    print(name, "...")
    comptime N_IN = 3
    comptime N_NEW = 2
    comptime D = 4
    comptime BATCH = 5
    comptime IN_N = BATCH * N_IN * D
    comptime OUT_N = BATCH * (N_IN + N_NEW) * D
    comptime NEW_OFF = 0 if PREPEND else N_IN * D
    comptime IN_OFF = N_NEW * D if PREPEND else 0

    var op = LearnedTokens[N_IN, N_NEW, D, PREPEND].make[
        target="cpu", INIT=Xavier
    ]()
    var x = _alloc(IN_N)
    var y = _alloc(OUT_N)
    var go = _alloc(OUT_N)
    var gi = _alloc(IN_N)
    for i in range(IN_N):
        x[i] = _spread(i, 1.5)
    for i in range(OUT_N):
        go[i] = _spread(i, 0.7)
    var xt = TileTensor(x, row_major[BATCH, N_IN * D]())
    var yt = TileTensor(y, row_major[BATCH, (N_IN + N_NEW) * D]())
    op.forward["cpu", BATCH](xt, output=yt)

    # forward placement: input is at IN_OFF; the same learned token block
    # appears in every sample at NEW_OFF.
    var max_place: Float64 = 0.0
    for bt in range(BATCH):
        for k in range(N_IN * D):
            var e = abs(
                Float64(y[bt * (N_IN + N_NEW) * D + IN_OFF + k])
                - Float64(x[bt * N_IN * D + k])
            )
            if e > max_place:
                max_place = e
        for k in range(N_NEW * D):  # learned block identical across samples
            var e = abs(
                Float64(y[bt * (N_IN + N_NEW) * D + NEW_OFF + k])
                - Float64(y[NEW_OFF + k])
            )
            if e > max_place:
                max_place = e
    assert_true(max_place < 1e-7, name + ": forward placement")

    op.zero_grad["cpu"]()
    var got = TileTensor(go, row_major[BATCH, (N_IN + N_NEW) * D]())
    var git = TileTensor(gi, row_major[BATCH, N_IN * D]())
    op.vjp["cpu", BATCH](got, git)

    # input gradcheck
    var max_in: Float64 = 0.0
    for kk in range(IN_N):
        var orig = x[kk]
        x[kk] = orig + Scalar[DT](1e-3)
        op.forward["cpu", BATCH](xt, output=yt)
        var lp: Float64 = 0.0
        for i in range(OUT_N):
            lp += Float64(y[i]) * Float64(go[i])
        x[kk] = orig - Scalar[DT](1e-3)
        op.forward["cpu", BATCH](xt, output=yt)
        var lm: Float64 = 0.0
        for i in range(OUT_N):
            lm += Float64(y[i]) * Float64(go[i])
        x[kk] = orig
        var fd = (lp - lm) / (2.0 * 1e-3)
        var d = abs(Float64(gi[kk]) - fd)
        if d > max_in:
            max_in = d
    assert_true(max_in < 1e-2, name + ": input gradcheck")

    # param gradcheck (FD over the learned-token parameter)
    var tp = op.tokens.value_unsafe_ptr_cpu()
    var gtok = op.tokens.grd.cpu.unsafe_ptr()
    var max_p: Float64 = 0.0
    for kk in range(N_NEW * D):
        var orig = tp[kk]
        tp[kk] = orig + Scalar[DT](1e-3)
        op.forward["cpu", BATCH](xt, output=yt)
        var lp: Float64 = 0.0
        for i in range(OUT_N):
            lp += Float64(y[i]) * Float64(go[i])
        tp[kk] = orig - Scalar[DT](1e-3)
        op.forward["cpu", BATCH](xt, output=yt)
        var lm: Float64 = 0.0
        for i in range(OUT_N):
            lm += Float64(y[i]) * Float64(go[i])
        tp[kk] = orig
        var fd = (lp - lm) / (2.0 * 1e-3)
        var d = abs(Float64(gtok[kk]) - fd)
        if d > max_p:
            max_p = d
    print("   place =", max_place, " in_grad =", max_in, " param_grad =", max_p)
    assert_true(max_p < 1e-2, name + ": param gradcheck")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Dreamer 4 encoder/decoder I/O leaves (Phase 1)")
    print("=" * 70)
    test_pos_bt_reference()
    _test_learned_tokens[True]("learned_tokens prepend (encoder)")
    _test_learned_tokens[False]("learned_tokens append (decoder)")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
