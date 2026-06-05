"""SIGReg gradcheck (Phase A) — CPU and GPU, each self-consistent.

CPU and GPU generate the random projection A differently (different PRNG
draw scheme + pointer-derived seed), so cross-target value parity is not
meaningful (same as the legacy `nn` op). Instead we gradcheck each target
against its OWN forward via central finite differences:

    L = sum_b w[b]·output[b,0] = G·stat,   G = sum_b w[b]
    analytic grad_input = vjp(w);   numeric = d(G·stat)/d input[k].

Run GPU side with:  pixi run -e apple mojo run -I . tests/nn2/test_sigreg.mojo
"""

from std.memory import alloc
from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.primitives.sigreg import SIGReg


comptime EPS: Scalar[DT] = 2e-3
comptime ATOL: Scalar[DT] = 5e-4
comptime RTOL: Scalar[DT] = 2e-2

comptime DIM = 4
comptime SEQ = 2
comptime PROJ = 4
comptime KN = 5
comptime BATCH = 4
comptime IN = SEQ * DIM
comptime N = BATCH * IN


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def _p(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _det(i: Int, scale: Float64) -> Scalar[DT]:
    var v = (Float64((i * 2654435761) % 1000) / 500.0) - 1.0
    return Scalar[DT](v * scale)


def _ok(a: Scalar[DT], b: Scalar[DT]) -> Bool:
    var ad = (a - b).__abs__()
    if ad < ATOL:
        return True
    return (ad / (a.__abs__() + b.__abs__() + Scalar[DT](1e-4))) < RTOL


def test_sigreg_cpu_gradcheck() raises:
    print("test_sigreg_cpu_gradcheck ...")
    var x = _a(N); var y = _a(BATCH); var w = _a(BATCH); var gx = _a(N)
    for k in range(N):
        x[k] = _det(k + 1, 1.0)
    for b in range(BATCH):
        w[b] = Scalar[DT](0.1 * Float64(b + 1))

    var m = SIGReg[DIM, SEQ, PROJ, KN].make[target="cpu", INIT=Kaiming]()
    var x_t = TileTensor(x, row_major[BATCH, IN]())
    var y_t = TileTensor(y, row_major[BATCH, 1]())
    m.forward["cpu", BATCH](x_t, output=y_t)

    # stat >= 0 and replicated across rows.
    assert_true(y[0] >= Scalar[DT](0.0), "SIGReg stat must be >= 0")
    for b in range(BATCH):
        assert_true((y[b] - y[0]).__abs__() < Scalar[DT](1e-6),
                    "SIGReg stat must be replicated")

    var w_t = TileTensor(w, row_major[BATCH, 1]())
    var gx_t = TileTensor(gx, row_major[BATCH, IN]())
    m.vjp["cpu", BATCH](w_t, gx_t)

    var G: Scalar[DT] = 0.0
    for b in range(BATCH):
        G += w[b]

    for k in range(N):
        var saved = x[k]
        x[k] = saved + EPS
        m.forward["cpu", BATCH](x_t, output=y_t)
        var lp = G * y[0]
        x[k] = saved - EPS
        m.forward["cpu", BATCH](x_t, output=y_t)
        var lm = G * y[0]
        x[k] = saved
        var num = (lp - lm) / (Scalar[DT](2.0) * EPS)
        assert_true(_ok(gx[k], num), "SIGReg CPU grad fd mismatch")

    x.free(); y.free(); w.free(); gx.free()
    print("  ok")


def test_sigreg_gpu_gradcheck() raises:
    print("test_sigreg_gpu_gradcheck ...")
    var ctx = DeviceContext()
    var x = _a(N); var w = _a(BATCH)
    for k in range(N):
        x[k] = _det(k + 1, 1.0)
    for b in range(BATCH):
        w[b] = Scalar[DT](0.1 * Float64(b + 1))
    var G: Scalar[DT] = 0.0
    for b in range(BATCH):
        G += w[b]

    var x_d = ctx.enqueue_create_buffer[DT](N)
    var y_d = ctx.enqueue_create_buffer[DT](BATCH)
    var w_d = ctx.enqueue_create_buffer[DT](BATCH)
    var gx_d = ctx.enqueue_create_buffer[DT](N)
    var xh = ctx.enqueue_create_host_buffer[DT](N)
    var yh = ctx.enqueue_create_host_buffer[DT](BATCH)
    var wh = ctx.enqueue_create_host_buffer[DT](BATCH)
    var gxh = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()

    for b in range(BATCH):
        wh.unsafe_ptr()[b] = w[b]
    ctx.enqueue_copy(w_d, wh)

    var m = SIGReg[DIM, SEQ, PROJ, KN].make[target="gpu", INIT=Kaiming](ctx)
    var x_t = TileTensor(_p(x_d), row_major[BATCH, IN]())
    var y_t = TileTensor(_p(y_d), row_major[BATCH, 1]())
    var w_t = TileTensor(_p(w_d), row_major[BATCH, 1]())
    var gx_t = TileTensor(_p(gx_d), row_major[BATCH, IN]())

    @parameter
    def fwd_stat() raises -> Scalar[DT]:
        for k in range(N):
            xh.unsafe_ptr()[k] = x[k]
        ctx.enqueue_copy(x_d, xh)
        m.forward["gpu", BATCH](x_t, output=y_t)
        ctx.enqueue_copy(yh, y_d)
        ctx.synchronize()
        return yh.unsafe_ptr()[0]

    var s0 = fwd_stat()
    assert_true(s0 >= Scalar[DT](-1e-5), "SIGReg GPU stat >= 0")

    # analytic grad at base point (forward already ran on base x).
    m.vjp["gpu", BATCH](w_t, gx_t)
    ctx.enqueue_copy(gxh, gx_d)
    ctx.synchronize()

    for k in range(N):
        var saved = x[k]
        x[k] = saved + EPS
        var lp = G * fwd_stat()
        x[k] = saved - EPS
        var lm = G * fwd_stat()
        x[k] = saved
        var num = (lp - lm) / (Scalar[DT](2.0) * EPS)
        assert_true(_ok(gxh.unsafe_ptr()[k], num),
                    "SIGReg GPU grad fd mismatch")

    print("  ok")


def main() raises:
    print("=" * 70)
    print("SIGReg gradcheck (Phase A)")
    print("=" * 70)
    test_sigreg_cpu_gradcheck()
    test_sigreg_gpu_gradcheck()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
