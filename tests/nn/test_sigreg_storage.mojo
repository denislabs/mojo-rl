"""SIGReg storage primitive — CPU and GPU gradcheck, each self-consistent.

Standalone storage test (no legacy oracle — converted from the former
`_storage_parity` test in legacy-removal Phase 0b). SIGReg's random projection A
is derived from the cache_z BUFFER ADDRESS, so two DIFFERENT instances (CPU vs
GPU, or any two constructions) draw DIFFERENT A and thus DIFFERENT statistics —
value parity across instances is NOT meaningful, and golden fingerprints are not
reproducible. So, exactly like the legacy `tests/nn/test_sigreg.mojo`, we
gradcheck EACH target of the STORAGE op against its OWN forward via central
finite differences:

    L = sum_b w[b]·out[b,0] = G·stat,   G = sum_b w[b]
    analytic grad_input = vjp(w);   numeric = d(G·stat)/d input[k].

We also assert the transform semantics the consumer relies on: stat >= 0 and
replicated across rows.

Run:
  rm -f mojo_rl.mojoc && pixi run mojo run -I . tests/nn/test_sigreg_storage.mojo
  rm -f mojo_rl.mojoc && pixi run -e apple mojo run -I . tests/nn/test_sigreg_storage.mojo
"""

from max.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.primitives.sigreg import SIGReg


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


def _det(i: Int, scale: Float64) -> Scalar[DT]:
    var v = (Float64((i * 2654435761) % 1000) / 500.0) - 1.0
    return Scalar[DT](v * scale)


def _ok(a: Scalar[DT], b: Scalar[DT]) -> Bool:
    var ad = (a - b).__abs__()
    if ad < ATOL:
        return True
    return (ad / (a.__abs__() + b.__abs__() + Scalar[DT](1e-4))) < RTOL


def test_sigreg_storage_cpu_gradcheck() raises:
    print("test_sigreg_storage_cpu_gradcheck ...")
    var x = Tensor.alloc(N)
    var y = Tensor.alloc(BATCH)
    var w = Tensor.alloc(BATCH)
    var gx = Tensor.alloc(N)
    for k in range(N):
        x.data[k] = _det(k + 1, 1.0)
    var G: Scalar[DT] = 0.0
    for b in range(BATCH):
        w.data[b] = Scalar[DT](0.1 * Float64(b + 1))
        G += w.data[b]

    var m = SIGReg[DIM, SEQ, PROJ, KN].make[target="cpu", INIT=Kaiming]()

    @parameter
    def fwd() raises:
        m.forward["cpu", BATCH](TensorRefs[1](x), out=y)

    fwd()
    # stat >= 0 and replicated across rows.
    assert_true(y.data[0] >= Scalar[DT](0.0), "SIGReg stat must be >= 0")
    for b in range(BATCH):
        assert_true(
            (y.data[b] - y.data[0]).__abs__() < Scalar[DT](1e-6),
            "SIGReg stat must be replicated",
        )

    m.vjp["cpu", BATCH](
        TensorRefs[1](x),
        grad_output=w,
        grad_inputs=TensorRefs[1](gx),
    )

    for k in range(N):
        var saved = x.data[k]
        x.data[k] = saved + EPS
        fwd()
        var lp = G * y.data[0]
        x.data[k] = saved - EPS
        fwd()
        var lm = G * y.data[0]
        x.data[k] = saved
        var num = (lp - lm) / (Scalar[DT](2.0) * EPS)
        assert_true(_ok(gx.data[k], num), "SIGReg storage CPU grad fd mismatch")

    print("  ok")


def test_sigreg_storage_gpu_gradcheck() raises:
    print("test_sigreg_storage_gpu_gradcheck ...")
    var ctx = DeviceContext()
    # Host-side x driving the device buffer + analytic readback.
    var xh = List[Scalar[DT]](length=N, fill=Scalar[DT](0))
    for k in range(N):
        xh[k] = _det(k + 1, 1.0)
    var G: Scalar[DT] = 0.0
    for b in range(BATCH):
        G += Scalar[DT](0.1 * Float64(b + 1))

    var x = Tensor.alloc_gpu(ctx, N)
    var y = Tensor.alloc_gpu(ctx, BATCH)
    var w = Tensor.alloc_gpu(ctx, BATCH)
    var gx = Tensor.alloc_gpu(ctx, N)
    # seed w on device.
    w.data = List[Scalar[DT]](length=BATCH, fill=Scalar[DT](0))
    for b in range(BATCH):
        w.data[b] = Scalar[DT](0.1 * Float64(b + 1))
    w.n = BATCH
    w.upload(ctx)

    var m = SIGReg[DIM, SEQ, PROJ, KN].make[target="gpu", INIT=Kaiming](ctx)

    @parameter
    def fwd_stat() raises -> Scalar[DT]:
        # push xh → x.dev, run forward, read back y[0].
        x.data = xh.copy()
        x.n = N
        x.upload(ctx)
        m.forward["gpu", BATCH](TensorRefs[1](x), out=y, ctx=Optional(ctx))
        y.download(ctx)
        return y.data[0]

    var s0 = fwd_stat()
    assert_true(s0 >= Scalar[DT](-1e-5), "SIGReg GPU stat >= 0")

    # analytic grad at base point (forward already ran on base x).
    m.vjp["gpu", BATCH](
        TensorRefs[1](x),
        grad_output=w,
        grad_inputs=TensorRefs[1](gx),
        ctx=Optional(ctx),
    )
    gx.download(ctx)

    for k in range(N):
        var saved = xh[k]
        xh[k] = saved + EPS
        var lp = G * fwd_stat()
        xh[k] = saved - EPS
        var lm = G * fwd_stat()
        xh[k] = saved
        var num = (lp - lm) / (Scalar[DT](2.0) * EPS)
        assert_true(
            _ok(gx.data[k], num), "SIGReg storage GPU grad fd mismatch"
        )

    print("  ok")


def main() raises:
    print("=" * 70)
    print("SIGReg storage primitive (CPU + GPU gradcheck)")
    print("=" * 70)
    test_sigreg_storage_cpu_gradcheck()
    test_sigreg_storage_gpu_gradcheck()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
