"""Wave A transformer primitives — CPU↔GPU parity.

Runs Transpose2D / TokenMean / BiasAdd / Embedding forward + vjp on GPU
and compares against the CPU path (params set identically on both). Run:

    pixi run -e apple mojo run -I . tests/nn2/test_transformer_primitives_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/nn2/test_transformer_primitives_gpu.mojo

Tolerance 1e-5 (fp32). Docs: docs/NN2_TRANSFORMER_PORT.md Phase 1 Wave A.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import alloc
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.transpose_2d import Transpose2D
from mojo_rl.nn2.primitives.token_mean import TokenMean
from mojo_rl.nn2.primitives.bias_add import BiasAdd
from mojo_rl.nn2.primitives.embedding import Embedding


comptime TOL: Float64 = 1e-5


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](n)
    )


def _mao(
    b: DeviceBuffer[DT],
) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _spread(i: Int, seed: Float64) -> Scalar[DT]:
    var x = seed + 0.7 * Float64(i)
    var t = x - 6.2831853 * Float64(Int(x / 6.2831853))
    return Scalar[DT](0.37 * (t - (t * t * t) / 6.0))


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


def test_transpose2d_parity(ctx: DeviceContext) raises:
    print("test_transpose2d_parity ...")
    comptime BATCH = 2
    comptime A = 3
    comptime B = 4
    comptime N = BATCH * A * B

    var cpu = Transpose2D[A, B].make[target="cpu", INIT=Zero]()
    var gpu = Transpose2D[A, B].make[target="gpu", INIT=Zero](ctx)

    var xh = ctx.enqueue_create_host_buffer[DT](N)
    var yh = ctx.enqueue_create_host_buffer[DT](N)
    var goh = ctx.enqueue_create_host_buffer[DT](N)
    var gih = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()
    for i in range(N):
        xh.unsafe_ptr()[i] = _spread(i, 1.0)
        goh.unsafe_ptr()[i] = _spread(i, 5.0)

    # CPU.
    var ycpu = _alloc(N)
    var gicpu = _alloc(N)
    var xc = TileTensor(xh.unsafe_ptr(), row_major[BATCH, A * B]())
    var yc = TileTensor(ycpu, row_major[BATCH, A * B]())
    cpu.forward["cpu", BATCH](xc, output=yc)
    var goc = TileTensor(goh.unsafe_ptr(), row_major[BATCH, A * B]())
    var gic = TileTensor(gicpu, row_major[BATCH, A * B]())
    cpu.vjp["cpu", BATCH](goc, gic)

    # GPU.
    var xd = ctx.enqueue_create_buffer[DT](N)
    var yd = ctx.enqueue_create_buffer[DT](N)
    var god = ctx.enqueue_create_buffer[DT](N)
    var gid = ctx.enqueue_create_buffer[DT](N)
    ctx.enqueue_copy(xd, xh)
    ctx.enqueue_copy(god, goh)
    ctx.synchronize()
    var xt = TileTensor(_mao(xd), row_major[BATCH, A * B]())
    var yt = TileTensor(_mao(yd), row_major[BATCH, A * B]())
    gpu.forward["gpu", BATCH](xt, output=yt)
    var got = TileTensor(_mao(god), row_major[BATCH, A * B]())
    var git = TileTensor(_mao(gid), row_major[BATCH, A * B]())
    gpu.vjp["gpu", BATCH](got, git)
    ctx.enqueue_copy(yh, yd)
    ctx.enqueue_copy(gih, gid)
    ctx.synchronize()

    var mf = _maxdiff(yh.unsafe_ptr(), ycpu, N)
    var mb = _maxdiff(gih.unsafe_ptr(), gicpu, N)
    print("   fwd diff =", mf, "  bwd diff =", mb)
    assert_true(mf < TOL, "Transpose2D fwd CPU/GPU parity")
    assert_true(mb < TOL, "Transpose2D bwd CPU/GPU parity")
    print("  ok")


def test_token_mean_parity(ctx: DeviceContext) raises:
    print("test_token_mean_parity ...")
    comptime BATCH = 2
    comptime SEQ = 5
    comptime DIM = 3
    comptime IN_N = BATCH * SEQ * DIM
    comptime OUT_N = BATCH * DIM

    var cpu = TokenMean[SEQ, DIM].make[target="cpu", INIT=Zero]()
    var gpu = TokenMean[SEQ, DIM].make[target="gpu", INIT=Zero](ctx)

    var xh = ctx.enqueue_create_host_buffer[DT](IN_N)
    var yh = ctx.enqueue_create_host_buffer[DT](OUT_N)
    var goh = ctx.enqueue_create_host_buffer[DT](OUT_N)
    var gih = ctx.enqueue_create_host_buffer[DT](IN_N)
    ctx.synchronize()
    for i in range(IN_N):
        xh.unsafe_ptr()[i] = _spread(i, 2.0)
    for i in range(OUT_N):
        goh.unsafe_ptr()[i] = _spread(i, 9.0)

    var ycpu = _alloc(OUT_N)
    var gicpu = _alloc(IN_N)
    var xc = TileTensor(xh.unsafe_ptr(), row_major[BATCH, SEQ * DIM]())
    var yc = TileTensor(ycpu, row_major[BATCH, DIM]())
    cpu.forward["cpu", BATCH](xc, output=yc)
    var goc = TileTensor(goh.unsafe_ptr(), row_major[BATCH, DIM]())
    var gic = TileTensor(gicpu, row_major[BATCH, SEQ * DIM]())
    cpu.vjp["cpu", BATCH](goc, gic)

    var xd = ctx.enqueue_create_buffer[DT](IN_N)
    var yd = ctx.enqueue_create_buffer[DT](OUT_N)
    var god = ctx.enqueue_create_buffer[DT](OUT_N)
    var gid = ctx.enqueue_create_buffer[DT](IN_N)
    ctx.enqueue_copy(xd, xh)
    ctx.enqueue_copy(god, goh)
    ctx.synchronize()
    var xt = TileTensor(_mao(xd), row_major[BATCH, SEQ * DIM]())
    var yt = TileTensor(_mao(yd), row_major[BATCH, DIM]())
    gpu.forward["gpu", BATCH](xt, output=yt)
    var got = TileTensor(_mao(god), row_major[BATCH, DIM]())
    var git = TileTensor(_mao(gid), row_major[BATCH, SEQ * DIM]())
    gpu.vjp["gpu", BATCH](got, git)
    ctx.enqueue_copy(yh, yd)
    ctx.enqueue_copy(gih, gid)
    ctx.synchronize()

    var mf = _maxdiff(yh.unsafe_ptr(), ycpu, OUT_N)
    var mb = _maxdiff(gih.unsafe_ptr(), gicpu, IN_N)
    print("   fwd diff =", mf, "  bwd diff =", mb)
    assert_true(mf < TOL, "TokenMean fwd CPU/GPU parity")
    assert_true(mb < TOL, "TokenMean bwd CPU/GPU parity")
    print("  ok")


def test_bias_add_parity(ctx: DeviceContext) raises:
    print("test_bias_add_parity ...")
    comptime BATCH = 4
    comptime DIM = 6
    comptime N = BATCH * DIM

    var cpu = BiasAdd[DIM].make[target="cpu", INIT=Zero]()
    var gpu = BiasAdd[DIM].make[target="gpu", INIT=Zero](ctx)
    # Identical bias on both.
    var bh = ctx.enqueue_create_host_buffer[DT](DIM)
    ctx.synchronize()
    var bc = cpu.bias.value_unsafe_ptr_cpu()
    for i in range(DIM):
        var v = Scalar[DT](0.1 * Float64(i) - 0.25)
        bc[i] = v
        bh.unsafe_ptr()[i] = v
    ctx.enqueue_copy(gpu.bias.value_dev.value(), bh)
    ctx.synchronize()

    var xh = ctx.enqueue_create_host_buffer[DT](N)
    var yh = ctx.enqueue_create_host_buffer[DT](N)
    var goh = ctx.enqueue_create_host_buffer[DT](N)
    var gih = ctx.enqueue_create_host_buffer[DT](N)
    var gbh = ctx.enqueue_create_host_buffer[DT](DIM)
    ctx.synchronize()
    for i in range(N):
        xh.unsafe_ptr()[i] = _spread(i, 3.0)
        goh.unsafe_ptr()[i] = _spread(i, 7.0)

    var ycpu = _alloc(N)
    var gicpu = _alloc(N)
    cpu.zero_grad["cpu"]()
    var xc = TileTensor(xh.unsafe_ptr(), row_major[BATCH, DIM]())
    var yc = TileTensor(ycpu, row_major[BATCH, DIM]())
    cpu.forward["cpu", BATCH](xc, output=yc)
    var goc = TileTensor(goh.unsafe_ptr(), row_major[BATCH, DIM]())
    var gic = TileTensor(gicpu, row_major[BATCH, DIM]())
    cpu.vjp["cpu", BATCH](goc, gic)

    var xd = ctx.enqueue_create_buffer[DT](N)
    var yd = ctx.enqueue_create_buffer[DT](N)
    var god = ctx.enqueue_create_buffer[DT](N)
    var gid = ctx.enqueue_create_buffer[DT](N)
    ctx.enqueue_copy(xd, xh)
    ctx.enqueue_copy(god, goh)
    ctx.synchronize()
    gpu.zero_grad["gpu"]()
    var xt = TileTensor(_mao(xd), row_major[BATCH, DIM]())
    var yt = TileTensor(_mao(yd), row_major[BATCH, DIM]())
    gpu.forward["gpu", BATCH](xt, output=yt)
    var got = TileTensor(_mao(god), row_major[BATCH, DIM]())
    var git = TileTensor(_mao(gid), row_major[BATCH, DIM]())
    gpu.vjp["gpu", BATCH](got, git)
    ctx.enqueue_copy(yh, yd)
    ctx.enqueue_copy(gih, gid)
    ctx.enqueue_copy(gbh, gpu.bias.grad_dev.value())
    ctx.synchronize()

    var mf = _maxdiff(yh.unsafe_ptr(), ycpu, N)
    var mb = _maxdiff(gih.unsafe_ptr(), gicpu, N)
    var mgb = _maxdiff(
        gbh.unsafe_ptr(), cpu.bias.grad_unsafe_ptr_cpu(), DIM
    )
    print("   fwd diff =", mf, "  bwd diff =", mb, "  dbias diff =", mgb)
    assert_true(mf < TOL, "BiasAdd fwd parity")
    assert_true(mb < TOL, "BiasAdd grad_input parity")
    assert_true(mgb < TOL, "BiasAdd grad_bias parity")
    print("  ok")


def test_embedding_parity(ctx: DeviceContext) raises:
    print("test_embedding_parity ...")
    comptime BATCH = 3
    comptime VOCAB = 5
    comptime EMBED = 4
    comptime IN_N = BATCH * VOCAB
    comptime OUT_N = BATCH * EMBED
    comptime W_N = VOCAB * EMBED

    var cpu = Embedding[VOCAB, EMBED].make[target="cpu", INIT=Zero]()
    var gpu = Embedding[VOCAB, EMBED].make[target="gpu", INIT=Zero](ctx)
    var wh = ctx.enqueue_create_host_buffer[DT](W_N)
    ctx.synchronize()
    var wc = cpu.weight.value_unsafe_ptr_cpu()
    for i in range(W_N):
        var v = Scalar[DT](0.2 * Float64(i) - 0.5)
        wc[i] = v
        wh.unsafe_ptr()[i] = v
    ctx.enqueue_copy(gpu.weight.value_dev.value(), wh)
    ctx.synchronize()

    var xh = ctx.enqueue_create_host_buffer[DT](IN_N)
    var yh = ctx.enqueue_create_host_buffer[DT](OUT_N)
    var goh = ctx.enqueue_create_host_buffer[DT](OUT_N)
    var gih = ctx.enqueue_create_host_buffer[DT](IN_N)
    var gwh = ctx.enqueue_create_host_buffer[DT](W_N)
    ctx.synchronize()
    for i in range(IN_N):
        xh.unsafe_ptr()[i] = Scalar[DT](0.0)
    for b in range(BATCH):
        xh.unsafe_ptr()[b * VOCAB + (b % VOCAB)] = Scalar[DT](1.0)
    for i in range(OUT_N):
        goh.unsafe_ptr()[i] = _spread(i, 4.0)

    var ycpu = _alloc(OUT_N)
    var gicpu = _alloc(IN_N)
    cpu.zero_grad["cpu"]()
    var xc = TileTensor(xh.unsafe_ptr(), row_major[BATCH, VOCAB]())
    var yc = TileTensor(ycpu, row_major[BATCH, EMBED]())
    cpu.forward["cpu", BATCH](xc, output=yc)
    var goc = TileTensor(goh.unsafe_ptr(), row_major[BATCH, EMBED]())
    var gic = TileTensor(gicpu, row_major[BATCH, VOCAB]())
    cpu.vjp["cpu", BATCH](goc, gic)

    var xd = ctx.enqueue_create_buffer[DT](IN_N)
    var yd = ctx.enqueue_create_buffer[DT](OUT_N)
    var god = ctx.enqueue_create_buffer[DT](OUT_N)
    var gid = ctx.enqueue_create_buffer[DT](IN_N)
    ctx.enqueue_copy(xd, xh)
    ctx.enqueue_copy(god, goh)
    ctx.synchronize()
    gpu.zero_grad["gpu"]()
    var xt = TileTensor(_mao(xd), row_major[BATCH, VOCAB]())
    var yt = TileTensor(_mao(yd), row_major[BATCH, EMBED]())
    gpu.forward["gpu", BATCH](xt, output=yt)
    var got = TileTensor(_mao(god), row_major[BATCH, EMBED]())
    var git = TileTensor(_mao(gid), row_major[BATCH, VOCAB]())
    gpu.vjp["gpu", BATCH](got, git)
    ctx.enqueue_copy(yh, yd)
    ctx.enqueue_copy(gih, gid)
    ctx.enqueue_copy(gwh, gpu.weight.grad_dev.value())
    ctx.synchronize()

    var mf = _maxdiff(yh.unsafe_ptr(), ycpu, OUT_N)
    var mb = _maxdiff(gih.unsafe_ptr(), gicpu, IN_N)
    var mgw = _maxdiff(
        gwh.unsafe_ptr(), cpu.weight.grad_unsafe_ptr_cpu(), W_N
    )
    print("   fwd diff =", mf, "  bwd diff =", mb, "  dW diff =", mgw)
    assert_true(mf < TOL, "Embedding fwd parity")
    assert_true(mb < TOL, "Embedding grad_input parity")
    assert_true(mgw < TOL, "Embedding grad_weight parity")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Wave A transformer primitives — CPU↔GPU parity")
    print("=" * 70)
    var ctx = DeviceContext()
    test_transpose2d_parity(ctx)
    test_token_mean_parity(ctx)
    test_bias_add_parity(ctx)
    test_embedding_parity(ctx)
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
