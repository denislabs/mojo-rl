"""Tokenwise[SEQ_LEN, Linear] — CPU↔GPU parity (Wave B).

    pixi run -e apple  mojo run -I . tests/nn2/test_tokenwise_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/nn2/test_tokenwise_gpu.mojo

Builds Tokenwise[SEQ, Linear[IN,OUT]] on CPU and GPU with identical inner
params, then compares forward + grad_input + grad_weight + grad_bias.
Tolerance 1e-5 (fp32). Docs: docs/NN2_TRANSFORMER_PORT.md Phase 1 Wave B.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import alloc
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.combinators.tokenwise import Tokenwise


comptime TOL: Float64 = 1e-5


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](n)
    )


def _mao(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


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


def test_tokenwise_linear_parity(ctx: DeviceContext) raises:
    print("test_tokenwise_linear_parity ...")
    comptime BATCH = 2
    comptime SEQ = 3
    comptime IN = 4
    comptime OUT = 5
    comptime IN_N = BATCH * SEQ * IN
    comptime OUT_N = BATCH * SEQ * OUT
    comptime W_N = IN * OUT

    var cpu = Tokenwise[SEQ, Linear[IN, OUT]].make[target="cpu", INIT=Zero]()
    var gpu = Tokenwise[SEQ, Linear[IN, OUT]].make[
        target="gpu", INIT=Zero
    ](ctx)

    # Identical inner Linear params on both.
    var wh = ctx.enqueue_create_host_buffer[DT](W_N)
    var bh = ctx.enqueue_create_host_buffer[DT](OUT)
    ctx.synchronize()
    var wc = cpu.inner.weight.value_unsafe_ptr_cpu()
    var bc = cpu.inner.bias.value_unsafe_ptr_cpu()
    for i in range(W_N):
        var v = Scalar[DT](0.05 * Float64(i) - 0.3)
        wc[i] = v
        wh.unsafe_ptr()[i] = v
    for o in range(OUT):
        var v = Scalar[DT](0.1 * Float64(o) - 0.2)
        bc[o] = v
        bh.unsafe_ptr()[o] = v
    ctx.enqueue_copy(gpu.inner.weight.value_dev.value(), wh)
    ctx.enqueue_copy(gpu.inner.bias.value_dev.value(), bh)
    ctx.synchronize()

    var xh = ctx.enqueue_create_host_buffer[DT](IN_N)
    var yh = ctx.enqueue_create_host_buffer[DT](OUT_N)
    var goh = ctx.enqueue_create_host_buffer[DT](OUT_N)
    var gih = ctx.enqueue_create_host_buffer[DT](IN_N)
    var gwh = ctx.enqueue_create_host_buffer[DT](W_N)
    var gbh = ctx.enqueue_create_host_buffer[DT](OUT)
    ctx.synchronize()
    for i in range(IN_N):
        xh.unsafe_ptr()[i] = Scalar[DT](0.13 * Float64(i) - 0.5)
    for i in range(OUT_N):
        goh.unsafe_ptr()[i] = Scalar[DT](0.07 * Float64(i) - 0.25)

    # CPU.
    var ycpu = _alloc(OUT_N)
    var gicpu = _alloc(IN_N)
    cpu.zero_grad["cpu"]()
    var xc = TileTensor(xh.unsafe_ptr(), row_major[BATCH, SEQ * IN]())
    var yc = TileTensor(ycpu, row_major[BATCH, SEQ * OUT]())
    cpu.forward["cpu", BATCH](xc, output=yc)
    var goc = TileTensor(goh.unsafe_ptr(), row_major[BATCH, SEQ * OUT]())
    var gic = TileTensor(gicpu, row_major[BATCH, SEQ * IN]())
    cpu.vjp["cpu", BATCH](goc, gic)

    # GPU.
    var xd = ctx.enqueue_create_buffer[DT](IN_N)
    var yd = ctx.enqueue_create_buffer[DT](OUT_N)
    var god = ctx.enqueue_create_buffer[DT](OUT_N)
    var gid = ctx.enqueue_create_buffer[DT](IN_N)
    ctx.enqueue_copy(xd, xh)
    ctx.enqueue_copy(god, goh)
    ctx.synchronize()
    gpu.zero_grad["gpu"]()
    var xt = TileTensor(_mao(xd), row_major[BATCH, SEQ * IN]())
    var yt = TileTensor(_mao(yd), row_major[BATCH, SEQ * OUT]())
    gpu.forward["gpu", BATCH](xt, output=yt)
    var got = TileTensor(_mao(god), row_major[BATCH, SEQ * OUT]())
    var git = TileTensor(_mao(gid), row_major[BATCH, SEQ * IN]())
    gpu.vjp["gpu", BATCH](got, git)
    ctx.enqueue_copy(yh, yd)
    ctx.enqueue_copy(gih, gid)
    ctx.enqueue_copy(gwh, gpu.inner.weight.grad_dev.value())
    ctx.enqueue_copy(gbh, gpu.inner.bias.grad_dev.value())
    ctx.synchronize()

    var mf = _maxdiff(yh.unsafe_ptr(), ycpu, OUT_N)
    var mb = _maxdiff(gih.unsafe_ptr(), gicpu, IN_N)
    var mw = _maxdiff(
        gwh.unsafe_ptr(), cpu.inner.weight.grad_unsafe_ptr_cpu(), W_N
    )
    var mbias = _maxdiff(
        gbh.unsafe_ptr(), cpu.inner.bias.grad_unsafe_ptr_cpu(), OUT
    )
    print(
        "   fwd =", mf, " grad_in =", mb, " dW =", mw, " dbias =", mbias
    )
    assert_true(mf < TOL, "Tokenwise fwd parity")
    assert_true(mb < TOL, "Tokenwise grad_input parity")
    assert_true(mw < TOL, "Tokenwise grad_weight parity")
    assert_true(mbias < TOL, "Tokenwise grad_bias parity")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Tokenwise[SEQ, Linear] CPU↔GPU parity (Wave B)")
    print("=" * 70)
    var ctx = DeviceContext()
    test_tokenwise_linear_parity(ctx)
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
