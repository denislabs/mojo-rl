"""LearnedTokens — CPU↔GPU parity (forward + grad_input + grad_param).

    pixi run -e apple  mojo run -I . tests/nn2/test_learned_tokens_gpu.mojo

Params are forced identical on both sides (overwritten with a deterministic
pattern) so forward, grad_input, and the batch-reduced grad_param must match.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import alloc
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.primitives.learned_tokens import LearnedTokens


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


def _run[
    N_IN: Int, N_NEW: Int, D: Int, PREPEND: Bool
](ctx: DeviceContext, name: String) raises:
    print(name, "...")
    comptime BATCH = 6
    comptime IN_N = BATCH * N_IN * D
    comptime OUT_N = BATCH * (N_IN + N_NEW) * D
    comptime PN = N_NEW * D

    var cpu = LearnedTokens[N_IN, N_NEW, D, PREPEND].make[
        target="cpu", INIT=Xavier
    ]()
    var gpu = LearnedTokens[N_IN, N_NEW, D, PREPEND].make[
        target="gpu", INIT=Xavier
    ](ctx)

    # force identical params on both sides.
    var phost = ctx.enqueue_create_host_buffer[DT](PN)
    ctx.synchronize()
    var cpu_p = cpu.tokens.value_unsafe_ptr_cpu()
    for k in range(PN):
        var v = _spread(k, 0.33)
        cpu_p[k] = v
        phost.unsafe_ptr()[k] = v
    ctx.enqueue_copy(gpu.tokens.val.dev.value(), phost)
    ctx.synchronize()

    var xh = ctx.enqueue_create_host_buffer[DT](IN_N)
    var goh = ctx.enqueue_create_host_buffer[DT](OUT_N)
    ctx.synchronize()
    for i in range(IN_N):
        xh.unsafe_ptr()[i] = _spread(i, 1.7)
    for i in range(OUT_N):
        goh.unsafe_ptr()[i] = _spread(i, 0.9)

    # CPU forward + vjp.
    var ycpu = _alloc(OUT_N)
    var gicpu = _alloc(IN_N)
    var xc = TileTensor(xh.unsafe_ptr(), row_major[BATCH, N_IN * D]())
    var yct = TileTensor(ycpu, row_major[BATCH, (N_IN + N_NEW) * D]())
    cpu.forward["cpu", BATCH](xc, output=yct)
    cpu.zero_grad["cpu"]()
    var goc = TileTensor(goh.unsafe_ptr(), row_major[BATCH, (N_IN + N_NEW) * D]())
    var gict = TileTensor(gicpu, row_major[BATCH, N_IN * D]())
    cpu.vjp["cpu", BATCH](goc, gict)

    # GPU forward + vjp.
    var xd = ctx.enqueue_create_buffer[DT](IN_N)
    var yd = ctx.enqueue_create_buffer[DT](OUT_N)
    var god = ctx.enqueue_create_buffer[DT](OUT_N)
    var gid = ctx.enqueue_create_buffer[DT](IN_N)
    ctx.enqueue_copy(xd, xh)
    ctx.enqueue_copy(god, goh)
    ctx.synchronize()
    var xt = TileTensor(_mao(xd), row_major[BATCH, N_IN * D]())
    var yt = TileTensor(_mao(yd), row_major[BATCH, (N_IN + N_NEW) * D]())
    gpu.forward["gpu", BATCH](xt, output=yt)
    gpu.zero_grad["gpu"]()
    var got = TileTensor(_mao(god), row_major[BATCH, (N_IN + N_NEW) * D]())
    var git = TileTensor(_mao(gid), row_major[BATCH, N_IN * D]())
    gpu.vjp["gpu", BATCH](got, git)

    var yh = ctx.enqueue_create_host_buffer[DT](OUT_N)
    var gih = ctx.enqueue_create_host_buffer[DT](IN_N)
    var gph = ctx.enqueue_create_host_buffer[DT](PN)
    ctx.enqueue_copy(yh, yd)
    ctx.enqueue_copy(gih, gid)
    ctx.enqueue_copy(gph, gpu.tokens.grd.dev.value())
    ctx.synchronize()

    var mf = _maxdiff(yh.unsafe_ptr(), ycpu, OUT_N)
    var mi = _maxdiff(gih.unsafe_ptr(), gicpu, IN_N)
    var mp = _maxdiff(gph.unsafe_ptr(), cpu.tokens.grd.cpu.unsafe_ptr(), PN)
    print("   fwd =", mf, " grad_in =", mi, " grad_param =", mp)
    assert_true(mf < TOL and mi < TOL and mp < TOL, name + ": parity")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("LearnedTokens CPU↔GPU parity")
    print("=" * 70)
    var ctx = DeviceContext()
    _run[3, 2, 4, True](ctx, "prepend (encoder)")
    _run[3, 2, 4, False](ctx, "append (decoder)")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
