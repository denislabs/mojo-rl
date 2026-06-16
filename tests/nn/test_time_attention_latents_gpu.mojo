"""TimeAttentionLatents — CPU↔GPU parity (forward + grad_input).

    pixi run -e apple  mojo run -I . tests/nn/test_time_attention_latents_gpu.mojo

Validates the device gather → inner causal MHA over T → scatter path against
the CPU path. Params are made identical by reseeding the global RNG to the
same value before each `make` (Xavier draws the same sequence), so the only
difference is the CPU vs GPU execution.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import alloc
from std.math import abs
from std.random import seed
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.primitives.time_attention_latents import TimeAttentionLatents


comptime TOL: Float64 = 1e-4


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def _mao(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _spread(i: Int, s: Float64) -> Scalar[DT]:
    var x = s + 0.7 * Float64(i)
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


def main() raises:
    print("=" * 70)
    print("TimeAttentionLatents CPU↔GPU parity")
    print("=" * 70)
    comptime D = 4
    comptime NH = 2
    comptime T = 3
    comptime S = 5
    comptime L = 2
    comptime B = 2
    comptime BATCH = B * T
    comptime IN_N = BATCH * S * D
    comptime ctxseed = 4242

    var ctx = DeviceContext()
    seed(ctxseed)
    var cpu = TimeAttentionLatents[D, NH, T, S, L].make[target="cpu", INIT=Xavier]()
    seed(ctxseed)
    var gpu = TimeAttentionLatents[D, NH, T, S, L].make[target="gpu", INIT=Xavier](ctx)

    var xh = ctx.enqueue_create_host_buffer[DT](IN_N)
    var goh = ctx.enqueue_create_host_buffer[DT](IN_N)
    ctx.synchronize()
    for i in range(IN_N):
        xh.unsafe_ptr()[i] = _spread(i, 1.3)
        goh.unsafe_ptr()[i] = _spread(i, 4.1)

    # CPU.
    var ycpu = _alloc(IN_N)
    var gicpu = _alloc(IN_N)
    var xc = TileTensor(xh.unsafe_ptr(), row_major[BATCH, S * D]())
    var yct = TileTensor(ycpu, row_major[BATCH, S * D]())
    cpu.forward["cpu", BATCH](xc, output=yct)
    var goc = TileTensor(goh.unsafe_ptr(), row_major[BATCH, S * D]())
    var gict = TileTensor(gicpu, row_major[BATCH, S * D]())
    cpu.vjp["cpu", BATCH](goc, gict)

    # GPU.
    var xd = ctx.enqueue_create_buffer[DT](IN_N)
    var yd = ctx.enqueue_create_buffer[DT](IN_N)
    var god = ctx.enqueue_create_buffer[DT](IN_N)
    var gid = ctx.enqueue_create_buffer[DT](IN_N)
    ctx.enqueue_copy(xd, xh)
    ctx.enqueue_copy(god, goh)
    ctx.synchronize()
    var xt = TileTensor(_mao(xd), row_major[BATCH, S * D]())
    var yt = TileTensor(_mao(yd), row_major[BATCH, S * D]())
    gpu.forward["gpu", BATCH](xt, output=yt)
    var got = TileTensor(_mao(god), row_major[BATCH, S * D]())
    var git = TileTensor(_mao(gid), row_major[BATCH, S * D]())
    gpu.vjp["gpu", BATCH](got, git)

    var yh = ctx.enqueue_create_host_buffer[DT](IN_N)
    var gih = ctx.enqueue_create_host_buffer[DT](IN_N)
    ctx.enqueue_copy(yh, yd)
    ctx.enqueue_copy(gih, gid)
    ctx.synchronize()

    var mf = _maxdiff(yh.unsafe_ptr(), ycpu, IN_N)
    var mi = _maxdiff(gih.unsafe_ptr(), gicpu, IN_N)
    print("   fwd diff =", mf, "  grad_input diff =", mi)
    assert_true(mf < TOL and mi < TOL, "TimeAttentionLatents parity")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
