"""MAEReplacer — CPU↔GPU parity (forward + grad_input + grad_mask_token).

    pixi run -e apple  mojo run -I . tests/nn/test_mae_replacer_gpu.mojo

Forward parity is the key check: it only matches if the Float32 PhiloxRandom
keep decision is bit-identical on CPU and GPU (a mismatched patch would flip
kept↔dropped and blow up the diff). Also checks grad_input and the
batch-reduced grad_mask_token. mask_token is forced identical on both sides.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import alloc
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.primitives.mae_replacer import MAEReplacer


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


def main() raises:
    print("=" * 70)
    print("MAEReplacer CPU↔GPU parity")
    print("=" * 70)
    comptime NP = 6
    comptime D = 4
    comptime BATCH = 8
    comptime N = BATCH * NP * D
    var ctx = DeviceContext()

    var cpu = MAEReplacer[NP, D, 0.5, 0.5, 12345].make[target="cpu", INIT=Xavier]()
    var gpu = MAEReplacer[NP, D, 0.5, 0.5, 12345].make[target="gpu", INIT=Xavier](ctx)

    # force identical mask_token.
    var mth = ctx.enqueue_create_host_buffer[DT](D)
    ctx.synchronize()
    var cpu_mt = cpu.mask_token.value_unsafe_ptr_cpu()
    for k in range(D):
        var v = _spread(k, 0.21)
        cpu_mt[k] = v
        mth.unsafe_ptr()[k] = v
    ctx.enqueue_copy(gpu.mask_token.val.dev.value(), mth)
    ctx.synchronize()

    var xh = ctx.enqueue_create_host_buffer[DT](N)
    var goh = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()
    for i in range(N):
        xh.unsafe_ptr()[i] = _spread(i, 1.3)
        goh.unsafe_ptr()[i] = _spread(i, 4.1)

    # CPU.
    var ycpu = _alloc(N)
    var gicpu = _alloc(N)
    var xc = TileTensor(xh.unsafe_ptr(), row_major[BATCH, NP * D]())
    var yct = TileTensor(ycpu, row_major[BATCH, NP * D]())
    cpu.forward["cpu", BATCH](xc, output=yct)
    cpu.zero_grad["cpu"]()
    var goc = TileTensor(goh.unsafe_ptr(), row_major[BATCH, NP * D]())
    var gict = TileTensor(gicpu, row_major[BATCH, NP * D]())
    cpu.vjp["cpu", BATCH](goc, gict)

    # GPU.
    var xd = ctx.enqueue_create_buffer[DT](N)
    var yd = ctx.enqueue_create_buffer[DT](N)
    var god = ctx.enqueue_create_buffer[DT](N)
    var gid = ctx.enqueue_create_buffer[DT](N)
    ctx.enqueue_copy(xd, xh)
    ctx.enqueue_copy(god, goh)
    ctx.synchronize()
    var xt = TileTensor(_mao(xd), row_major[BATCH, NP * D]())
    var yt = TileTensor(_mao(yd), row_major[BATCH, NP * D]())
    gpu.forward["gpu", BATCH](xt, output=yt)
    gpu.zero_grad["gpu"]()
    var got = TileTensor(_mao(god), row_major[BATCH, NP * D]())
    var git = TileTensor(_mao(gid), row_major[BATCH, NP * D]())
    gpu.vjp["gpu", BATCH](got, git)

    var yh = ctx.enqueue_create_host_buffer[DT](N)
    var gih = ctx.enqueue_create_host_buffer[DT](N)
    var gth = ctx.enqueue_create_host_buffer[DT](D)
    ctx.enqueue_copy(yh, yd)
    ctx.enqueue_copy(gih, gid)
    ctx.enqueue_copy(gth, gpu.mask_token.grd.dev.value())
    ctx.synchronize()

    var mf = _maxdiff(yh.unsafe_ptr(), ycpu, N)
    var mi = _maxdiff(gih.unsafe_ptr(), gicpu, N)
    var mt = _maxdiff(gth.unsafe_ptr(), cpu.mask_token.grd.cpu.unsafe_ptr(), D)
    print("   fwd =", mf, " grad_in =", mi, " grad_mask_token =", mt)
    assert_true(mf < TOL, "forward parity (keep mask must bit-match)")
    assert_true(mi < TOL and mt < TOL, "grad parity")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
