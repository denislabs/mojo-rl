"""ScaledDotProductAttention — CPU↔GPU parity (Wave C 6c).

    pixi run -e apple  mojo run -I . tests/nn/test_attention_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_attention_gpu.mojo

Compares the custom per-(b,h) GPU kernels against the CPU path for
forward + grad_input, non-causal and causal, multi-head. Attention has no
params. Tolerance 1e-4 (fp32; GPU accumulates in Float32 vs CPU Float64,
so a touch looser than the linear-op tests). Docs: NN2_TRANSFORMER_PORT.md.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import alloc
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.primitives.attention import ScaledDotProductAttention


comptime TOL: Float64 = 1e-4


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](n)
    )


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


def _run_parity[
    DIM: Int, N_HEADS: Int, SEQ: Int, CAUSAL: Bool
](ctx: DeviceContext, name: String) raises:
    print(name, "...")
    comptime BATCH = 2
    comptime IN_N = BATCH * SEQ * DIM * 3
    comptime OUT_N = BATCH * SEQ * DIM

    var cpu = ScaledDotProductAttention[
        DIM, N_HEADS, SEQ, CAUSAL
    ].make[target="cpu", INIT=Zero]()
    var gpu = ScaledDotProductAttention[
        DIM, N_HEADS, SEQ, CAUSAL
    ].make[target="gpu", INIT=Zero](ctx)

    var xh = ctx.enqueue_create_host_buffer[DT](IN_N)
    var yh = ctx.enqueue_create_host_buffer[DT](OUT_N)
    var goh = ctx.enqueue_create_host_buffer[DT](OUT_N)
    var gih = ctx.enqueue_create_host_buffer[DT](IN_N)
    ctx.synchronize()
    for i in range(IN_N):
        xh.unsafe_ptr()[i] = _spread(i, 1.3)
    for i in range(OUT_N):
        goh.unsafe_ptr()[i] = _spread(i, 4.1)

    # CPU.
    var ycpu = _alloc(OUT_N)
    var gicpu = _alloc(IN_N)
    var xc = TileTensor(xh.unsafe_ptr(), row_major[BATCH, SEQ * DIM * 3]())
    var yc = TileTensor(ycpu, row_major[BATCH, SEQ * DIM]())
    cpu.forward["cpu", BATCH](xc, output=yc)
    var goc = TileTensor(goh.unsafe_ptr(), row_major[BATCH, SEQ * DIM]())
    var gic = TileTensor(gicpu, row_major[BATCH, SEQ * DIM * 3]())
    cpu.vjp["cpu", BATCH](goc, gic)

    # GPU.
    var xd = ctx.enqueue_create_buffer[DT](IN_N)
    var yd = ctx.enqueue_create_buffer[DT](OUT_N)
    var god = ctx.enqueue_create_buffer[DT](OUT_N)
    var gid = ctx.enqueue_create_buffer[DT](IN_N)
    ctx.enqueue_copy(xd, xh)
    ctx.enqueue_copy(god, goh)
    ctx.synchronize()
    var xt = TileTensor(_mao(xd), row_major[BATCH, SEQ * DIM * 3]())
    var yt = TileTensor(_mao(yd), row_major[BATCH, SEQ * DIM]())
    gpu.forward["gpu", BATCH](xt, output=yt)
    var got = TileTensor(_mao(god), row_major[BATCH, SEQ * DIM]())
    var git = TileTensor(_mao(gid), row_major[BATCH, SEQ * DIM * 3]())
    gpu.vjp["gpu", BATCH](got, git)
    ctx.enqueue_copy(yh, yd)
    ctx.enqueue_copy(gih, gid)
    ctx.synchronize()

    var mf = _maxdiff(yh.unsafe_ptr(), ycpu, OUT_N)
    var mb = _maxdiff(gih.unsafe_ptr(), gicpu, IN_N)
    print("   fwd diff =", mf, "  grad_input diff =", mb)
    assert_true(mf < TOL, name + ": forward parity")
    assert_true(mb < TOL, name + ": grad_input parity")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("ScaledDotProductAttention CPU↔GPU parity (Wave C 6c)")
    print("=" * 70)
    var ctx = DeviceContext()
    _run_parity[4, 2, 3, False](ctx, "noncausal_mh")
    _run_parity[4, 2, 3, True](ctx, "causal_mh")
    _run_parity[6, 1, 4, False](ctx, "singlehead")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
