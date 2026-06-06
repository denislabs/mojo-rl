"""ModalitySpaceAttention — wrapper installs the right mask + delegates.

    pixi run            mojo run -I . tests/nn2/test_modality_space_attention.mojo  # CPU
    pixi run -e apple   mojo run -I . tests/nn2/test_modality_space_attention.mojo  # +GPU

Decisive check: the comptime-configured wrapper must be BIT-IDENTICAL to a
bare MaskedAttention with the same modality mask installed by hand (forward
+ vjp), on CPU and GPU, for encoder and decoder modes. This proves the
mask-build-in-make + forward/vjp delegation are correct, so the wrapper is a
valid drop-in Sequential leaf.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import alloc
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.masked_attention import (
    MaskedAttention,
    build_modality_mask,
)
from mojo_rl.nn2.primitives.modality_space_attention import ModalitySpaceAttention


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


def _ids[S: Int, L: Int]() -> List[Int]:
    var ids = List[Int]()
    for _ in range(L):
        ids.append(-1)
    for _ in range(S - L):
        ids.append(0)
    return ids^


def test_cpu_parity[
    D: Int, NH: Int, S: Int, L: Int, MODE: StaticString
](name: String) raises:
    print(name, "(cpu) ...")
    comptime BATCH = 2
    comptime IN_N = BATCH * S * D * 3
    comptime OUT_N = BATCH * S * D

    var wrap = ModalitySpaceAttention[D, NH, S, L, MODE].make[
        target="cpu", INIT=Zero
    ]()
    var bare = MaskedAttention[D, NH, S].make[target="cpu", INIT=Zero]()
    bare.set_mask(build_modality_mask[MODE](_ids[S, L](), L))

    var x = _alloc(IN_N)
    var go = _alloc(OUT_N)
    for i in range(IN_N):
        x[i] = _spread(i, 1.7)
    for i in range(OUT_N):
        go[i] = _spread(i, 0.9)
    var xt = TileTensor(x, row_major[BATCH, S * D * 3]())
    var got = TileTensor(go, row_major[BATCH, S * D]())

    var yw = _alloc(OUT_N)
    var yb = _alloc(OUT_N)
    var giw = _alloc(IN_N)
    var gib = _alloc(IN_N)
    var ywt = TileTensor(yw, row_major[BATCH, S * D]())
    var ybt = TileTensor(yb, row_major[BATCH, S * D]())
    var giwt = TileTensor(giw, row_major[BATCH, S * D * 3]())
    var gibt = TileTensor(gib, row_major[BATCH, S * D * 3]())
    wrap.forward["cpu", BATCH](xt, output=ywt)
    bare.forward["cpu", BATCH](xt, output=ybt)
    wrap.vjp["cpu", BATCH](got, giwt)
    bare.vjp["cpu", BATCH](got, gibt)

    var mf = _maxdiff(yw, yb, OUT_N)
    var mb = _maxdiff(giw, gib, IN_N)
    print("   fwd diff =", mf, "  bwd diff =", mb)
    assert_true(mf == 0.0 and mb == 0.0, name + ": cpu parity (must be exact)")
    print("  ok")


def test_gpu_parity[
    D: Int, NH: Int, S: Int, L: Int, MODE: StaticString
](ctx: DeviceContext, name: String) raises:
    print(name, "(gpu) ...")
    comptime BATCH = 2
    comptime IN_N = BATCH * S * D * 3
    comptime OUT_N = BATCH * S * D

    var wrap = ModalitySpaceAttention[D, NH, S, L, MODE].make[
        target="gpu", INIT=Zero
    ](ctx)
    var bare = MaskedAttention[D, NH, S].make[target="gpu", INIT=Zero](ctx)
    bare.set_mask(build_modality_mask[MODE](_ids[S, L](), L))

    var xh = ctx.enqueue_create_host_buffer[DT](IN_N)
    var goh = ctx.enqueue_create_host_buffer[DT](OUT_N)
    var ywh = ctx.enqueue_create_host_buffer[DT](OUT_N)
    var ybh = ctx.enqueue_create_host_buffer[DT](OUT_N)
    var giwh = ctx.enqueue_create_host_buffer[DT](IN_N)
    var gibh = ctx.enqueue_create_host_buffer[DT](IN_N)
    ctx.synchronize()
    for i in range(IN_N):
        xh.unsafe_ptr()[i] = _spread(i, 1.7)
    for i in range(OUT_N):
        goh.unsafe_ptr()[i] = _spread(i, 0.9)

    var xd = ctx.enqueue_create_buffer[DT](IN_N)
    var god = ctx.enqueue_create_buffer[DT](OUT_N)
    var ywd = ctx.enqueue_create_buffer[DT](OUT_N)
    var ybd = ctx.enqueue_create_buffer[DT](OUT_N)
    var giwd = ctx.enqueue_create_buffer[DT](IN_N)
    var gibd = ctx.enqueue_create_buffer[DT](IN_N)
    ctx.enqueue_copy(xd, xh)
    ctx.enqueue_copy(god, goh)
    ctx.synchronize()
    var xt = TileTensor(_mao(xd), row_major[BATCH, S * D * 3]())
    var got = TileTensor(_mao(god), row_major[BATCH, S * D]())
    var ywt = TileTensor(_mao(ywd), row_major[BATCH, S * D]())
    var ybt = TileTensor(_mao(ybd), row_major[BATCH, S * D]())
    var giwt = TileTensor(_mao(giwd), row_major[BATCH, S * D * 3]())
    var gibt = TileTensor(_mao(gibd), row_major[BATCH, S * D * 3]())
    wrap.forward["gpu", BATCH](xt, output=ywt)
    bare.forward["gpu", BATCH](xt, output=ybt)
    wrap.vjp["gpu", BATCH](got, giwt)
    bare.vjp["gpu", BATCH](got, gibt)
    ctx.enqueue_copy(ywh, ywd)
    ctx.enqueue_copy(ybh, ybd)
    ctx.enqueue_copy(giwh, giwd)
    ctx.enqueue_copy(gibh, gibd)
    ctx.synchronize()

    var mf = _maxdiff(ywh.unsafe_ptr(), ybh.unsafe_ptr(), OUT_N)
    var mb = _maxdiff(giwh.unsafe_ptr(), gibh.unsafe_ptr(), IN_N)
    print("   fwd diff =", mf, "  bwd diff =", mb)
    assert_true(mf == 0.0 and mb == 0.0, name + ": gpu parity (must be exact)")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("ModalitySpaceAttention — wrapper parity (Phase 1)")
    print("=" * 70)
    test_cpu_parity[4, 2, 6, 2, "encoder"]("encoder")
    test_cpu_parity[4, 2, 6, 2, "decoder"]("decoder")
    var ctx = DeviceContext()
    test_gpu_parity[4, 2, 6, 2, "encoder"](ctx, "encoder")
    test_gpu_parity[4, 2, 6, 2, "decoder"](ctx, "decoder")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
