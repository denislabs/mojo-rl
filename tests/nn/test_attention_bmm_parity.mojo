"""ScaledDotProductAttention — BMM fast path vs custom path parity (Wave C 6d).

The optimized GPU path (USE_MAX_KERNELS=True → batched_matmul for QKᵀ and
attn·V) must produce the same forward + grad_input as the serial per-(b,h)
custom path (USE_MAX_KERNELS=False), so flipping the flag changes speed, not
results. Runs both on GPU over identical inputs and compares, non-causal +
causal, multi-head.

    pixi run -e apple  mojo run -I . tests/nn/test_attention_bmm_parity.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_attention_bmm_parity.mojo

Tolerance 2e-3: the bmm path reorders the matmul accumulation vs the custom
path's scalar dot products. On Apple Metal both are fp32. On NVIDIA the
batched GEMM may use TF32 → expect a larger (but still small) gap; bump TOL
if so. Docs: docs/NN_TRANSFORMER_PORT.md.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import alloc
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.primitives.attention import ScaledDotProductAttention


comptime TOL: Float64 = 2e-3


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


def _run[
    DIM: Int, N_HEADS: Int, SEQ: Int, CAUSAL: Bool
](ctx: DeviceContext, name: String) raises:
    print(name, "...")
    comptime BATCH = 2
    comptime IN_N = BATCH * SEQ * DIM * 3
    comptime OUT_N = BATCH * SEQ * DIM

    var custom = ScaledDotProductAttention[
        DIM, N_HEADS, SEQ, CAUSAL, False
    ].make[target="gpu", INIT=Zero](ctx)
    var bmm = ScaledDotProductAttention[
        DIM, N_HEADS, SEQ, CAUSAL, True
    ].make[target="gpu", INIT=Zero](ctx)

    var xh = ctx.enqueue_create_host_buffer[DT](IN_N)
    var goh = ctx.enqueue_create_host_buffer[DT](OUT_N)
    var y_cu = ctx.enqueue_create_host_buffer[DT](OUT_N)
    var y_bm = ctx.enqueue_create_host_buffer[DT](OUT_N)
    var gi_cu = ctx.enqueue_create_host_buffer[DT](IN_N)
    var gi_bm = ctx.enqueue_create_host_buffer[DT](IN_N)
    ctx.synchronize()
    for i in range(IN_N):
        xh.unsafe_ptr()[i] = _spread(i, 1.3)
    for i in range(OUT_N):
        goh.unsafe_ptr()[i] = _spread(i, 4.1)

    var xd = ctx.enqueue_create_buffer[DT](IN_N)
    var god = ctx.enqueue_create_buffer[DT](OUT_N)
    var y_cu_d = ctx.enqueue_create_buffer[DT](OUT_N)
    var y_bm_d = ctx.enqueue_create_buffer[DT](OUT_N)
    var gi_cu_d = ctx.enqueue_create_buffer[DT](IN_N)
    var gi_bm_d = ctx.enqueue_create_buffer[DT](IN_N)
    ctx.enqueue_copy(xd, xh)
    ctx.enqueue_copy(god, goh)
    ctx.synchronize()

    var x_tt = TileTensor(_mao(xd), row_major[BATCH, SEQ * DIM * 3]())
    var go_tt = TileTensor(_mao(god), row_major[BATCH, SEQ * DIM]())

    # Custom path.
    var ycu_tt = TileTensor(_mao(y_cu_d), row_major[BATCH, SEQ * DIM]())
    var gicu_tt = TileTensor(_mao(gi_cu_d), row_major[BATCH, SEQ * DIM * 3]())
    custom.forward["gpu", BATCH](x_tt, output=ycu_tt)
    custom.vjp["gpu", BATCH](go_tt, gicu_tt)

    # BMM path.
    var ybm_tt = TileTensor(_mao(y_bm_d), row_major[BATCH, SEQ * DIM]())
    var gibm_tt = TileTensor(_mao(gi_bm_d), row_major[BATCH, SEQ * DIM * 3]())
    bmm.forward["gpu", BATCH](x_tt, output=ybm_tt)
    bmm.vjp["gpu", BATCH](go_tt, gibm_tt)

    ctx.enqueue_copy(y_cu, y_cu_d)
    ctx.enqueue_copy(y_bm, y_bm_d)
    ctx.enqueue_copy(gi_cu, gi_cu_d)
    ctx.enqueue_copy(gi_bm, gi_bm_d)
    ctx.synchronize()

    var mf = _maxdiff(y_cu.unsafe_ptr(), y_bm.unsafe_ptr(), OUT_N)
    var mb = _maxdiff(gi_cu.unsafe_ptr(), gi_bm.unsafe_ptr(), IN_N)
    print("   fwd diff =", mf, "  grad_input diff =", mb)
    assert_true(mf < TOL, name + ": forward bmm vs custom")
    assert_true(mb < TOL, name + ": grad_input bmm vs custom")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Attention BMM fast path vs custom path parity (Wave C 6d)")
    print("=" * 70)
    var ctx = DeviceContext()
    _run[8, 2, 4, False](ctx, "noncausal_mh")
    _run[8, 2, 4, True](ctx, "causal_mh")
    _run[16, 4, 8, False](ctx, "noncausal_mh_larger")
    _run[16, 4, 8, True](ctx, "causal_mh_larger")
    _run[6, 1, 5, True](ctx, "singlehead_causal")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
