"""Our `RoPE` leaf vs MAX's shipped `nn.rope.rope_ragged`, on device.

The interleaving convention is the classic way a RoPE port loads fine and
computes gibberish, so it is not checked against our own restatement of it. It
is checked against a THIRD-PARTY implementation — the `rope_ragged` kernel that
ships in `nn.mojoc` — driven on the same inputs.

Two independent things are asserted:

  1. **Parity vs MAX.** Same x, same frequencies, same positions. Our leaf
     builds cos/sin from `THETA`; MAX takes a `freqs_cis` table, so the table is
     built from the SAME theta in the interleaved (re, im) layout its kernel
     expects. Compared elementwise, with the compared count printed — a silent
     zero-length comparison is the failure mode this whole file exists to avoid.

  2. **The adjoint identity**, `<RoPE(x), y> == <x, vjp(y)>`. A rotation is
     linear and orthogonal, so this holds EXACTLY (to float rounding) and is a
     complete check of the backward — no finite differences, no tolerance
     tuning. A backward that rotated the wrong way, or forgot a sign, fails it.

Run:
  pixi run -e apple mojo run -I . tests/nn/test_rope_vs_max_kernel.mojo
"""

from std.math import cos, sin, exp, log, abs
from std.testing import assert_true
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major, Coord
from nn.rope import rope_ragged
from std.utils import IndexList

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.rope import RoPE

comptime SEQ = 6
comptime HEADS = 3
comptime HD = 8
comptime THETA = 10000.0
comptime B = 1
comptime N = SEQ * HEADS * HD
comptime HALF = HD // 2

comptime x_layout = row_major[SEQ, HEADS, HD]()
comptime ro_layout = row_major[2]()
comptime sp_layout = row_major[1]()
comptime fc_layout = row_major[SEQ, HD]()


def main() raises:
    print("=" * 66)
    print("RoPE — ours vs MAX `rope_ragged`, and the adjoint identity")
    print("=" * 66)
    var ctx = DeviceContext()

    # ── shared input ─────────────────────────────────────────────────────
    var host = List[Scalar[DT]]()
    for i in range(N):
        host.append(Scalar[DT](((i * 37) % 19) - 9) * 0.1)

    # ── MAX's kernel ─────────────────────────────────────────────────────
    var xb = ctx.enqueue_create_buffer[DT](N)
    var xh = ctx.enqueue_create_host_buffer[DT](N)
    var ob = ctx.enqueue_create_buffer[DT](N)
    var oh = ctx.enqueue_create_host_buffer[DT](N)
    var rob = ctx.enqueue_create_buffer[DType.uint32](2)
    var roh = ctx.enqueue_create_host_buffer[DType.uint32](2)
    var spb = ctx.enqueue_create_buffer[DType.uint32](1)
    var sph = ctx.enqueue_create_host_buffer[DType.uint32](1)
    var fcb = ctx.enqueue_create_buffer[DT](SEQ * HD)
    var fch = ctx.enqueue_create_host_buffer[DT](SEQ * HD)
    ctx.synchronize()

    for i in range(N):
        xh[i] = host[i]
    roh[0] = UInt32(0)
    roh[1] = UInt32(SEQ)
    sph[0] = UInt32(0)
    # freqs_cis[p] = [re_0, im_0, re_1, im_1, ...] from the SAME theta.
    for p in range(SEQ):
        for j in range(HALF):
            var inv = exp(-(2.0 * Float64(j)) / Float64(HD) * log(THETA))
            var a = Float64(p) * inv
            fch[p * HD + 2 * j] = Scalar[DT](cos(a))
            fch[p * HD + 2 * j + 1] = Scalar[DT](sin(a))
    ctx.enqueue_copy(xb, xh)
    ctx.enqueue_copy(rob, roh)
    ctx.enqueue_copy(spb, sph)
    ctx.enqueue_copy(fcb, fch)
    ctx.synchronize()

    var x_t = TileTensor(xb, x_layout)
    var ro_t = TileTensor(rob, ro_layout)
    var sp_t = TileTensor(spb, sp_layout)
    var fc_t = TileTensor(fcb, fc_layout)
    var o_t = TileTensor(ob, x_layout)

    @always_inline
    def output_fn[
        width: SIMDLength, alignment: Int
    ](idx: IndexList[3], val: SIMD[DT, width]) {var o_t} -> None:
        o_t.store[width=width](Coord(idx), val)

    # interleaved=False == the split-halves (safetensors) convention, which is
    # what our leaf implements and what HF checkpoints store.
    rope_ragged[DT, DT, interleaved=False, target = StaticString("gpu")](
        x=x_t.as_unsafe_any_origin(),
        input_row_offsets=ro_t.as_unsafe_any_origin(),
        start_pos=sp_t.as_unsafe_any_origin(),
        freqs_cis=fc_t.as_unsafe_any_origin(),
        context=ctx,
        output_fn=output_fn,
    )
    ctx.synchronize()
    ctx.enqueue_copy(oh, ob)
    ctx.synchronize()

    # ── ours ─────────────────────────────────────────────────────────────
    var rope = RoPE[SEQ, HEADS, HD, THETA].make["cpu", Deterministic]()
    var xin = Tensor.alloc(B * N)
    for i in range(N):
        xin.data[i] = host[i]
    var ours = Tensor.alloc(B * N)
    rope.forward["cpu", B](TensorRefs[1](xin), ours, None)

    var compared = 0
    var bad = 0
    var worst = Scalar[DT](0)
    for i in range(N):
        compared += 1
        var d = abs(ours.data[i] - oh[i])
        if d > worst:
            worst = d
        if d > 1e-5:
            bad += 1
    print("  [1] vs MAX rope_ragged: compared", compared, " mismatched", bad,
          " worst_abs", worst)
    assert_true(compared == N, "must compare every element")
    assert_true(bad == 0, "our RoPE disagrees with MAX's kernel — the"
                          " interleaving convention is wrong")

    # ── [2] adjoint identity <RoPE(x), y> == <x, vjp(y)> ─────────────────
    var y = Tensor.alloc(B * N)
    for i in range(N):
        y.data[i] = Scalar[DT](((i * 53) % 23) - 11) * 0.07
    var gi = Tensor.alloc(B * N)
    rope.vjp["cpu", B](TensorRefs[1](xin), y, TensorRefs[1](gi), None)

    var lhs = Float64(0)
    var rhs = Float64(0)
    var scale = Float64(0)
    for i in range(N):
        var p = Float64(ours.data[i]) * Float64(y.data[i])
        lhs += p
        scale += abs(p)
        rhs += Float64(xin.data[i]) * Float64(gi.data[i])
    # ⚠ Normalised by the sum of |terms|, not by |lhs|: these inner products
    # cancel, and dividing by the cancelled result measures the cancellation
    # rather than the vjp. See `test_repeat_kv_heads.mojo`, where the same
    # identity reports 4e-7 against |lhs| and 6e-10 against sum|terms| for a
    # gap that is exactly fp32 epsilon.
    var rel = abs(lhs - rhs) / (scale + 1e-12)
    print("  [2] adjoint: <RoPE(x),y> =", lhs, " <x,vjp(y)> =", rhs,
          " sum|terms| =", scale, " rel =", rel)
    assert_true(abs(lhs) > 1e-6, "degenerate: the inner product is ~0, so the"
                                 " identity would hold vacuously")
    assert_true(rel < 1e-8, "backward is not the adjoint of forward")

    # ── [3] our GPU path == our CPU path ─────────────────────────────────
    # The GPU kernel is separate code from the CPU loop; checks [1] and [2]
    # only exercised the latter.
    var ropeg = RoPE[SEQ, HEADS, HD, THETA].make["gpu", Deterministic](
        Optional(ctx)
    )
    var xg = Tensor.alloc(B * N)
    for i in range(N):
        xg.data[i] = host[i]
    xg.upload(ctx)
    var og = Tensor.alloc(B * N)
    ropeg.forward["gpu", B](TensorRefs[1](xg), og, Optional(ctx))
    og.download(ctx)

    var gcmp = 0
    var gbad = 0
    var gworst = Scalar[DT](0)
    for i in range(N):
        gcmp += 1
        var d = abs(og.data[i] - ours.data[i])
        if d > gworst:
            gworst = d
        if d > 1e-6:
            gbad += 1
    print("  [3] GPU vs CPU: compared", gcmp, " mismatched", gbad,
          " worst_abs", gworst)
    assert_true(gcmp == N and gbad == 0, "our GPU RoPE disagrees with our CPU")

    var yg = Tensor.alloc(B * N)
    for i in range(N):
        yg.data[i] = y.data[i]
    yg.upload(ctx)
    var gig = Tensor.alloc(B * N)
    ropeg.vjp["gpu", B](TensorRefs[1](xg), yg, TensorRefs[1](gig), Optional(ctx))
    gig.download(ctx)
    var gvbad = 0
    for i in range(N):
        if abs(gig.data[i] - gi.data[i]) > 1e-6:
            gvbad += 1
    print("      GPU vjp vs CPU vjp: compared", N, " mismatched", gvbad)
    assert_true(gvbad == 0, "our GPU RoPE backward disagrees with our CPU")

    print()
    print("PASSED")
