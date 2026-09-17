# +--------------------------------------------------------------------------+ #
# | CrossAttention's GPU forward — four variants, at SigLIP's shape AND ACT's
# +--------------------------------------------------------------------------+ #
"""Which CrossAttention forward to ship, measured at the shapes that use it.

    pixi run -e jetson cross-attn-bench-jetson
    pixi run -e apple mojo run -I . benchmarks/cross_attention_bench.mojo

⚠⚠ WHY THIS EXISTS. nsys on the Orin: SmolVLA's two SigLIP towers spend ~1.88 s
of a 2.33 s query in this primitive's attention — the Q.Kt matmul through MAX's
naive batched matmul (49 ms/layer, 1.4% of fp32 peak), the softmax (20 ms), and
the A.V matmul (8.7 ms). The same method that took the expert's attention 6.1x
on the board (`smolvla_block_attention_bench.mojo`) is the obvious candidate.

⚠⚠ BUT THIS PRIMITIVE IS SHARED, and that changes what "wins" means. ACT uses
it four ways — encoder self-attention, decoder self and cross, and the CVAE
encoder, its ONLY masked user — and ACT runs at 30 Hz on the board today. So a
variant is only shippable if it is faster at SigLIP's shape AND no slower at
ACT's, at deploy batch (1) AND training batch (16). One run answers both.

⚠ ONE MORE REASON NOT TO ASSUME: the element-indexed kernels reached only
~1.9% of the Orin's peak on the expert. MAX's naive matmul already does 7.7% on
SigLIP's A.V. Replacing a 7.7% stage with a 1.9% one would be a regression that
a "same method worked last time" argument would ship. Hence C, which keeps the
matmul for A.V:

    A  the shipped forward: pack, bmm Q.Kt TRANSPOSED, softmax + cache, bmm A.V
    B  same, but Kt materialised CONTIGUOUS first  (is the transpose the cost?)
    C  element-indexed scores + row softmax INTO THE CACHE, bmm A.V
    D  element-indexed everything, no pack, no unpack

⚠ THE CACHE IS PART OF THE CONTRACT. The GPU backward reads the softmax weights
from `self.attn`, laid out [b, h, i, j] — so every variant is checked on its
cached weights as well as its output, against a FLOAT64 reference, in std
units. C and D write the scores straight into that layout, so the weights are
materialised once instead of into a scratch slab and then copied.
"""

from std.math import exp, sqrt
from std.gpu import global_idx
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.core.mm import bmm
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.primitives.cross_attention import (
    CrossAttention,
    XATTN_MASK_NEG,
    _xa_pack_kernel,
    _xa_softmax_kernel,
    _xa_unpack_kernel,
)
from mojo_rl.deep_agents.smolvla.block_attention import (
    BA_ROW_BLOCK,
    _ba_context_kernel,
    _ba_softmax_kernel,
)


comptime WARMUP = 2
comptime REPS = 8
comptime GATE_STD_UNITS = 1.0e-3
comptime POISON = 12345.0
"""Far outside any softmax weight (0..1) or context value (|v| <= 1 here)."""


# ═══════════════════════════════════════════════════════════════════════════
# the two new kernels (softmax and context are block_attention's, already gated)
# ═══════════════════════════════════════════════════════════════════════════


def _xe_scores_kernel[
    BATCH: Int, D: Int, NH: Int, Q: Int, KL: Int, H: Int, MASKED: Bool
](
    q: LayoutTensor[DT, Layout.row_major(BATCH, Q * D), MutAnyOrigin],
    k: LayoutTensor[DT, Layout.row_major(BATCH, KL * D), MutAnyOrigin],
    mask: LayoutTensor[DT, Layout.row_major(BATCH, KL), MutAnyOrigin],
    attn: LayoutTensor[DT, Layout.row_major(BATCH * NH * Q * KL), MutAnyOrigin],
):
    """One thread per (b, h, i, j), written straight into the cache layout.

    The key-padding mask is PER SAMPLE — `m[b, j]`, 1.0 attend / 0.0 ignore —
    which is this primitive's mask, not block_attention's `[Q, KV]` one.
    """
    var idx = Int(global_idx.x)
    if idx >= BATCH * NH * Q * KL:
        return
    var j = idx % KL
    var r = idx // KL
    var i = r % Q
    var r2 = r // Q
    var h = r2 % NH
    var b = r2 // NH
    var qb = b * (Q * D) + i * D + h * H
    var kb = b * (KL * D) + j * D + h * H
    var s = Scalar[DT](0)
    for d in range(H):
        s += rebind[Scalar[DT]](q.ptr[unsafe_offset = qb + d]) * rebind[
            Scalar[DT]
        ](k.ptr[unsafe_offset = kb + d])
    s = s * (Scalar[DT](1.0) / sqrt(Scalar[DT](H)))
    comptime if MASKED:
        if rebind[Scalar[DT]](mask.ptr[unsafe_offset = b * KL + j]) < Scalar[DT](0.5):
            s = XATTN_MASK_NEG
    attn.ptr[unsafe_offset = idx] = s


def _xt_transpose_kernel[BH: Int, KL: Int, H: Int](
    src: LayoutTensor[DT, Layout.row_major(BH * KL * H), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(BH * H * KL), MutAnyOrigin],
):
    """[BH, KL, H] -> [BH, H, KL], contiguous — variant B's only change."""
    var idx = Int(global_idx.x)
    if idx >= BH * H * KL:
        return
    var j = idx % KL
    var r = idx // KL
    var d = r % H
    var bh = r // H
    dst.ptr[unsafe_offset = idx] = rebind[Scalar[DT]](
        src.ptr[unsafe_offset = bh * KL * H + j * H + d]
    )


# ═══════════════════════════════════════════════════════════════════════════
# host
# ═══════════════════════════════════════════════════════════════════════════


struct Lcg(Movable):
    var s: UInt64

    def __init__(out self, seed: Int):
        self.s = UInt64(seed * 2 + 1)

    def __init__(out self, *, deinit move: Self):
        self.s = move.s

    def unit(mut self) -> Float32:
        self.s = self.s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        return Float32(Float64((self.s >> 40) & UInt64(0xFFFFFF)) / 16777216.0)


def _std_units(ref got: Tensor, ref refv: List[Float64]) -> Float64:
    """max |got - ref| / rms(ref). See `smolvla_block_attention_bench.mojo`
    for why relative-to-|ref| is the wrong question here."""
    var ss = 0.0
    for n in range(len(refv)):
        ss += refv[n] * refv[n]
    var rms = sqrt(ss / Float64(len(refv)))
    if rms < 1.0e-30:
        rms = 1.0
    var worst = 0.0
    for n in range(len(refv)):
        var e = abs(Float64(got.data[n]) - refv[n])
        if e > worst:
            worst = e
    return worst / rms


def _fmt(x: Float64, digits: Int) -> String:
    var p = 1.0
    for _ in range(digits):
        p *= 10.0
    return String(Float64(Int(x * p + 0.5)) / p)


def _pad(s: String, w: Int) -> String:
    var out = s
    while out.byte_length() < w:
        out += " "
    return out^


def run_shape[
    B: Int, DIM: Int, H: Int, QL: Int, KL: Int, MASKED: Bool
](name: String, mut ctx: DeviceContext) raises -> Bool:
    """All four variants at one shape. Returns False if any disagrees."""
    comptime HD = DIM // H
    comptime BH = B * H
    comptime QN = B * QL * DIM
    comptime KN = B * KL * DIM
    comptime PQ = BH * QL * HD
    comptime PK = BH * KL * HD
    comptime SC = BH * QL * KL
    comptime ATTN_SIZE = H * QL * KL
    comptime XA = CrossAttention[DIM, H, QL, KL, MASKED]

    var rng = Lcg(B * 7919 + QL * 131 + KL)
    var q = Tensor.alloc(QN)
    var k = Tensor.alloc(KN)
    var v = Tensor.alloc(KN)
    for n in range(QN):
        q.data[n] = Scalar[DT](rng.unit() * 2.0 - 1.0)
    for n in range(KN):
        k.data[n] = Scalar[DT](rng.unit() * 2.0 - 1.0)
        v.data[n] = Scalar[DT](rng.unit() * 2.0 - 1.0)
    # Per-sample valid prefix, at least 1 key — ACT's CVAE never masks cls/qpos.
    var m = Tensor.alloc(B * KL)
    for b in range(B):
        var valid = KL - (b % 7)
        if valid < 1:
            valid = 1
        for j in range(KL):
            m.data[b * KL + j] = Scalar[DT](1.0) if (not MASKED or j < valid) else Scalar[DT](0.0)

    # ── float64 reference: output AND the cached weights ─────────────────
    var ref_out = List[Float64](length=QN, fill=0.0)
    var ref_attn = List[Float64](length=SC, fill=0.0)
    var scale = 1.0 / sqrt(Float64(HD))
    for b in range(B):
        for h in range(H):
            for i in range(QL):
                var row = List[Float64](length=KL, fill=0.0)
                var mx = -1.0e300
                var any = False
                for j in range(KL):
                    if Float64(m.data[b * KL + j]) < 0.5:
                        row[j] = -1.0e300
                        continue
                    var s = 0.0
                    for d in range(HD):
                        s += Float64(q.data[b * QL * DIM + i * DIM + h * HD + d]) * Float64(
                            k.data[b * KL * DIM + j * DIM + h * HD + d]
                        )
                    row[j] = s * scale
                    if row[j] > mx:
                        mx = row[j]
                    any = True
                if not any:
                    continue
                var denom = 0.0
                for j in range(KL):
                    if row[j] > -1.0e299:
                        denom += exp(row[j] - mx)
                for j in range(KL):
                    if row[j] <= -1.0e299:
                        continue
                    var w = exp(row[j] - mx) / denom
                    ref_attn[((b * H + h) * QL + i) * KL + j] = w
                    for d in range(HD):
                        ref_out[b * QL * DIM + i * DIM + h * HD + d] += w * Float64(
                            v.data[b * KL * DIM + j * DIM + h * HD + d]
                        )

    q.upload(ctx)
    k.upload(ctx)
    v.upload(ctx)
    m.upload(ctx)

    # scratch shared by B, C, D
    var sq = Tensor()
    var sk = Tensor()
    var sv = Tensor()
    var skt = Tensor()
    var ss = Tensor()
    var pout = Tensor()
    sq.ensure_gpu(ctx, PQ)
    sk.ensure_gpu(ctx, PK)
    sv.ensure_gpu(ctx, PK)
    skt.ensure_gpu(ctx, PK)
    ss.ensure_gpu(ctx, SC)
    pout.ensure_gpu(ctx, PQ)

    comptime lay_q = Layout.row_major(B, QL * DIM)
    comptime lay_kv = Layout.row_major(B, KL * DIM)
    comptime lay_pq = Layout.row_major(PQ)
    comptime lay_pk = Layout.row_major(PK)
    comptime lay_s = Layout.row_major(SC)
    comptime lay_a = Layout.row_major(B, ATTN_SIZE)
    comptime lay_m = Layout.row_major(B, KL)
    comptime qblocks = (QN + TPB - 1) // TPB
    comptime kblocks = (KN + TPB - 1) // TPB

    print("")
    print(
        "── " + name + ": B " + String(B) + ", Q " + String(QL) + " x KV "
        + String(KL) + ", " + String(H) + " heads x " + String(HD)
        + (", MASKED" if MASKED else "") + " ──"
    )
    print(
        "   " + _pad(String("variant"), 36) + _pad(String("best ms"), 10)
        + _pad(String("vs A"), 8) + _pad(String("out (std u.)"), 24)
        + _pad(String("cache (std u.)"), 24) + _pad(String("out bits != A"), 16)
        + "cache bits != A"
    )

    var names: List[String] = [
        String("A  shipped (bmm Q.Kt transposed)"),
        String("B  Kt contiguous, bmm"),
        String("C  scores+softmax in cache, bmm A.V"),
        String("D  element-indexed, no pack"),
    ]
    var ok = True
    var best_a = 0.0
    # ⚠ A's output and cache as their OWN lists. Without a bit count, rows that
    # all print the same error could be four genuinely identical results OR four
    # reads of one buffer — opposite conclusions, the same printout.
    var a_out = List[Scalar[DT]]()
    var a_attn = List[Scalar[DT]]()
    for variant in range(4):
        var mod = XA.make["gpu", Deterministic](Optional(ctx))
        var out = Tensor()
        out.ensure_gpu(ctx, QN)
        var attn = Tensor()
        attn.ensure_gpu(ctx, B * ATTN_SIZE)
        # ⚠⚠ PRE-POISONED, AND THE BIT COUNT IS MEANINGLESS WITHOUT IT. Every
        # variant reports 0 bits different from A. A fresh device buffer can be
        # handed the memory A just freed, STILL HOLDING A's ANSWER — so a
        # variant whose kernel silently wrote nowhere would match A exactly.
        # A sentinel no attention output can produce makes an unwritten
        # element fail loudly instead.
        out.dev.value().enqueue_fill(Scalar[DT](POISON))
        attn.dev.value().enqueue_fill(Scalar[DT](POISON))
        var best = 1.0e30
        for rep in range(WARMUP + REPS):
            var t0 = perf_counter_ns()
            if variant == 0:
                # ⚠ `rebind` BRIDGES A SYMBOLIC ARITY. `forward` takes
                # `TensorRefs[4 if MASKED else 3]`, which a literal 4 or 3
                # does not unify with while MASKED is still a parameter; inside
                # each branch the two ARE the same type once instantiated.
                # `cross_attention.mojo` bridges its own mask slot the same way.
                comptime if MASKED:
                    mod.forward["gpu", B](
                        rebind[TensorRefs[XA.ARITY, MutAnyOrigin]](
                            TensorRefs[4, MutAnyOrigin](q, k, v, m)
                        ), out, ctx,
                    )
                else:
                    mod.forward["gpu", B](
                        rebind[TensorRefs[XA.ARITY, MutAnyOrigin]](
                            TensorRefs[3, MutAnyOrigin](q, k, v)
                        ), out, ctx,
                    )
            else:
                if variant == 1 or variant == 2:
                    # V head-major, for the A.V matmul.
                    ctx.enqueue_function[
                        _xa_pack_kernel[B, DIM, H, KL, HD, PK]
                    ](sv.lt["gpu", lay_pk](), v.lt["gpu", lay_kv](),
                      grid_dim=kblocks, block_dim=TPB)
                if variant == 1:
                    ctx.enqueue_function[
                        _xa_pack_kernel[B, DIM, H, QL, HD, PQ]
                    ](sq.lt["gpu", lay_pq](), q.lt["gpu", lay_q](),
                      grid_dim=qblocks, block_dim=TPB)
                    ctx.enqueue_function[
                        _xa_pack_kernel[B, DIM, H, KL, HD, PK]
                    ](sk.lt["gpu", lay_pk](), k.lt["gpu", lay_kv](),
                      grid_dim=kblocks, block_dim=TPB)
                    ctx.enqueue_function[_xt_transpose_kernel[BH, KL, HD]](
                        sk.lt["gpu", lay_pk](), skt.lt["gpu", lay_pk](),
                        grid_dim=(PK + TPB - 1) // TPB, block_dim=TPB,
                    )
                    bmm[A0=BH, A1=QL, A2=HD, B0=BH, B1=HD, B2=KL, O0=BH, O1=QL, O2=KL](
                        ss.dev.value(), sq.dev.value(), skt.dev.value(), ctx
                    )
                    ctx.enqueue_function[
                        _xa_softmax_kernel[B, H, QL, KL, HD, MASKED, ATTN_SIZE, SC, BH]
                    ](ss.lt["gpu", lay_s](), attn.lt["gpu", lay_a](),
                      m.lt["gpu", lay_m](), grid_dim=BH, block_dim=TPB)
                    bmm[A0=BH, A1=QL, A2=KL, B0=BH, B1=KL, B2=HD, O0=BH, O1=QL, O2=HD](
                        pout.dev.value(), ss.dev.value(), sv.dev.value(), ctx
                    )
                else:
                    ctx.enqueue_function[
                        _xe_scores_kernel[B, DIM, H, QL, KL, HD, MASKED]
                    ](q.lt["gpu", lay_q](), k.lt["gpu", lay_kv](),
                      m.lt["gpu", lay_m](), attn.lt["gpu", lay_s](),
                      grid_dim=(SC + TPB - 1) // TPB, block_dim=TPB)
                    ctx.enqueue_function[_ba_softmax_kernel[B, H, QL, KL]](
                        attn.lt["gpu", lay_s](),
                        grid_dim=(BH * QL + BA_ROW_BLOCK - 1) // BA_ROW_BLOCK,
                        block_dim=BA_ROW_BLOCK,
                    )
                    if variant == 2:
                        bmm[A0=BH, A1=QL, A2=KL, B0=BH, B1=KL, B2=HD, O0=BH, O1=QL, O2=HD](
                            pout.dev.value(), attn.dev.value(), sv.dev.value(), ctx
                        )
                    else:
                        ctx.enqueue_function[_ba_context_kernel[B, DIM, H, QL, KL, HD]](
                            attn.lt["gpu", lay_s](), v.lt["gpu", lay_kv](),
                            out.lt["gpu", lay_q](),
                            grid_dim=(PQ + TPB - 1) // TPB, block_dim=TPB,
                        )
                if variant == 1 or variant == 2:
                    ctx.enqueue_function[
                        _xa_unpack_kernel[B, DIM, H, QL, HD, PQ]
                    ](out.lt["gpu", lay_q](), pout.lt["gpu", lay_pq](),
                      grid_dim=qblocks, block_dim=TPB)
            # ⚠ SYNCHRONISE BEFORE STOPPING THE CLOCK.
            ctx.synchronize()
            var dt = Float64(perf_counter_ns() - t0) / 1e6
            if rep >= WARMUP and dt < best:
                best = dt
        out.download(ctx)
        var cache_err = 0.0
        if variant == 0:
            mod.attn.download(ctx)
            cache_err = _std_units(mod.attn, ref_attn)
        else:
            attn.download(ctx)
            cache_err = _std_units(attn, ref_attn)
        var out_err = _std_units(out, ref_out)
        var out_bits = 0
        var attn_bits = 0
        if variant == 0:
            best_a = best
            for n in range(QN):
                a_out.append(out.data[n])
            for n in range(SC):
                a_attn.append(mod.attn.data[n])
        else:
            for n in range(QN):
                if out.data[n] != a_out[n]:
                    out_bits += 1
            for n in range(SC):
                if attn.data[n] != a_attn[n]:
                    attn_bits += 1
        var flag = String("")
        if out_err > GATE_STD_UNITS or cache_err > GATE_STD_UNITS:
            flag = "   ⚠⚠ DISAGREES"
            ok = False
        print(
            "   " + _pad(names[variant], 36) + _pad(_fmt(best, 3), 10)
            + _pad(_fmt(best_a / best, 2) + "x", 8)
            + _pad(String(out_err), 24) + _pad(String(cache_err), 24)
            + _pad(String(out_bits) + "/" + String(QN), 16)
            + String(attn_bits) + "/" + String(SC) + flag
        )
    return ok


def main() raises:
    comptime assert has_accelerator(), "this benchmark times GPU kernels"
    var ctx = DeviceContext()
    print("=" * 100)
    print("CrossAttention forward — four variants, " + String(ctx.name()))
    print("=" * 100)
    var ok = True
    # SigLIP-B/16 @ 512: the target. B=1 — one tower call per camera.
    if not run_shape[1, 768, 12, 1024, 1024, False](String("SigLIP self-attention"), ctx):
        ok = False
    # ACT, training batch and deploy batch, every way it uses the primitive.
    if not run_shape[16, 256, 8, 162, 162, False](String("ACT encoder self (train)"), ctx):
        ok = False
    if not run_shape[1, 256, 8, 60, 162, False](String("ACT decoder cross (deploy)"), ctx):
        ok = False
    if not run_shape[16, 256, 8, 62, 62, True](String("ACT CVAE encoder, masked (train)"), ctx):
        ok = False
    print("")
    if not ok:
        raise Error("a variant produced different numbers — its timing is not a result")
    print("  every variant within 1e-3 std units of float64, output AND cached weights")
