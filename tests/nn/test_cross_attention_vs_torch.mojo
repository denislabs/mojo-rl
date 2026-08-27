# +--------------------------------------------------------------------------+ #
# | M1 gate — CrossAttention vs PyTorch
# +--------------------------------------------------------------------------+ #
"""Gates `nn.primitives.CrossAttention` against PyTorch's attention arithmetic.

    pixi run -e act-ref python tools/act/dump_act_reference.py --out /tmp/act_ref
    pixi run mojo build -I . -o /tmp/t tests/nn/test_cross_attention_vs_torch.mojo && /tmp/t

The reference runs at float64 and our leaf at float32, so the tolerances below
are float32 accumulation limits, not agreement limits. What is being checked:

  * forward at `Q_LEN != KV_LEN` (5 queries over 7 keys) — the case
    `ScaledDotProductAttention` cannot represent at all;
  * the softmax weights themselves, per head — an error in head splitting
    survives the output check when heads happen to average out;
  * a PER-SAMPLE key padding mask at three different lengths (7/4/1 valid) —
    the case `MaskedAttention`'s batch-shared `[SEQ,SEQ]` bias cannot represent;
  * dq/dk/dv against autograd, seeded by a fixed random grad_output rather than
    ones (a ones-seed hides per-position errors);
  * the DETR self-attention shape: pos added to q and k but NOT v.

Plus a finite-difference gradcheck that depends on no reference at all.
"""

from std.math import sqrt

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.primitives.cross_attention import CrossAttention


comptime REF_DIR = "/tmp/act_ref"

comptime B = 3
comptime HEADS = 4
comptime DIM = 16
comptime QL = 5
comptime KL = 7
comptime SL = 6  # self-attention case

comptime TOL_FWD = 2e-6
comptime TOL_GRAD = 5e-6


def check(mut fails: Int, name: String, ok: Bool, detail: String = String("")):
    if ok:
        print("  PASS  " + name + ("  " + detail if detail else ""))
    else:
        fails += 1
        print("  FAIL  " + name + ("  " + detail if detail else ""))


def load(name: String, n: Int) raises -> List[Scalar[DT]]:
    """Read one float32 blob from the reference dump."""
    var path = String(REF_DIR) + "/" + name + ".bin"
    var f = open(path, "r")
    var bytes = f.read_bytes()
    f.close()
    if len(bytes) != n * 4:
        raise Error(
            "gate: " + name + ".bin is " + String(len(bytes))
            + " bytes, expected " + String(n * 4)
            + " — regenerate with tools/act/dump_act_reference.py"
        )
    var out = List[Scalar[DT]](unsafe_uninit_length=n)
    var p = bytes.unsafe_ptr().unsafe_bitcast[Scalar[DT]]()
    for i in range(n):
        out[i] = p[unsafe_offset=i]
    _ = bytes^
    return out^


def max_abs_diff(
    ref a: List[Scalar[DT]], ref b: List[Scalar[DT]], n: Int
) -> Float64:
    var w = Float64(0.0)
    for i in range(n):
        w = max(w, abs(Float64(a[i]) - Float64(b[i])))
    return w


def max_abs_diff_t(
    mut t: Tensor, ref b: List[Scalar[DT]], n: Int
) -> Float64:
    var w = Float64(0.0)
    for i in range(n):
        w = max(w, abs(Float64(t.data[i]) - Float64(b[i])))
    return w


def fill(mut t: Tensor, ref src: List[Scalar[DT]], n: Int):
    t.ensure(n)
    for i in range(n):
        t.data[i] = src[i]


def main() raises:
    var fails = 0
    print("CrossAttention gate (reference: " + String(REF_DIR) + ")")
    print("")

    comptime QN = B * QL * DIM
    comptime KN = B * KL * DIM
    comptime MN = B * KL

    var q_ref = load(String("xattn_q"), QN)
    var k_ref = load(String("xattn_k"), KN)
    var v_ref = load(String("xattn_v"), KN)
    var valid = load(String("xattn_valid"), MN)
    var gout = load(String("xattn_gout"), QN)

    # ── unmasked forward + backward ──────────────────────────────────────
    var m = CrossAttention[DIM, HEADS, QL, KL, False].make[
        "cpu", Kaiming
    ]()

    var pack = TensorPack[3]()
    fill(pack[0], q_ref, QN)
    fill(pack[1], k_ref, KN)
    fill(pack[2], v_ref, KN)
    var out = Tensor()

    var inputs = TensorRefs[3, MutAnyOrigin](pack[0], pack[1], pack[2])
    m.forward["cpu", B](inputs, out)

    var out_ref = load(String("xattn_out_plain"), QN)
    check(
        fails,
        "forward, Q_LEN=5 over KV_LEN=7",
        max_abs_diff_t(out, out_ref, QN) < TOL_FWD,
        "max|diff| = " + String(max_abs_diff_t(out, out_ref, QN)),
    )

    # The softmax weights themselves, per head. An output-only check passes
    # even when heads are permuted, because attn·V then averages the error away.
    comptime AN = B * HEADS * QL * KL
    var attn_ref = load(String("xattn_attn_plain"), AN)
    var attn_d = max_abs_diff_t(m.attn, attn_ref, AN)
    check(
        fails,
        "per-head softmax weights",
        attn_d < TOL_FWD,
        "max|diff| = " + String(attn_d),
    )

    var gpack = TensorPack[3]()
    gpack[0].ensure(QN)
    gpack[1].ensure(KN)
    gpack[2].ensure(KN)
    var gout_t = Tensor()
    fill(gout_t, gout, QN)
    var ginputs = TensorRefs[3, MutAnyOrigin](gpack[0], gpack[1], gpack[2])
    m.vjp["cpu", B](inputs, gout_t, ginputs)

    var dq_ref = load(String("xattn_dq_plain"), QN)
    var dk_ref = load(String("xattn_dk_plain"), KN)
    var dv_ref = load(String("xattn_dv_plain"), KN)
    check(
        fails,
        "dq vs autograd",
        max_abs_diff_t(gpack[0], dq_ref, QN) < TOL_GRAD,
        "max|diff| = " + String(max_abs_diff_t(gpack[0], dq_ref, QN)),
    )
    check(
        fails,
        "dk vs autograd",
        max_abs_diff_t(gpack[1], dk_ref, KN) < TOL_GRAD,
        "max|diff| = " + String(max_abs_diff_t(gpack[1], dk_ref, KN)),
    )
    check(
        fails,
        "dv vs autograd",
        max_abs_diff_t(gpack[2], dv_ref, KN) < TOL_GRAD,
        "max|diff| = " + String(max_abs_diff_t(gpack[2], dv_ref, KN)),
    )

    # ── masked: three different valid lengths in one batch ───────────────
    var mm = CrossAttention[DIM, HEADS, QL, KL, True].make["cpu", Kaiming]()
    var mpack = TensorPack[4]()
    fill(mpack[0], q_ref, QN)
    fill(mpack[1], k_ref, KN)
    fill(mpack[2], v_ref, KN)
    fill(mpack[3], valid, MN)
    var mout = Tensor()
    var minputs = TensorRefs[4, MutAnyOrigin](mpack[0], mpack[1], mpack[2], mpack[3])
    mm.forward["cpu", B](minputs, mout)

    var mout_ref = load(String("xattn_out_masked"), QN)
    var mdiff = max_abs_diff_t(mout, mout_ref, QN)
    check(
        fails,
        "masked forward (valid 7/4/1 per sample)",
        mdiff < TOL_FWD,
        "max|diff| = " + String(mdiff),
    )

    var mattn_ref = load(String("xattn_attn_masked"), AN)
    check(
        fails,
        "masked softmax weights",
        max_abs_diff_t(mm.attn, mattn_ref, AN) < TOL_FWD,
        "max|diff| = " + String(max_abs_diff_t(mm.attn, mattn_ref, AN)),
    )

    # A masked key must receive EXACTLY zero weight, not merely a small one.
    # `exp(-1e30)` underflowing is the mechanism; an additive bias that was
    # only "very negative" would leak.
    var leak = Float64(0.0)
    for b in range(B):
        for h in range(HEADS):
            for i in range(QL):
                for j in range(KL):
                    if valid[b * KL + j] < Scalar[DT](0.5):
                        leak = max(
                            leak,
                            abs(
                                Float64(
                                    mm.attn.data[
                                        b * HEADS * QL * KL
                                        + h * QL * KL
                                        + i * KL
                                        + j
                                    ]
                                )
                            ),
                        )
    check(
        fails,
        "masked keys get exactly zero weight",
        leak == 0.0,
        "max weight on a masked key = " + String(leak),
    )
    # ...and the masked run must actually DIFFER from the unmasked one, or the
    # two checks above are both passing on the same (unmasked) computation.
    var mask_effect = max_abs_diff_t(mout, out_ref, QN)
    check(
        fails,
        "the mask changes the result",
        mask_effect > 0.01,
        "max|masked-plain| = " + String(mask_effect),
    )

    var mgpack = TensorPack[4]()
    mgpack[0].ensure(QN)
    mgpack[1].ensure(KN)
    mgpack[2].ensure(KN)
    mgpack[3].ensure(MN)
    var mgout = Tensor()
    fill(mgout, gout, QN)
    var mginputs = TensorRefs[4, MutAnyOrigin](mgpack[0], mgpack[1], mgpack[2], mgpack[3])
    mm.vjp["cpu", B](minputs, mgout, mginputs)

    var mdq = load(String("xattn_dq_masked"), QN)
    var mdk = load(String("xattn_dk_masked"), KN)
    var mdv = load(String("xattn_dv_masked"), KN)
    check(
        fails,
        "masked dq vs autograd",
        max_abs_diff_t(mgpack[0], mdq, QN) < TOL_GRAD,
        "max|diff| = " + String(max_abs_diff_t(mgpack[0], mdq, QN)),
    )
    check(
        fails,
        "masked dk vs autograd",
        max_abs_diff_t(mgpack[1], mdk, KN) < TOL_GRAD,
        "max|diff| = " + String(max_abs_diff_t(mgpack[1], mdk, KN)),
    )
    check(
        fails,
        "masked dv vs autograd",
        max_abs_diff_t(mgpack[2], mdv, KN) < TOL_GRAD,
        "max|diff| = " + String(max_abs_diff_t(mgpack[2], mdv, KN)),
    )

    # ── DETR self-attention: pos on q and k, not v ───────────────────────
    comptime SN = B * SL * DIM
    var sx = load(String("xattn_selfx"), SN)
    var spos = load(String("xattn_selfpos"), SN)
    var sm = CrossAttention[DIM, HEADS, SL, SL, False].make["cpu", Kaiming]()
    var spack = TensorPack[3]()
    spack[0].ensure(SN)
    spack[1].ensure(SN)
    spack[2].ensure(SN)
    for i in range(SN):
        spack[0].data[i] = sx[i] + spos[i]  # q = x + pos
        spack[1].data[i] = sx[i] + spos[i]  # k = x + pos
        spack[2].data[i] = sx[i]  # v = x  (no pos)
    var sout = Tensor()
    var sinputs = TensorRefs[3, MutAnyOrigin](spack[0], spack[1], spack[2])
    sm.forward["cpu", B](sinputs, sout)
    var sref = load(String("xattn_self_out"), SN)
    check(
        fails,
        "DETR self-attention (pos on q,k not v)",
        max_abs_diff_t(sout, sref, SN) < TOL_FWD,
        "max|diff| = " + String(max_abs_diff_t(sout, sref, SN)),
    )

    # ── finite-difference gradcheck (no reference involved) ──────────────
    # Central differences on a handful of coordinates of each input. fp32 with
    # eps=1e-2 keeps truncation and round-off both near 1e-4, so the tolerance
    # is loose by construction; this catches a wrong-shape or wrong-sign VJP,
    # not a last-ulp one. The autograd comparisons above do the tight work.
    var fd_worst = Float64(0.0)
    for which in range(3):
        var n = QN if which == 0 else KN
        for probe in range(5):
            var idx = (probe * 37 + which * 11) % n
            var eps = Scalar[DT](1e-2)
            var saved = pack[which].data[idx]

            pack[which].data[idx] = saved + eps
            var op = Tensor()
            m.forward["cpu", B](inputs, op)
            var lp = Float64(0.0)
            for i in range(QN):
                lp += Float64(op.data[i]) * Float64(gout[i])

            pack[which].data[idx] = saved - eps
            var om = Tensor()
            m.forward["cpu", B](inputs, om)
            var lm = Float64(0.0)
            for i in range(QN):
                lm += Float64(om.data[i]) * Float64(gout[i])

            pack[which].data[idx] = saved
            var numeric = (lp - lm) / (2.0 * Float64(eps))
            var analytic = Float64(gpack[which].data[idx])
            fd_worst = max(
                fd_worst, abs(numeric - analytic) / (1.0 + abs(analytic))
            )
    check(
        fails,
        "finite-difference gradcheck (15 coordinates)",
        fd_worst < 1e-3,
        "max rel err = " + String(fd_worst),
    )

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("cross attention gate failed")
