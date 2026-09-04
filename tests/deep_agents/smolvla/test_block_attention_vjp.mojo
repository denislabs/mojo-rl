# +--------------------------------------------------------------------------+ #
# | BlockCrossAttention.vjp — against central differences of the definition
# +--------------------------------------------------------------------------+ #
"""The first gate of V2. It checks a gradient against nothing this repo wrote.

    pixi run -e apple mojo run -I . \\
        tests/deep_agents/smolvla/test_block_attention_vjp.mojo

`BlockCrossAttention` is the leaf V1 had to write because no `nn` attention
fits the denoising step, and it shipped forward-only. Fine-tuning needs its
backward, and a backward is the easiest thing in this codebase to get
*plausibly* wrong: a dropped term, a transposed index or a missing scale all
produce finite gradients of the right shape and a training run that simply
learns worse than it should. Nothing downstream raises.

So the reference here is **not our forward**. It is a Float64 transcription of
the mathematical definition, written in this file, differentiated by central
differences:

    L(q,k,v) = Σ_t g_t · out_t(q,k,v)        g fixed, arbitrary
    ∂L/∂x_t ≈ [L(x_t + h) − L(x_t − h)] / 2h

and EVERY input component is probed — all 1,472 of them — not a sample. Two
independent errors would have to cancel across all of them for this to pass on
a wrong gradient.

⚠ Central differences, not forward differences. The one-sided error is O(h) and
would need a tolerance loose enough to hide a real 1% defect; the two-sided
error is O(h²), which is what buys the 1e-3 relative band below.

⚠ The FD reference is Float64 and the analytic gradient recomputes its
probabilities in the forward's fp32. They therefore agree to about 1e-7
relative, not to bit-parity, and the band is set from that — not from what the
implementation happened to produce.

## The corners a random probe does not reach

  * **A query row masked entirely.** The forward floors the denominator and
    emits a zero context; the backward must emit an exactly-zero gradient and
    no NaN, since 0/0 anywhere in the softmax backward would poison a whole
    training step silently.
  * **A key no query attends to.** Column 8 of the mask is closed for every
    row, so `dK[.., 8, ..]` and `dV[.., 8, ..]` must be EXACTLY zero. This is
    what catches a backward that re-derives the mask instead of inheriting it
    through `p = 0`, or one that reads the mask with a transposed index.

## What a broken backward looks like here — MEASURED, not asserted

Four defects introduced into the CPU `vjp` one at a time and reverted, with
what leg [2] actually printed:

    defect                              dQ wrong   dK wrong   dV wrong  worst rel
    dropped the `− dot` term            250/320    511/576      0/576       31.4
    dQ reduced against q, not k         256/320      0/576      0/576        1.00
    dropped the 1/sqrt(HD) on ds        256/320    512/576      0/576        1.83
    transposed mask read (j*QL+i)       256/320    192/576    192/576       43.5

Two things that table says and a passing run cannot:

  * Every defect is caught, and by a wide margin — the smallest is 1.00
    relative, a thousand times the 1e-3 band. There is no defect here that
    "just barely" fails.
  * ⚠ **dV was untouched by three of the four.** It is `Σ_i p·g`, so it sees
    nothing wrong with the softmax Jacobian, the scale, or the reduction
    index. A gate that spot-checked dV — the easiest of the three to write a
    reference for — would have passed all three of those.

Legs [3] and [4] never got to run in any of the four: leg [2] aborts first. So
they are not load-bearing against these defects, and they are not there for
them. [4] in particular states something [2] structurally cannot: leg [2]
compares with a relative band floored at 1e-6, so a spurious 1e-9 leaking into
a closed key column would pass it. [4] demands exact zero.
"""

from std.math import abs, exp, sqrt
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.deep_agents.smolvla.block_attention import (
    BlockCrossAttention, BA_MASK_NEG,
)

comptime DIM = 32
comptime HEADS = 4
comptime QL = 5
comptime KL = 9
comptime HD = DIM // HEADS
comptime B = 2
comptime QN = QL * DIM
comptime KN = KL * DIM
comptime BA = BlockCrossAttention[DIM, HEADS, QL, KL]

comptime CLOSED_J = 8      # a key column no query may attend to
comptime CLOSED_I = 4      # a query row that attends to nothing
comptime FD_H = 1.0e-4


def _ref_out(
    ref q: List[Float64], ref k: List[Float64], ref v: List[Float64],
    ref mask: List[Scalar[DT]],
) -> List[Float64]:
    """Masked attention straight from the definition, in Float64.

    ⚠ Deliberately NOT a call into `BlockCrossAttention.forward`. A gate whose
    reference shares an implementation with the thing under test can only see
    the disagreements, never the shared mistake.
    """
    var out = List[Float64](length=B * QN, fill=0.0)
    var scale = 1.0 / sqrt(Float64(HD))
    for b in range(B):
        for h in range(HEADS):
            for i in range(QL):
                var qb = b * QN + i * DIM + h * HD
                var sc = List[Float64](length=KL, fill=0.0)
                var mx = -1.0e300
                for j in range(KL):
                    if mask[i * KL + j] <= BA_MASK_NEG:
                        continue
                    var kb = b * KN + j * DIM + h * HD
                    var s = 0.0
                    for d in range(HD):
                        s += q[qb + d] * k[kb + d]
                    sc[j] = s * scale
                    if sc[j] > mx:
                        mx = sc[j]
                var den = 0.0
                for j in range(KL):
                    if mask[i * KL + j] <= BA_MASK_NEG:
                        continue
                    var w = exp(sc[j] - mx)
                    den += w
                    sc[j] = w
                    var kb = b * KN + j * DIM + h * HD
                    for d in range(HD):
                        out[qb + d] += w * v[kb + d]
                var inv = 0.0
                if den > 1.0e-30:
                    inv = 1.0 / den
                for d in range(HD):
                    out[qb + d] *= inv
    return out^


def _loss(
    ref q: List[Float64], ref k: List[Float64], ref v: List[Float64],
    ref mask: List[Scalar[DT]], ref g: List[Float64],
) -> Float64:
    var o = _ref_out(q, k, v, mask)
    var s = 0.0
    for i in range(B * QN):
        s += g[i] * o[i]
    return s


def _poke(
    which: Int, t: Int, val: Float64,
    mut q: List[Float64], mut k: List[Float64], mut v: List[Float64],
):
    if which == 0:
        q[t] = val
    elif which == 1:
        k[t] = val
    else:
        v[t] = val


def _fd(
    which: Int, t: Int,
    mut q: List[Float64], mut k: List[Float64], mut v: List[Float64],
    ref mask: List[Scalar[DT]], ref g: List[Float64],
) -> Float64:
    """Central difference of `_loss` in one component. It is restored exactly.

    `which` selects q / k / v rather than taking the array by reference: the
    same list would then be passed twice, once mutably, and Mojo rejects the
    alias.
    """
    var keep: Float64
    if which == 0:
        keep = q[t]
    elif which == 1:
        keep = k[t]
    else:
        keep = v[t]
    _poke(which, t, keep + FD_H, q, k, v)
    var lp = _loss(q, k, v, mask, g)
    _poke(which, t, keep - FD_H, q, k, v)
    var lm = _loss(q, k, v, mask, g)
    _poke(which, t, keep, q, k, v)
    return (lp - lm) / (2.0 * FD_H)


struct Cmp(Movable):
    """Compared / differing, with the worst offender kept."""
    var n: Int
    var bad: Int
    var worst: Float64
    var at: Int

    def __init__(out self):
        self.n = 0
        self.bad = 0
        self.worst = 0.0
        self.at = -1

    def __init__(out self, *, deinit move: Self):
        self.n = move.n
        self.bad = move.bad
        self.worst = move.worst
        self.at = move.at

    def add(mut self, got: Float64, want: Float64, idx: Int):
        self.n += 1
        var scale = abs(want)
        if scale < 1.0e-6:
            scale = 1.0e-6
        var rel = abs(got - want) / scale
        if rel > self.worst:
            self.worst = rel
            self.at = idx
        if rel > 1.0e-3:
            self.bad += 1

    def report(self, name: String):
        print(
            "      " + name + ": compared " + String(self.n) + ", differing "
            + String(self.bad) + "  (worst rel " + String(self.worst)
            + " at " + String(self.at) + ")"
        )


def main() raises:
    print("=" * 70)
    print("BlockCrossAttention.vjp vs central differences of the definition")
    print("=" * 70)

    # ── the mask ─────────────────────────────────────────────────────────
    var mask = List[Scalar[DT]]()
    for i in range(QL):
        for j in range(KL):
            var ok: Bool
            if j == CLOSED_J:
                ok = False                    # a key nothing attends to
            elif i == CLOSED_I:
                ok = False                    # a query that attends to nothing
            elif i < 2:
                ok = True
            else:
                ok = (j < 4) or (j <= 4 + (i - 2))
            mask.append(Scalar[DT](0.0) if ok else BA_MASK_NEG)

    # ── inputs. Float64 masters; the fp32 copies are derived from them, so
    #    the two paths differentiate the same numbers. ────────────────────
    var qh = List[Float64]()
    var kh = List[Float64]()
    var vh = List[Float64]()
    var gh = List[Float64]()
    for i in range(B * QN):
        qh.append(Float64(((i * 37) % 19) - 9) * 0.1)
        gh.append(Float64(((i * 61) % 13) - 6) * 0.15)
    for i in range(B * KN):
        kh.append(Float64(((i * 53) % 23) - 11) * 0.07)
        vh.append(Float64(((i * 29) % 17) - 8) * 0.05)

    var qt = Tensor.alloc(B * QN)
    var kt = Tensor.alloc(B * KN)
    var vt = Tensor.alloc(B * KN)
    var gt = Tensor.alloc(B * QN)
    for i in range(B * QN):
        qt.data[i] = Scalar[DT](qh[i])
        gt.data[i] = Scalar[DT](gh[i])
    for i in range(B * KN):
        kt.data[i] = Scalar[DT](kh[i])
        vt.data[i] = Scalar[DT](vh[i])

    # ── [1] the analytic gradient ────────────────────────────────────────
    var ba = BA.make["cpu"](mask)
    var dq = Tensor.alloc(B * QN)
    var dk = Tensor.alloc(B * KN)
    var dv = Tensor.alloc(B * KN)
    ba.vjp["cpu", B](qt, kt, vt, gt, dq, dk, dv, None)
    print("  [1] analytic dQ/dK/dV computed  (" + String(B * QN) + " + "
          + String(B * KN) + " + " + String(B * KN) + " components)")

    # ── [2] every component against a central difference ─────────────────
    print("  [2] central differences of the Float64 definition, h =",
          FD_H)
    var cq = Cmp()
    for t in range(B * QN):
        cq.add(Float64(dq.data[t]), _fd(0, t, qh, kh, vh, mask, gh), t)
    cq.report("dQ")
    var ck = Cmp()
    for t in range(B * KN):
        ck.add(Float64(dk.data[t]), _fd(1, t, qh, kh, vh, mask, gh), t)
    ck.report("dK")
    var cv = Cmp()
    for t in range(B * KN):
        cv.add(Float64(dv.data[t]), _fd(2, t, qh, kh, vh, mask, gh), t)
    cv.report("dV")

    var total = cq.n + ck.n + cv.n
    assert_true(
        total == B * QN + 2 * B * KN,
        "every input component must be probed, not a sample",
    )
    assert_true(cq.bad == 0, "dQ disagrees with a central difference")
    assert_true(ck.bad == 0, "dK disagrees with a central difference")
    assert_true(cv.bad == 0, "dV disagrees with a central difference")

    # ── [3] the fully-masked query row ───────────────────────────────────
    var nan = 0
    var nz = 0
    for b in range(B):
        for d in range(DIM):
            var y = dq.data[b * QN + CLOSED_I * DIM + d]
            if y != y:
                nan += 1
            elif y != Scalar[DT](0):
                nz += 1
    print("  [3] query row", CLOSED_I, "attends to nothing: nan", nan,
          " nonzero", nz, " (of", B * DIM, ")")
    assert_true(nan == 0, "a fully-masked query row produced NaN in dQ")
    assert_true(nz == 0, "a fully-masked query row must have zero gradient")

    # ── [4] the closed key column ────────────────────────────────────────
    var knz = 0
    var vnz = 0
    for b in range(B):
        for d in range(DIM):
            if dk.data[b * KN + CLOSED_J * DIM + d] != Scalar[DT](0):
                knz += 1
            if dv.data[b * KN + CLOSED_J * DIM + d] != Scalar[DT](0):
                vnz += 1
    print("  [4] key column", CLOSED_J, "is closed to every query: dK nonzero",
          knz, " dV nonzero", vnz, " (of", B * DIM, "each)")
    assert_true(knz == 0, "a key no query attends to has a nonzero dK")
    assert_true(vnz == 0, "a key no query attends to has a nonzero dV")

    # ── [5] GPU vs CPU ───────────────────────────────────────────────────
    # ⚠ Not bit-parity, and not expected to be: the CPU path accumulates the
    # softmax backward in Float64 and Metal has no fp64, so the GPU path is
    # fp32 throughout. Agreement here is a statement about conditioning.
    var c = DeviceContext()
    var gq = Tensor.alloc(B * QN)
    var gk = Tensor.alloc(B * KN)
    var gv = Tensor.alloc(B * KN)
    var gg = Tensor.alloc(B * QN)
    for i in range(B * QN):
        gq.data[i] = Scalar[DT](qh[i])
        gg.data[i] = Scalar[DT](gh[i])
    for i in range(B * KN):
        gk.data[i] = Scalar[DT](kh[i])
        gv.data[i] = Scalar[DT](vh[i])
    gq.upload(c)
    gk.upload(c)
    gv.upload(c)
    gg.upload(c)

    var gba = BA.make["gpu"](mask, Optional(c))
    var Gdq = Tensor.alloc(B * QN)
    var Gdk = Tensor.alloc(B * KN)
    var Gdv = Tensor.alloc(B * KN)
    Gdq.upload(c)
    Gdk.upload(c)
    Gdv.upload(c)
    gba.vjp["gpu", B](gq, gk, gv, gg, Gdq, Gdk, Gdv, Optional(c))
    c.synchronize()
    Gdq.download(c)
    Gdk.download(c)
    Gdv.download(c)

    var gc = Cmp()
    for i in range(B * QN):
        gc.add(Float64(Gdq.data[i]), Float64(dq.data[i]), i)
    for i in range(B * KN):
        gc.add(Float64(Gdk.data[i]), Float64(dk.data[i]), B * QN + i)
        gc.add(Float64(Gdv.data[i]), Float64(dv.data[i]),
               B * QN + B * KN + i)
    gc.report("GPU vs CPU (dQ|dK|dV)")
    assert_true(
        gc.n == B * QN + 2 * B * KN, "GPU leg must compare every component"
    )
    assert_true(gc.bad == 0, "the GPU backward disagrees with the CPU one")

    print()
    print("PASSED — " + String(total) + " components against central"
          " differences, " + String(gc.n) + " against the GPU")
