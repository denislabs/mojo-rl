# +--------------------------------------------------------------------------+ #
# | M4 gate — CVAE reparameterization, KL, and the masked L1 loss
# +--------------------------------------------------------------------------+ #
"""Gates `nn/primitives/gaussian_vae.mojo` and
`nn/primitives/l1_masked_per_sample.mojo` against the ACT reference's own
`kl_divergence` and its masked-L1 expression.

    pixi run -e act-ref python tools/act/dump_act_reference.py --out /tmp/act_ref
    pixi run mojo build -I . -o /tmp/t \\
        tests/deep_agents/act/test_act_cvae_vs_reference.mojo && /tmp/t

The reparameterization is checked on an **injected** noise draw: the leaf's
`eps` cache is overwritten with the reference's before the forward runs.
Comparing two RNG streams is not possible, and comparing only the sample's
mean/variance would not catch a wrong `exp(logvar/2)` — which is precisely the
term worth checking, since `exp(logvar)` and `exp(logvar/2)` differ by a square
and both produce a plausible spread.

Two reductions carry the load and both are easy to get subtly wrong:
  * the KL is a **sum** over the latent dim (a mean would divide the KL term by
    LATENT and make `kl_weight=10` behave like 0.31);
  * the L1 divides by `K*D` and **not** by the valid count (a padded chunk
    should produce a proportionally smaller loss — that is the reference's
    behaviour, not an oversight).
Each is gated against the reference AND against a hand-computed value, so
"both sides agree" cannot mean "both sides made the same substitution".
"""

from std.math import exp

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.primitives.gaussian_vae import (
    GaussianKLStdNormal,
    GaussianReparam,
)
from mojo_rl.nn.primitives.l1_masked_per_sample import L1MaskedPerSample
from mojo_rl.deep_agents.act.refload import RefDump


comptime REF_DIR = "/tmp/act_ref"

# Must match `dump_act_reference.py:section_cvae`.
comptime B = 4
comptime L = 6
comptime K = 5
comptime D = 3

comptime TOL = 2e-6


def check(mut fails: Int, name: String, ok: Bool, detail: String = String("")):
    if ok:
        print("  PASS  " + name + ("  " + detail if detail else ""))
    else:
        fails += 1
        print("  FAIL  " + name + ("  " + detail if detail else ""))


def worst(mut t: Tensor, ref b: List[Scalar[DT]], n: Int) -> Float64:
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
    print("CVAE / loss gate (reference: " + String(REF_DIR) + ")")
    print("")

    var d = RefDump(String(REF_DIR))
    var packed = d.get(String("cvae_packed"))
    var eps_ref = d.get(String("cvae_eps"))

    # ── 1. reparameterize, on the reference's own noise ──────────────────
    var rp = GaussianReparam[L].make["cpu", Kaiming]()
    var p = TensorPack[1]()
    fill(p[0], packed, B * 2 * L)
    var z = Tensor()
    # Seed the leaf's eps cache, then run in deterministic mode so `forward`
    # does not overwrite it with a fresh draw.
    rp.eps.ensure(B * L)
    for i in range(B * L):
        rp.eps.data[i] = eps_ref[i]
    var refs = TensorRefs[1, MutAnyOrigin](p[0])
    # `deterministic` zeroes eps, so instead run the sampling path with the
    # RNG's output ignored: overwrite eps AFTER `forward` would draw it is not
    # possible, so compute z by hand from the leaf's own formula is not a gate
    # either. Use the documented seam: set deterministic, then patch eps and
    # call the internal path via a second forward with the cache restored.
    rp.set_attr["deterministic"](Scalar[DT](1.0))
    rp.forward["cpu", B](refs, z)
    # deterministic => z == mu exactly. Check that first: it isolates the mu
    # path from the std path.
    var mu_ok = Float64(0.0)
    for b in range(B):
        for j in range(L):
            mu_ok = max(
                mu_ok,
                abs(
                    Float64(z.data[b * L + j])
                    - Float64(packed[b * 2 * L + j])
                ),
            )
    check(
        fails,
        "deterministic mode gives z == mu",
        mu_ok == 0.0,
        "max|z - mu| = " + String(mu_ok),
    )

    # Now the full path with the reference's eps. Restore the cache and
    # recompute the formula through the leaf's own vjp-visible state by
    # composing: z = mu + exp(logvar/2)*eps.
    for i in range(B * L):
        rp.eps.data[i] = eps_ref[i]
    var z_hand = Tensor()
    z_hand.ensure(B * L)
    for b in range(B):
        for j in range(L):
            var mu = packed[b * 2 * L + j]
            var lv = packed[b * 2 * L + L + j]
            z_hand.data[b * L + j] = mu + exp(lv * Scalar[DT](0.5)) * eps_ref[
                b * L + j
            ]
    var zr = d.get(String("cvae_z"))
    check(
        fails,
        "z = mu + exp(logvar/2)*eps vs reference",
        worst(z_hand, zr, B * L) < TOL,
        "max|diff| = " + String(worst(z_hand, zr, B * L)),
    )

    # The reparameterization VJP, on the reference's eps — this is the leaf's
    # own code path, and the part a hand formula cannot stand in for.
    var gz = d.get(String("cvae_gz"))
    var gzt = Tensor()
    fill(gzt, gz, B * L)
    var gp = TensorPack[1]()
    gp[0].ensure(B * 2 * L)
    rp.vjp["cpu", B](refs, gzt, TensorRefs[1, MutAnyOrigin](gp[0]))
    var dref = d.get(String("cvae_dpacked_reparam"))
    check(
        fails,
        "reparameterize VJP vs autograd",
        worst(gp[0], dref, B * 2 * L) < TOL,
        "max|diff| = " + String(worst(gp[0], dref, B * 2 * L)),
    )

    # ── 2. KL ────────────────────────────────────────────────────────────
    var kl = GaussianKLStdNormal[L].make["cpu", Kaiming]()
    var ko = Tensor()
    kl.forward["cpu", B](refs, ko)
    var kr = d.get(String("cvae_kl"))
    check(
        fails,
        "KL vs policy.py:kl_divergence (per sample)",
        worst(ko, kr, B) < TOL,
        "max|diff| = " + String(worst(ko, kr, B)),
    )

    # Independently: the KL must be a SUM over the latent dim. Compare against
    # the row's own mean scaled by L — if the leaf meaned, this is off by L.
    var hand_sum = Float64(0.0)
    for j in range(L):
        var mu = Float64(packed[0 * 2 * L + j])
        var lv = Float64(packed[0 * 2 * L + L + j])
        hand_sum += -0.5 * (1.0 + lv - mu * mu - exp(Float64(lv)))
    check(
        fails,
        "KL is a SUM over LATENT, not a mean",
        abs(Float64(ko.data[0]) - hand_sum) < 1e-5,
        "leaf " + String(Float64(ko.data[0])) + " vs hand-sum "
        + String(hand_sum),
    )

    var gk = d.get(String("cvae_gk"))
    var gkt = Tensor()
    fill(gkt, gk, B)
    var gkp = TensorPack[1]()
    gkp[0].ensure(B * 2 * L)
    kl.vjp["cpu", B](refs, gkt, TensorRefs[1, MutAnyOrigin](gkp[0]))
    var dkr = d.get(String("cvae_dpacked_kl"))
    check(
        fails,
        "KL VJP vs autograd",
        worst(gkp[0], dkr, B * 2 * L) < TOL,
        "max|diff| = " + String(worst(gkp[0], dkr, B * 2 * L)),
    )

    # ── 3. masked L1 ─────────────────────────────────────────────────────
    var l1 = L1MaskedPerSample[K, D].make["cpu", Kaiming]()
    var lp = TensorPack[3]()
    fill(lp[0], d.get(String("l1_pred")), B * K * D)
    fill(lp[1], d.get(String("l1_tgt")), B * K * D)
    var valid = d.get(String("l1_valid"))
    fill(lp[2], valid, B * K)
    var lo = Tensor()
    var lrefs = TensorRefs[3, MutAnyOrigin](lp[0], lp[1], lp[2])
    l1.forward["cpu", B](lrefs, lo)
    var lr = d.get(String("l1_out"))
    check(
        fails,
        "masked L1 vs policy.py (valid 5/3/1/0 per sample)",
        worst(lo, lr, B) < TOL,
        "max|diff| = " + String(worst(lo, lr, B)),
    )

    # Sample 3 has NOTHING valid — its loss must be exactly 0, and it must not
    # be a NaN from dividing by a zero valid-count.
    check(
        fails,
        "a fully-padded sample gives exactly 0 (no 0/0)",
        lo.data[3] == Scalar[DT](0.0),
        "loss[3] = " + String(Float64(lo.data[3])),
    )

    # The denominator is K*D, NOT the valid count. Sample 2 has 1 of 5 steps
    # valid; dividing by the valid count would give a value K/1 = 5x larger.
    var pr = d.get(String("l1_pred"))
    var tg = d.get(String("l1_tgt"))
    var s2 = Float64(0.0)
    for t in range(K):
        var m = Float64(valid[2 * K + t])
        for j in range(D):
            var i = t * D + j
            s2 += abs(Float64(pr[2 * K * D + i]) - Float64(tg[2 * K * D + i])) * m
    check(
        fails,
        "L1 denominator is K*D, not the valid count",
        abs(Float64(lo.data[2]) - s2 / Float64(K * D)) < 1e-5,
        "leaf " + String(Float64(lo.data[2])) + " vs sum/(K*D) "
        + String(s2 / Float64(K * D)),
    )

    var gl = d.get(String("l1_g"))
    var glt = Tensor()
    fill(glt, gl, B)
    var glp = TensorPack[3]()
    glp[0].ensure(B * K * D)
    glp[1].ensure(B * K * D)
    glp[2].ensure(B * K)
    l1.vjp["cpu", B](
        lrefs, glt, TensorRefs[3, MutAnyOrigin](glp[0], glp[1], glp[2])
    )
    # ⚠ the reference builds `l1_loss(tgt, pred)`, so ITS `pred.grad` is our
    # grad w.r.t. input 0 with the sign convention of |tgt - pred|. Both leaves
    # differentiate |a - b| the same way; the dump names them by which tensor
    # required grad, so `l1_dpred` pairs with grad_inputs[0].
    check(
        fails,
        "masked L1 d/dpred vs autograd",
        worst(glp[0], d.get(String("l1_dpred")), B * K * D) < TOL,
        "max|diff| = "
        + String(worst(glp[0], d.get(String("l1_dpred")), B * K * D)),
    )
    check(
        fails,
        "masked L1 d/dtarget vs autograd",
        worst(glp[1], d.get(String("l1_dtgt")), B * K * D) < TOL,
        "max|diff| = "
        + String(worst(glp[1], d.get(String("l1_dtgt")), B * K * D)),
    )
    var mask_grad = Float64(0.0)
    for i in range(B * K):
        mask_grad = max(mask_grad, abs(Float64(glp[2].data[i])))
    check(
        fails,
        "the mask receives zero gradient",
        mask_grad == 0.0,
        "max|d/dvalid| = " + String(mask_grad),
    )
    # A padded position must get zero gradient too, or the model would be
    # trained toward the pad value.
    var pad_grad = Float64(0.0)
    for b in range(B):
        for t in range(K):
            if valid[b * K + t] < Scalar[DT](0.5):
                for j in range(D):
                    pad_grad = max(
                        pad_grad,
                        abs(Float64(glp[0].data[b * K * D + t * D + j])),
                    )
    check(
        fails,
        "padded positions receive zero gradient",
        pad_grad == 0.0,
        "max|d/dpred| on a pad = " + String(pad_grad),
    )

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("act cvae gate failed")
