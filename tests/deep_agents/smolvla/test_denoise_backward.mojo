# +--------------------------------------------------------------------------+ #
# | SmolVLADenoise.backward — every gradient against a central difference
# +--------------------------------------------------------------------------+ #
"""The gate that decides whether V2 can train anything at all.

    pixi run mojo run -I . \\
        tests/deep_agents/smolvla/test_denoise_backward.mojo

A backward pass over sixteen alternating layers has roughly forty places to be
subtly wrong, and not one of them raises. Wrong slot, wrong transpose, a
gradient assigned where it should have been summed, an output-caching leaf
differentiated at another layer's cache — each produces a finite,
correctly-shaped gradient and a fine-tune that converges to somewhere else.
There is no symptom until the robot is worse than the base checkpoint.

So every gradient is compared against a central difference of the loss:

    L(theta) = sum_t g_t * out_t(theta)          g fixed, arbitrary
    dL/dtheta ~ [L(theta + h) - L(theta - h)] / 2h

taken through the SAME `step` the backward claims to invert. That makes this a
self-consistency gate rather than a parity gate — it cannot tell us the
forward matches `lerobot` (that is `test_parity_vs_hf.mojo`), only that the
backward differentiates the forward we have. That is exactly the property no
amount of parity testing gives.

⚠ **A shallow fixture, and deliberately not the checkpoint's.** 2 layers, not
16 — but 2 is the smallest number that contains BOTH kinds, one self and one
cross, and the two kinds differ in what feeds k/v (own stream vs the frozen
cache) and in where q is rotated from. A 1-layer fixture would test half the
driver. Everything else is small so that ~1,400 forward passes cost seconds.

⚠ **h = 1e-2, and the band is 2e-2 relative**, which is loose. It has to be:
the forward is fp32, so the difference of two losses loses precision as h
shrinks while the O(h^2) truncation grows as h rises, and around 1e-2 the two
meet at roughly 1e-3 relative. That is a real limit of differencing an fp32
function, not a tolerance chosen to make a run pass — the ablation table below
is what says the band is still tight enough to be worth having.

## The traps this file exists to catch, all of them already sprung

  * **`SwiGLU.vjp` ignores its `forward_input` and reads a leaf-owned cache.**
    One instance drives all sixteen layers, so at backward time that cache
    holds the LAST layer's values and every layer would be differentiated at
    layer 15's point. `backward` re-runs `glu.forward` on the layer's own CAT
    first. Found by reading the leaf, not by this gate — but this gate is what
    would have caught it.
  * **The forward wrote both layernorm outputs into one slot.** Split into
    `H` and `H2` when recording, or every q/k/v weight gradient in every layer
    is formed against the MLP norm's output.
  * **`Module.vjp` ASSIGNS `grad_inputs`.** `H` feeds q, k and v; `X` and `X2`
    each feed a residual and a norm. Every one of those is an explicit sum
    into a separate slab, because sharing a destination silently keeps the
    last writer.

## MEASURED — four defects introduced into `backward`, one at a time

    defect                              grad_x ||err||/||fd||   what caught it
    A1  no SwiGLU cache refill                 1.94e-01         leg [2], 24/24
    A2  the MLP vjps read H, not H2            2.79e-05         leg [3] ONLY
    A3  dH drops v's contribution              1.48e-01         leg [2], 24/24
    A4  no cache-scratch rebuild               2.79e-05         NOTHING

Three things that table says.

**A2 is why leg [3] exists.** Reading the wrong layernorm output leaves
`grad_x` BIT-IDENTICAL — 2.785314986565461e-05, the same digits as a clean
run — because the error is confined to the weight gradients of the MLP
projections, which `grad_x` never passes through. Only
`self.mlp.gate.weight`'s norm moved. A gate that checked the input gradient
alone, which is the cheap and obvious thing to check, would have shipped it.

**A1 is the trap that motivated the design.** Removing four lines that look
redundant — re-running a forward whose output is thrown away — corrupts every
gradient in the network. `SwiGLU` is output-caching and there is one instance
for all the layers.

**A4 changed nothing at all, and that is reported rather than quietly fixed.**
The rebuild of `[prefix; suffix]` before `RepeatKVHeads.vjp` is dead today:
that leaf ignores its `forward_input` entirely. It stays because passing the
last self layer's scratch as "this layer's forward input" is a false statement
in the source, and making it true costs two slab copies per self layer against
the GEMMs of a whole backward pass. If the leaf ever starts reading its input,
this call is already right — and no gate would have told us.
"""

from std.math import abs, sqrt
from std.testing import assert_true, assert_equal

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.deep_agents.smolvla.text import SMOLLM_THETA
from mojo_rl.deep_agents.smolvla.expert import SmolVLAExpert
from mojo_rl.deep_agents.smolvla.kv_cache import SmolVLAKVCache
from mojo_rl.deep_agents.smolvla.fused import SmolVLADenoise
from mojo_rl.deep_agents.smolvla.attn_mask import att_2d_mask, smolvla_ar

comptime P = 6
comptime S = 3
comptime B = 1
comptime L = 2           # ⚠ the smallest fixture holding BOTH layer kinds
comptime EW = 8
comptime EFF = 12
comptime W = 8
comptime HEADS = 2
comptime NKV = 1
comptime HD = 4
comptime KVW = NKV * HD
comptime XN = B * S * EW
comptime PKV = B * P * KVW

comptime Expert = SmolVLAExpert[L, EW, EFF, W, KVW, 2]
comptime Cache = SmolVLAKVCache[L, P, S, NKV, HD, B]
comptime Den = SmolVLADenoise[
    P, S, B, L, EW, EFF, W, HEADS, NKV, HD, SMOLLM_THETA, 2, KVW, True
]

# ⚠ TWO step sizes, because the two groups are limited by OPPOSITE things.
# Measured on this fixture, not guessed:
#
#   grad_x, as h shrinks          weights, as h shrinks
#   4e-2/2e-2   worst 9.5e-02     8e-2/4e-2   ||err||/||fd|| 2.7e-05, 8/680
#   2e-2/1e-2   worst 6.0e-03     4e-2/2e-2                  3.7e-05, 29/680
#   1e-2/5e-3   worst 4.3e-04     2e-2/1e-2                  7.1e-05, 59/680
#   5e-3/2.5e-3 worst 6.6e-04     1e-2/5e-3                       -- 82/680
#
# grad_x IMPROVES as h shrinks: it is truncation-limited, its components are
# O(1..10), and the difference stands well clear of the fp32 floor. The weight
# gradients get WORSE — a weight whose own contribution to L = 4.1 is ~1e-3
# cannot be differenced accurately in fp32 at all, and shrinking h only
# amplifies the cancellation. One h for both would mean choosing which group
# to measure badly.
#
# ⚠ That the weight error scales as 1/h is also the EVIDENCE that it is the
# reference and not the gradient. `cross.q.weight` — the smallest gradient
# group here, |grad|max 0.030 against 1.47 for `self.o.weight` — reads
# 2.6e-03, 1.1e-03, 5.8e-04 as h doubles: exactly proportional to 1/h. A wrong
# gradient does not care what h is.
comptime FD_HX = Scalar[DT](1.0e-2)
comptime FD_HX2 = Scalar[DT](5.0e-3)
comptime FD_HW = Scalar[DT](8.0e-2)
comptime FD_HW2 = Scalar[DT](4.0e-2)
comptime N_KINDS = 16
comptime BAND = 3.0e-3
comptime NORM_BAND = 2.0e-3
"""Relative band, against a scale floored at 1e-3 of the group's own largest
gradient. Flooring matters: a component that is 0.6 beside neighbours of 11
is not meaningfully "22% wrong" when it is off by 0.1 — the difference is at
the noise level of the vector it lives in, and a per-component ratio says
otherwise."""


def _pname(which: Int) -> String:
    if which == 0: return String("self.q.weight")
    if which == 1: return String("self.k.weight")
    if which == 2: return String("self.v.weight")
    if which == 3: return String("self.o.weight")
    if which == 4: return String("self.mlp.gate.weight")
    if which == 5: return String("self.mlp.down.weight")
    if which == 6: return String("self.input_ln.gamma")
    if which == 7: return String("self.post_ln.gamma")
    if which == 8: return String("self.q.bias")
    if which == 9: return String("cross.q.weight")
    if which == 10: return String("cross.k.weight  <- reads the KV cache")
    if which == 11: return String("cross.v.weight  <- reads the KV cache")
    if which == 12: return String("cross.o.weight")
    if which == 13: return String("cross.mlp.up.weight")
    if which == 14: return String("cross.input_ln.gamma")
    return String("expert.norm.gamma")


def _psize(which: Int) -> Int:
    if which == 0: return EW * W
    if which == 1: return EW * KVW
    if which == 2: return EW * KVW
    if which == 3: return W * EW
    if which == 4: return EW * EFF
    if which == 5: return EFF * EW
    if which == 6: return EW
    if which == 7: return EW
    if which == 8: return W
    if which == 9: return EW * W
    if which == 10: return KVW * KVW
    if which == 11: return KVW * KVW
    if which == 12: return W * EW
    if which == 13: return EW * EFF
    if which == 14: return EW
    return EW


def _pget(which: Int, t: Int, mut e: Expert) raises -> Scalar[DT]:
    if which == 0: return e.self_layers[0].q.weight.val.data[t]
    if which == 1: return e.self_layers[0].k.weight.val.data[t]
    if which == 2: return e.self_layers[0].v.weight.val.data[t]
    if which == 3: return e.self_layers[0].o.weight.val.data[t]
    if which == 4: return e.self_layers[0].mlp.gate.weight.val.data[t]
    if which == 5: return e.self_layers[0].mlp.down.weight.val.data[t]
    if which == 6: return e.self_layers[0].input_layernorm.gamma.val.data[t]
    if which == 7:
        return e.self_layers[0].post_attention_layernorm.gamma.val.data[t]
    if which == 8: return e.self_layers[0].q.bias.val.data[t]
    if which == 9: return e.cross_layers[0].q.weight.val.data[t]
    if which == 10: return e.cross_layers[0].k.weight.val.data[t]
    if which == 11: return e.cross_layers[0].v.weight.val.data[t]
    if which == 12: return e.cross_layers[0].o.weight.val.data[t]
    if which == 13: return e.cross_layers[0].mlp.up.weight.val.data[t]
    if which == 14: return e.cross_layers[0].input_layernorm.gamma.val.data[t]
    return e.norm.gamma.val.data[t]


def _pset(which: Int, t: Int, v: Scalar[DT], mut e: Expert) raises:
    if which == 0: e.self_layers[0].q.weight.val.data[t] = v
    elif which == 1: e.self_layers[0].k.weight.val.data[t] = v
    elif which == 2: e.self_layers[0].v.weight.val.data[t] = v
    elif which == 3: e.self_layers[0].o.weight.val.data[t] = v
    elif which == 4: e.self_layers[0].mlp.gate.weight.val.data[t] = v
    elif which == 5: e.self_layers[0].mlp.down.weight.val.data[t] = v
    elif which == 6: e.self_layers[0].input_layernorm.gamma.val.data[t] = v
    elif which == 7:
        e.self_layers[0].post_attention_layernorm.gamma.val.data[t] = v
    elif which == 8: e.self_layers[0].q.bias.val.data[t] = v
    elif which == 9: e.cross_layers[0].q.weight.val.data[t] = v
    elif which == 10: e.cross_layers[0].k.weight.val.data[t] = v
    elif which == 11: e.cross_layers[0].v.weight.val.data[t] = v
    elif which == 12: e.cross_layers[0].o.weight.val.data[t] = v
    elif which == 13: e.cross_layers[0].mlp.up.weight.val.data[t] = v
    elif which == 14: e.cross_layers[0].input_layernorm.gamma.val.data[t] = v
    else: e.norm.gamma.val.data[t] = v


def _pgrad(which: Int, t: Int, mut e: Expert) raises -> Scalar[DT]:
    if which == 0: return e.self_layers[0].q.weight.grd.data[t]
    if which == 1: return e.self_layers[0].k.weight.grd.data[t]
    if which == 2: return e.self_layers[0].v.weight.grd.data[t]
    if which == 3: return e.self_layers[0].o.weight.grd.data[t]
    if which == 4: return e.self_layers[0].mlp.gate.weight.grd.data[t]
    if which == 5: return e.self_layers[0].mlp.down.weight.grd.data[t]
    if which == 6: return e.self_layers[0].input_layernorm.gamma.grd.data[t]
    if which == 7:
        return e.self_layers[0].post_attention_layernorm.gamma.grd.data[t]
    if which == 8: return e.self_layers[0].q.bias.grd.data[t]
    if which == 9: return e.cross_layers[0].q.weight.grd.data[t]
    if which == 10: return e.cross_layers[0].k.weight.grd.data[t]
    if which == 11: return e.cross_layers[0].v.weight.grd.data[t]
    if which == 12: return e.cross_layers[0].o.weight.grd.data[t]
    if which == 13: return e.cross_layers[0].mlp.up.weight.grd.data[t]
    if which == 14: return e.cross_layers[0].input_layernorm.gamma.grd.data[t]
    return e.norm.gamma.grd.data[t]


def _loss(
    mut den: Den, mut e: Expert, mut c: Cache, mut x: Tensor,
    ref g: List[Float64], mut out: Tensor,
) raises -> Float64:
    den.step["cpu"](e, c, x, out, None)
    var acc = 0.0
    for i in range(XN):
        acc += Float64(out.data[i]) * g[i]
    return acc


def _richardson(d1: Float64, d2: Float64) -> Float64:
    """Central differences at h and h/2, with the O(h^2) term removed.

    D(h) = f' + c*h^2 + O(h^4), so (4*D(h/2) - D(h))/3 cancels c exactly. The
    header records the measurement that says this is worth doing: the raw
    error falls by 3.9-4.0x per halving, which is that c*h^2 and nothing else.
    """
    return (4.0 * d2 - d1) / 3.0


def _fd_weight(
    which: Int, t: Int, mut den: Den, mut e: Expert, mut c: Cache,
    mut x: Tensor, ref g: List[Float64], mut out: Tensor,
) raises -> Float64:
    var keep = _pget(which, t, e)
    _pset(which, t, keep + FD_HW, e)
    var lp = _loss(den, e, c, x, g, out)
    _pset(which, t, keep - FD_HW, e)
    var lm = _loss(den, e, c, x, g, out)
    _pset(which, t, keep + FD_HW2, e)
    var lp2 = _loss(den, e, c, x, g, out)
    _pset(which, t, keep - FD_HW2, e)
    var lm2 = _loss(den, e, c, x, g, out)
    _pset(which, t, keep, e)
    return _richardson(
        (lp - lm) / (2.0 * Float64(FD_HW)),
        (lp2 - lm2) / (2.0 * Float64(FD_HW2)),
    )


def _fd_input(
    t: Int, mut den: Den, mut e: Expert, mut c: Cache, mut x: Tensor,
    ref g: List[Float64], mut out: Tensor,
) raises -> Float64:
    var keep = x.data[t]
    x.data[t] = keep + FD_HX
    var lp = _loss(den, e, c, x, g, out)
    x.data[t] = keep - FD_HX
    var lm = _loss(den, e, c, x, g, out)
    x.data[t] = keep + FD_HX2
    var lp2 = _loss(den, e, c, x, g, out)
    x.data[t] = keep - FD_HX2
    var lm2 = _loss(den, e, c, x, g, out)
    x.data[t] = keep
    return _richardson(
        (lp - lm) / (2.0 * Float64(FD_HX)),
        (lp2 - lm2) / (2.0 * Float64(FD_HX2)),
    )


struct Cmp(Movable):
    """Compared / differing, with the worst offender kept."""
    var n: Int
    var bad: Int
    var worst: Float64
    var at: Int
    var floor: Float64
    var num: Float64
    var den: Float64

    def __init__(out self):
        self.n = 0
        self.bad = 0
        self.worst = 0.0
        self.at = -1
        self.floor = 1.0e-6
        self.num = 0.0
        self.den = 0.0

    def __init__(out self, *, deinit move: Self):
        self.n = move.n
        self.bad = move.bad
        self.worst = move.worst
        self.at = move.at
        self.floor = move.floor
        self.num = move.num
        self.den = move.den

    def set_group(mut self, ref fd: List[Float64]):
        """Floor the scale at 1e-3 of the group's own largest gradient."""
        var mx = 0.0
        for i in range(len(fd)):
            if abs(fd[i]) > mx:
                mx = abs(fd[i])
        self.floor = mx * 1.0e-3
        if self.floor < 1.0e-6:
            self.floor = 1.0e-6

    def add(mut self, got: Float64, want: Float64, idx: Int):
        self.n += 1
        var sc = abs(want)
        if sc < self.floor:
            sc = self.floor
        var rel = abs(got - want) / sc
        if rel > self.worst:
            self.worst = rel
            self.at = idx
        if rel > BAND:
            self.bad += 1
        self.num += (got - want) * (got - want)
        self.den += want * want

    def rel_norm(self) -> Float64:
        """||analytic - fd|| / ||fd|| over the group.

        ⚠ THE load-bearing statistic here, not the per-component worst. The
        reference is a difference of two fp32 losses, and for a weight whose
        own contribution to L is 1e-3 of L that difference is near the fp32
        floor — measured: shrinking h makes the per-component agreement WORSE
        for weights while it makes it BETTER for inputs. A norm-relative error
        is not fooled by a handful of components that are individually below
        the reference's own noise, and a structural defect moves it to O(1)
        anyway (see the ablation table).
        """
        if self.den <= 0.0:
            return 0.0
        return sqrt(self.num / self.den)


def main() raises:
    print("=" * 70)
    print("SmolVLADenoise.backward vs central differences of its own forward")
    print("=" * 70)
    print("  P", P, " S", S, " layers", L, "(1 self + 1 cross)  EW", EW,
          " W", W, " heads", HEADS, " kv", NKV)

    var ar_full = smolvla_ar(3, 2, 1, S)
    assert_equal(len(ar_full), P + S, "ar length")
    var mask_self = att_2d_mask(ar_full, P, P + S, 0, P + S)
    var mask_cross = att_2d_mask(ar_full, P, P + S, 0, P)

    var e = Expert.make["cpu", Deterministic]()
    var c = Cache.make["cpu"]()
    var den = Den.make["cpu"](mask_self, mask_cross, None)

    # A filled cache, written directly — no VLM needed to test the expert.
    var kp = Tensor.alloc(PKV)
    var vp = Tensor.alloc(PKV)
    for l in range(L):
        for i in range(PKV):
            kp.data[i] = Scalar[DT](((i * 31 + l * 7) % 13) - 6) * 0.11
            vp.data[i] = Scalar[DT](((i * 17 + l * 5) % 11) - 5) * 0.09
        c.write_prefix["cpu"](l, kp, vp)

    var x = Tensor.alloc(XN)
    for i in range(XN):
        x.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.07
    var g = List[Float64]()
    for i in range(XN):
        g.append(Float64(((i * 23) % 7) - 3) * 0.3)

    # ── [1] one forward, one backward ────────────────────────────────────
    var out = Tensor.alloc(XN)
    var l0 = _loss(den, e, c, x, g, out)
    var grad_out = Tensor.alloc(XN)
    for i in range(XN):
        grad_out.data[i] = Scalar[DT](g[i])
    var grad_x = Tensor.alloc(XN)
    den.backward["cpu"](e, c, grad_out, grad_x, None)
    print("  [1] L =", l0, " backward ran")

    # Snapshot every probed gradient BEFORE the finite differences re-run the
    # forward — `step` overwrites the tape, and a `backward` afterwards would
    # be reading a perturbed one.
    var snap = List[Float64]()
    for which in range(N_KINDS):
        for t in range(_psize(which)):
            snap.append(Float64(_pgrad(which, t, e)))
    var gx = List[Float64]()
    for t in range(XN):
        gx.append(Float64(grad_x.data[t]))

    # ── [2] dL/dx, every component ───────────────────────────────────────
    var fdx = List[Float64]()
    for t in range(XN):
        fdx.append(_fd_input(t, den, e, c, x, g, out))
    var cx = Cmp()
    cx.set_group(fdx)
    for t in range(XN):
        cx.add(gx[t], fdx[t], t)
    print("  [2] grad_x: compared", cx.n, " outside band", cx.bad,
          " worst rel", cx.worst, " ||err||/||fd||", cx.rel_norm())
    assert_equal(cx.n, XN, "every input component must be probed")
    assert_true(
        cx.rel_norm() < NORM_BAND,
        "grad_x disagrees with a central difference in norm",
    )
    assert_true(cx.bad == 0, "a grad_x component is outside the band")

    # ── [3] every weight of every distinct parameter kind ────────────────
    print("  [3] weight gradients, all", N_KINDS, "parameter kinds"
          " (per-component band is REPORTED, not asserted — see the header):")
    var total = Cmp()
    var k0 = 0
    var nonzero_kinds = 0
    for which in range(N_KINDS):
        var fdw = List[Float64]()
        for t in range(_psize(which)):
            fdw.append(_fd_weight(which, t, den, e, c, x, g, out))
        var ck = Cmp()
        ck.set_group(fdw)
        var mag = 0.0
        for t in range(_psize(which)):
            ck.add(snap[k0 + t], fdw[t], t)
            total.add(snap[k0 + t], fdw[t], k0 + t)
            if abs(snap[k0 + t]) > mag:
                mag = abs(snap[k0 + t])
        k0 += _psize(which)
        if mag > 0.0:
            nonzero_kinds += 1
        print("      " + _pname(which) + ": " + String(ck.n)
              + " compared, ||err||/||fd|| " + String(ck.rel_norm())
              + ", outside band " + String(ck.bad)
              + ", |grad|max " + String(mag))
        assert_true(
            ck.rel_norm() < NORM_BAND,
            "gradient of " + _pname(which) + " disagrees with a central"
            " difference in norm",
        )
    # ⚠ Anti-vacuity. A backward that wrote nothing leaves every `.grd` at the
    # zero it was allocated with, and zero matches a central difference of a
    # parameter the loss does not depend on. It depends on all sixteen.
    print("  [4] parameter kinds with a nonzero gradient:", nonzero_kinds,
          "of", N_KINDS)
    assert_true(
        nonzero_kinds == N_KINDS,
        "a parameter kind came back with an all-zero gradient — the backward"
        " never reached it",
    )
    print("      TOTAL: compared", total.n, " ||err||/||fd||", total.rel_norm(),
          " outside band", total.bad, "worst rel", total.worst)
    assert_true(
        total.rel_norm() < NORM_BAND, "the weight gradients disagree in norm"
    )

    print()
    print("PASSED — " + String(total.n + cx.n) + " gradient components against"
          " central differences")
