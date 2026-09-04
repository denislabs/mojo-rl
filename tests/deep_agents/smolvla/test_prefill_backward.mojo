# +--------------------------------------------------------------------------+ #
# | The gradient all the way back to the prefix embeddings
# +--------------------------------------------------------------------------+ #
"""`SmolVLAPrefill.backward`, against central differences of the whole chain.

    pixi run mojo run -I . \\
        tests/deep_agents/smolvla/test_prefill_backward.mojo

This is what `train_state_proj = True` needs. The state token is one of the
prefix embeddings, so `state_proj`'s gradient is a slice of `grad_x` here, and
every VLM layer sits between it and the loss even though not one VLM weight is
trained.

The loss is the real chain:

    prefill(x) -> KV cache -> one denoising step -> L = sum(g * out)

and the gradient path is its exact reverse — `denoise.backward` emitting
dL/d(cache), `prefill.backward` consuming it. Differencing `x` measures the
whole thing at once, which is the point: the two drivers are separately gated
and the JOIN between them is not.

## Three things that only this gate can see

  1. **The two-source accumulation.** Layer i's post-RoPE K and its V each
     feed the cache AND that layer's own attention. Their gradients must be
     summed. Taking only the attention side gives `state_proj` a gradient
     that looks reasonable and is missing most of itself.
  2. **`MaskedAttention` is output-caching with one instance for every
     layer** — the same trap `SwiGLU` sprang in the denoise backward, and the
     prefill has BOTH. Each needs its cache refilled with the layer's own
     input before differentiating.
  3. **The cache slabs are indexed identically at both ends.**
     `_store_cache_grad` and `_load_cache_grad` both use `layer * LAYER_N`;
     if they disagreed, layer i's gradient would arrive at layer j and the
     result would be finite, plausible and wrong.

⚠ The starting gradient is ZERO. `run`'s output — the prefix's final norm — is
discarded by the training forward, so nothing flows in from the top and every
bit of gradient enters through the cache. Leg [2] checks the consequence: with
the cache gradient zeroed, `grad_x` must be exactly zero.

## MEASURED — three defects, all of them in the join

    defect                                  leg [3] ||err||/||fd||   outside
    A1  drop the cache term at KR
        (attention side only)                     2.07e-01           43/48
    A2  no MaskedAttention cache refill           6.51e-02           35/48
    A3  the load indexes the cache slab in
        REVERSE layer order                       1.19e+00           48/48

⚠ Leg [2] passes in all three. It checks only that nothing enters from the
top, which is a real property and not this one — a reminder that a leg which
passes during an ablation is not thereby useless, it is measuring something
else.

A1 is the two-source accumulation, and it is the defect most likely to be
written: the attention path is the one you are already holding when you reach
KR, and the cache term arrives from a different driver. It leaves
`state_proj` with 79% of a gradient.

A3 is the one that would have been hardest to find by reading. `_store_cache_grad`
and `_load_cache_grad` are forty lines and two structs apart, and nothing but
this gate connects their indexing.
"""

from std.math import abs, sqrt
from std.testing import assert_true, assert_equal

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.deep_agents.smolvla.text import SmolVLMTextLayers, SMOLLM_THETA
from mojo_rl.deep_agents.smolvla.expert import SmolVLAExpert
from mojo_rl.deep_agents.smolvla.kv_cache import SmolVLAKVCache
from mojo_rl.deep_agents.smolvla.fused import SmolVLAPrefill, SmolVLADenoise
from mojo_rl.deep_agents.smolvla.attn_mask import (
    att_2d_mask, att_2d_mask_square, smolvla_ar,
)

comptime P = 6
comptime S = 3
comptime B = 1
comptime L = 2
comptime W = 8
comptime FF_T = 12
comptime EW = 8
comptime EFF = 12
comptime HEADS = 2
comptime NKV = 1
comptime HD = 4
comptime KVW = NKV * HD
comptime XN_P = B * P * W
comptime XN_S = B * S * EW
comptime CACHE_N = L * B * P * KVW

comptime Tower = SmolVLMTextLayers[L, W, FF_T, KVW]
comptime Expert = SmolVLAExpert[L, EW, EFF, W, KVW, 2]
comptime Cache = SmolVLAKVCache[L, P, S, NKV, HD, B]
comptime Pre = SmolVLAPrefill[
    P, S, B, L, W, FF_T, HEADS, NKV, HD, SMOLLM_THETA, True
]
comptime Den = SmolVLADenoise[
    P, S, B, L, EW, EFF, W, HEADS, NKV, HD, SMOLLM_THETA, 2, KVW, True
]

comptime FD_H = 2.0e-2
comptime FD_H2 = 1.0e-2
comptime NORM_BAND = 3.0e-3


def _loss(
    mut pre: Pre, mut den: Den, mut tower: Tower, mut e: Expert,
    mut c: Cache, mut x: Tensor, mut xs: Tensor, mut po: Tensor,
    mut out: Tensor, ref g: List[Float64],
) raises -> Float64:
    """prefill(x) -> cache -> one denoising step -> sum(g * out)."""
    c.reset()
    pre.run["cpu"](tower, c, x, po, None)
    den.step["cpu"](e, c, xs, out, None)
    var acc = 0.0
    for i in range(XN_S):
        acc += Float64(out.data[i]) * g[i]
    return acc


def _fd(
    t: Int, mut pre: Pre, mut den: Den, mut tower: Tower, mut e: Expert,
    mut c: Cache, mut x: Tensor, mut xs: Tensor, mut po: Tensor,
    mut out: Tensor, ref g: List[Float64],
) raises -> Float64:
    var keep = x.data[t]
    x.data[t] = Scalar[DT](Float64(keep) + FD_H)
    var ap = Float64(x.data[t])
    var lp = _loss(pre, den, tower, e, c, x, xs, po, out, g)
    x.data[t] = Scalar[DT](Float64(keep) - FD_H)
    var am = Float64(x.data[t])
    var lm = _loss(pre, den, tower, e, c, x, xs, po, out, g)
    x.data[t] = Scalar[DT](Float64(keep) + FD_H2)
    var ap2 = Float64(x.data[t])
    var lp2 = _loss(pre, den, tower, e, c, x, xs, po, out, g)
    x.data[t] = Scalar[DT](Float64(keep) - FD_H2)
    var am2 = Float64(x.data[t])
    var lm2 = _loss(pre, den, tower, e, c, x, xs, po, out, g)
    x.data[t] = keep
    var d1 = (lp - lm) / (ap - am)
    var d2 = (lp2 - lm2) / (ap2 - am2)
    return (4.0 * d2 - d1) / 3.0


def main() raises:
    print("=" * 70)
    print("SmolVLAPrefill.backward vs central differences of prefill+denoise")
    print("=" * 70)
    print("  P", P, " S", S, " layers", L, " W", W, " kv", KVW)

    var ar_pre = smolvla_ar(3, 2, 1, 0)
    assert_equal(len(ar_pre), P, "prefix ar")
    var ar_full = smolvla_ar(3, 2, 1, S)
    var mask_pre = att_2d_mask_square(ar_pre)
    var mask_self = att_2d_mask(ar_full, P, P + S, 0, P + S)
    var mask_cross = att_2d_mask(ar_full, P, P + S, 0, P)

    var tower = Tower.make["cpu", Deterministic]()
    var e = Expert.make["cpu", Deterministic]()
    var c = Cache.make["cpu"]()
    var pre = Pre.make["cpu"](mask_pre, None)
    var den = Den.make["cpu"](mask_self, mask_cross, None)

    var x = Tensor.alloc(XN_P)
    for i in range(XN_P):
        x.data[i] = Scalar[DT](((i * 29) % 17) - 8) * 0.06
    var xs = Tensor.alloc(XN_S)
    for i in range(XN_S):
        xs.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.05
    var g = List[Float64]()
    for i in range(XN_S):
        g.append(Float64(((i * 23) % 7) - 3) * 0.3)

    var po = Tensor.alloc(XN_P)
    var out = Tensor.alloc(XN_S)

    # ── [1] the forward, then the two backwards ──────────────────────────
    var l0 = _loss(pre, den, tower, e, c, x, xs, po, out, g)
    var grad_out = Tensor.alloc(XN_S)
    for i in range(XN_S):
        grad_out.data[i] = Scalar[DT](g[i])
    var gx_s = Tensor.alloc(XN_S)
    var gck = Tensor.alloc(CACHE_N)
    var gcv = Tensor.alloc(CACHE_N)
    den.backward["cpu"](e, c, grad_out, gx_s, gck, gcv, None)
    var grad_x = Tensor.alloc(XN_P)
    pre.backward["cpu"](tower, gck, gcv, grad_x, None)
    print("  [1] L =", l0, " both backwards ran")

    var gx = List[Float64]()
    for t in range(XN_P):
        gx.append(Float64(grad_x.data[t]))

    # ── [2] with a ZERO cache gradient, grad_x must be exactly zero ──────
    # ⚠ The prefill's own output does not reach the loss, so the ONLY way
    # gradient enters this pass is sideways through the cache. If anything
    # leaked in from the top — a stale `GXO`, a `grad_out` that should not
    # exist — this is where it shows.
    var zk = Tensor.alloc(CACHE_N)
    var zv = Tensor.alloc(CACHE_N)
    var gz = Tensor.alloc(XN_P)
    pre.backward["cpu"](tower, zk, zv, gz, None)
    var leaked = 0
    for t in range(XN_P):
        if gz.data[t] != Scalar[DT](0):
            leaked += 1
    print("  [2] zero cache gradient -> grad_x nonzero in", leaked, "of",
          XN_P, "slots")
    assert_true(
        leaked == 0,
        "gradient appeared from nowhere: the prefill backward is not starting"
        " from zero",
    )
    # restore the real tape and gradient
    _ = _loss(pre, den, tower, e, c, x, xs, po, out, g)
    den.backward["cpu"](e, c, grad_out, gx_s, gck, gcv, None)
    pre.backward["cpu"](tower, gck, gcv, grad_x, None)

    # ── [3] every prefix-embedding component vs a central difference ─────
    var fd = List[Float64]()
    for t in range(XN_P):
        fd.append(_fd(t, pre, den, tower, e, c, x, xs, po, out, g))
    var mx = 0.0
    for t in range(XN_P):
        if abs(fd[t]) > mx:
            mx = abs(fd[t])
    var floor = mx * 1.0e-3
    if floor < 1.0e-6:
        floor = 1.0e-6
    var num = 0.0
    var den2 = 0.0
    var bad = 0
    var worst = 0.0
    for t in range(XN_P):
        var sc = abs(fd[t])
        if sc < floor:
            sc = floor
        var rel = abs(gx[t] - fd[t]) / sc
        if rel > worst:
            worst = rel
        if rel > 3.0e-2:
            bad += 1
        num += (gx[t] - fd[t]) * (gx[t] - fd[t])
        den2 += fd[t] * fd[t]
    var rel_norm = sqrt(num / den2) if den2 > 0.0 else 0.0
    print("  [3] grad_x: compared", XN_P, " ||err||/||fd||", rel_norm,
          " outside band", bad, " worst rel", worst, " |grad|max", mx)
    assert_equal(XN_P, len(fd), "every prefix component must be probed")
    assert_true(
        mx > 1.0e-4,
        "the finite differences are all ~0 — the loss does not depend on the"
        " prefix embeddings and this gate proves nothing",
    )
    assert_true(
        rel_norm < NORM_BAND,
        "grad_x disagrees with a central difference of prefill+denoise",
    )

    # ── [4] the STATE token's slice, which is what state_proj sees ───────
    # The state is the last prefix token (`smolvla_ar` puts image+language
    # first, then state). Its slice of grad_x is `state_proj`'s gradient, and
    # it must not be zero or `train_state_proj = True` would train nothing.
    var snz = 0
    var smax = 0.0
    for j in range(W):
        var v = abs(gx[(P - 1) * W + j])
        if v != 0.0:
            snz += 1
        if v > smax:
            smax = v
    print("  [4] the state token's gradient slice: nonzero", snz, "of", W,
          " largest", smax)
    assert_true(
        snz == W,
        "the state token has no gradient — state_proj would never train",
    )

    print()
    print("PASSED — " + String(XN_P) + " prefix components through both"
          " drivers")
