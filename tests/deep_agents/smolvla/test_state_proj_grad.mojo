# +--------------------------------------------------------------------------+ #
# | state_proj's gradient — the one trainable weight upstream of the frozen VLM
# +--------------------------------------------------------------------------+ #
"""`train_state_proj = True`, end to end and against central differences.

    pixi run mojo run -I . \\
        tests/deep_agents/smolvla/test_state_proj_grad.mojo

`state_proj` is the reason V2's last stage exists. Every other trainable
parameter sits BELOW the KV cache, where a gradient reaches it without the VLM
being involved at all. `state_proj` sits above: its output is one of the
prefix embeddings, so its gradient arrives only after

    loss -> action_out -> the expert's layers -> dL/d(cached K/V)
         -> all sixteen VLM layers -> the prefix embeddings -> state_proj

and not one VLM weight is trained anywhere along it. That is why
`train_expert_only = True` sets `requires_grad = False` on the tower but does
NOT wrap it in `no_grad`, which is the distinction the whole stage turns on.

The chain here is the real one — `state_proj`, the prefill, one denoising
step — with the image and language halves of the prefix baked as constants,
since they do not depend on `state_proj` and running a SigLIP tower to
difference a 32x8 matrix would be four orders of magnitude of wasted work.

⚠ **The state token is written UNSCALED.** Image and language segments carry a
sqrt(W) factor and the state does not, so its slice of `grad_x` is
dL/d(state_proj's output) with nothing to undo. A gate that assumed the scale
applied uniformly would be wrong by sqrt(960) = 31, which is large enough to
notice and small enough to be mistaken for a learning-rate problem.

## MEASURED — two wiring defects in the last hop

    defect                                   leg [2] ||err||/||fd||   outside
    A1  the state taken as the FIRST prefix
        token instead of the last                  8.79e-01           40/256
    A2  the state slice scaled by sqrt(W),
        as image and language are                  1.83e+00           40/256

Both are one-line confusions about a layout stated in one place
(`SmolVLAPrefixEmbed.run`) and consumed in another. A2 is the sharper of the
two to think about: at the real widths the factor is sqrt(960) = 31, which is
large enough to notice as "training is unstable" and not obviously a wiring
bug — it looks exactly like a learning rate that wants lowering, and lowering
it would make the symptom go away while leaving `state_proj` learning 31 times
faster than everything else.
"""

from std.math import abs, sqrt
from std.testing import assert_true, assert_equal

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.deep_agents.smolvla.text import SmolVLMTextLayers, SMOLLM_THETA
from mojo_rl.deep_agents.smolvla.expert import SmolVLAExpert
from mojo_rl.deep_agents.smolvla.kv_cache import SmolVLAKVCache
from mojo_rl.deep_agents.smolvla.fused import SmolVLAPrefill, SmolVLADenoise
from mojo_rl.deep_agents.smolvla.finetune import state_proj_backward
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
comptime SDIM = 32
comptime XN_P = B * P * W
comptime XN_S = B * S * EW
comptime CACHE_N = L * B * P * KVW
comptime SP_N = SDIM * W

comptime Tower = SmolVLMTextLayers[L, W, FF_T, KVW]
comptime Expert = SmolVLAExpert[L, EW, EFF, W, KVW, 2]
comptime Cache = SmolVLAKVCache[L, P, S, NKV, HD, B]
comptime Pre = SmolVLAPrefill[
    P, S, B, L, W, FF_T, HEADS, NKV, HD, SMOLLM_THETA, True
]
comptime Den = SmolVLADenoise[
    P, S, B, L, EW, EFF, W, HEADS, NKV, HD, SMOLLM_THETA, 2, KVW, True
]
comptime SProj = Linear[SDIM, W]

comptime FD_H = 4.0e-2
comptime FD_H2 = 2.0e-2
comptime NORM_BAND = 3.0e-3


def _loss(
    mut sp: SProj, mut pre: Pre, mut den: Den, mut tower: Tower,
    mut e: Expert, mut c: Cache, mut st: Tensor, ref base: List[Scalar[DT]],
    mut tok: Tensor, mut x: Tensor, mut xs: Tensor, mut po: Tensor,
    mut out: Tensor, ref g: List[Float64],
) raises -> Float64:
    """state_proj -> the last prefix token -> prefill -> one denoise step."""
    sp.forward["cpu", B](TensorRefs[1](st), tok, None)
    x.ensure(XN_P)
    for i in range(XN_P):
        x.data[i] = base[i]
    # ⚠ LAST token, UNSCALED. Both facts are `SmolVLAPrefixEmbed.run`'s.
    for b in range(B):
        for j in range(W):
            x.data[b * P * W + (P - 1) * W + j] = tok.data[b * W + j]
    c.reset()
    pre.run["cpu"](tower, c, x, po, None)
    den.step["cpu"](e, c, xs, out, None)
    var acc = 0.0
    for i in range(XN_S):
        acc += Float64(out.data[i]) * g[i]
    return acc


def main() raises:
    print("=" * 70)
    print("state_proj's gradient, through the frozen VLM")
    print("=" * 70)
    print("  P", P, " layers", L, " state_proj", SDIM, "->", W,
          " (", SP_N, "weights )")

    var ar_pre = smolvla_ar(3, 2, 1, 0)
    var ar_full = smolvla_ar(3, 2, 1, S)
    var mask_pre = att_2d_mask_square(ar_pre)
    var mask_self = att_2d_mask(ar_full, P, P + S, 0, P + S)
    var mask_cross = att_2d_mask(ar_full, P, P + S, 0, P)

    var tower = Tower.make["cpu", Deterministic]()
    var e = Expert.make["cpu", Deterministic]()
    var c = Cache.make["cpu"]()
    var pre = Pre.make["cpu"](mask_pre, None)
    var den = Den.make["cpu"](mask_self, mask_cross, None)
    var sp = SProj.make["cpu", Deterministic]()

    # image + language: constants w.r.t. state_proj
    var base = List[Scalar[DT]](unsafe_uninit_length=XN_P)
    for i in range(XN_P):
        base[i] = Scalar[DT](((i * 29) % 17) - 8) * 0.06
    var st = Tensor.alloc(B * SDIM)
    for j in range(SDIM):
        # the robot's 6 real dims, then the zero padding
        st.data[j] = Scalar[DT](Float32(j) * 0.3 - 0.9) if j < 6 else Scalar[
            DT
        ](0)
    var tok = Tensor.alloc(B * W)
    var x = Tensor.alloc(XN_P)
    var xs = Tensor.alloc(XN_S)
    for i in range(XN_S):
        xs.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.05
    var g = List[Float64]()
    for i in range(XN_S):
        g.append(Float64(((i * 23) % 7) - 3) * 0.3)
    var po = Tensor.alloc(XN_P)
    var out = Tensor.alloc(XN_S)

    # ── [1] forward, then the three backwards ────────────────────────────
    var l0 = _loss(sp, pre, den, tower, e, c, st, base, tok, x, xs, po, out, g)
    var grad_out = Tensor.alloc(XN_S)
    for i in range(XN_S):
        grad_out.data[i] = Scalar[DT](g[i])
    var gx_s = Tensor.alloc(XN_S)
    var gck = Tensor.alloc(CACHE_N)
    var gcv = Tensor.alloc(CACHE_N)
    den.backward["cpu"](e, c, grad_out, gx_s, gck, gcv, None)
    var grad_x = Tensor.alloc(XN_P)
    pre.backward["cpu"](tower, gck, gcv, grad_x, None)
    var g_tok = Tensor.alloc(B * W)
    var g_state = Tensor.alloc(B * SDIM)
    state_proj_backward["cpu", B, P, W, SDIM](
        sp, st, grad_x, g_tok, g_state, None
    )
    print("  [1] L =", l0, " expert -> cache -> VLM -> state_proj")

    var snap = List[Float64]()
    for t in range(SP_N):
        snap.append(Float64(sp.weight.grd.data[t]))

    # ── [2] every state_proj weight vs a central difference ──────────────
    var fd = List[Float64]()
    for t in range(SP_N):
        var keep = sp.weight.val.data[t]
        sp.weight.val.data[t] = Scalar[DT](Float64(keep) + FD_H)
        var ap = Float64(sp.weight.val.data[t])
        var lp = _loss(sp, pre, den, tower, e, c, st, base, tok, x, xs, po,
                       out, g)
        sp.weight.val.data[t] = Scalar[DT](Float64(keep) - FD_H)
        var am = Float64(sp.weight.val.data[t])
        var lm = _loss(sp, pre, den, tower, e, c, st, base, tok, x, xs, po,
                       out, g)
        sp.weight.val.data[t] = Scalar[DT](Float64(keep) + FD_H2)
        var ap2 = Float64(sp.weight.val.data[t])
        var lp2 = _loss(sp, pre, den, tower, e, c, st, base, tok, x, xs, po,
                        out, g)
        sp.weight.val.data[t] = Scalar[DT](Float64(keep) - FD_H2)
        var am2 = Float64(sp.weight.val.data[t])
        var lm2 = _loss(sp, pre, den, tower, e, c, st, base, tok, x, xs, po,
                        out, g)
        sp.weight.val.data[t] = keep
        var d1 = (lp - lm) / (ap - am)
        var d2 = (lp2 - lm2) / (ap2 - am2)
        fd.append((4.0 * d2 - d1) / 3.0)

    var mx = 0.0
    for t in range(SP_N):
        if abs(fd[t]) > mx:
            mx = abs(fd[t])
    var floor = mx * 1.0e-3
    if floor < 1.0e-6:
        floor = 1.0e-6
    var num = 0.0
    var dsum = 0.0
    var bad = 0
    var worst = 0.0
    for t in range(SP_N):
        var sc = abs(fd[t])
        if sc < floor:
            sc = floor
        var rel = abs(snap[t] - fd[t]) / sc
        if rel > worst:
            worst = rel
        if rel > 3.0e-2:
            bad += 1
        num += (snap[t] - fd[t]) * (snap[t] - fd[t])
        dsum += fd[t] * fd[t]
    var rel_norm = sqrt(num / dsum) if dsum > 0.0 else 0.0
    print("  [2] state_proj.weight: compared", SP_N, " ||err||/||fd||",
          rel_norm, " outside band", bad, " worst rel", worst,
          " |grad|max", mx)
    assert_equal(len(fd), SP_N, "every state_proj weight must be probed")
    assert_true(
        mx > 1.0e-4,
        "the finite differences are all ~0 — the loss does not depend on"
        " state_proj and this gate proves nothing",
    )
    assert_true(
        rel_norm < NORM_BAND,
        "state_proj's gradient disagrees with a central difference of the"
        " whole chain",
    )

    # ── [3] the PADDED state rows must have no gradient ──────────────────
    # ⚠ `state_proj` is [32 -> W] and the robot has 6 joints, so rows 6..31 of
    # its weight multiply a zero input and can never move. A nonzero gradient
    # there means the state was not zero-padded — the same class of defect the
    # action columns have, on the other side of the network.
    var pad_nz = 0
    var pad_n = 0
    var real_z = 0
    var real_n = 0
    for i in range(SDIM):
        for j in range(W):
            var v = snap[i * W + j]
            if i >= 6:
                pad_n += 1
                if v != 0.0:
                    pad_nz += 1
            else:
                real_n += 1
                if v == 0.0:
                    real_z += 1
    print("  [3] padded state rows:", pad_n, " nonzero", pad_nz,
          " | real rows:", real_n, " zero", real_z)
    assert_true(
        pad_nz == 0,
        "a padded state row has a gradient — the state reaching state_proj is"
        " not zero-padded past the robot's 6 joints",
    )
    assert_true(
        real_z == 0, "a real state row has no gradient — leg [3] is vacuous"
    )

    print()
    print("PASSED — " + String(SP_N) + " state_proj weights through the"
          " expert, the cache and every VLM layer")
