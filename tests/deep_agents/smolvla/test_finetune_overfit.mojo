# +--------------------------------------------------------------------------+ #
# | Can it learn? A loss that falls, on a batch it is allowed to memorise
# +--------------------------------------------------------------------------+ #
"""The first gate in V2 that asks for a RESULT rather than a derivative.

    pixi run mojo run -I . \\
        tests/deep_agents/smolvla/test_finetune_overfit.mojo

Everything before this checked that gradients are correct. Correct gradients
are necessary and not sufficient: an optimizer wired to four of five
components trains four of them, a missing `zero_grad` turns every step into a
running sum, and a missing version bump leaves the forward reading pre-update
weights. All three produce exact gradients and a model that does not improve,
or improves strangely, with nothing anywhere reporting a problem.

So this overfits ONE fixed batch and demands the loss collapse.

⚠ **Overfitting is the point, not a flaw.** With the noise, the action chunk
and the timestep all frozen, the target is a single constant vector and a
network with this many parameters must be able to memorise it. A loss that
does NOT fall under those conditions is broken plumbing; a loss that falls
says the gradient reaches the weights and the update has the right sign and a
usable scale. It says nothing about generalisation, which is what real data
is for.

## The three legs, and why the last two are the load-bearing ones

  [1] the loss falls. Necessary, and the weakest of the three: a loop that
      trained only `action_out` would also make this batch's loss fall, since
      `action_out` alone can fit a constant.
  [2] EVERY parameter group moved. This is what catches the four-of-five bug.
      A group whose weights are bit-identical after the run never trained, and
      leg [1] cannot see it.
  [3] a control at lr = 0, which must NOT fall. Without it, leg [1] would pass
      on a loss that drifts for any reason at all — and the loss here is
      recomputed each step from a forward pass, so "it changed" is not
      evidence that the optimizer did anything.

## MEASURED — three training-loop defects

    defect                              leg [1] ratio      caught by
    (clean)                             8.573e-04          —
    A1  action_in dropped from the
        UPDATE walk                     8.581e-04          leg [2] only
    A2  action_out dropped from the
        ZERO list                       1.412e-02          leg [1], leg [5]
    A3  the ParamVersionBump removed    8.573e-04          NOTHING

**A1 is why leg [2] exists.** The loss falls to within 0.1% of the clean run
while `action_in` never trains at all — 0 of 48 weights moved. A network this
size memorises one constant target without needing its input projection to
move, so the loss curve is not merely a weak signal here, it is a blind one.

**A2 is why leg [5] exists**, even though the tightened leg [1] now fires
first. At the original 0.05 band it reached 1.4e-02 and passed, with every
group moving, on a run whose gradients were the running sum of every step so
far — Adam's normalisation hides the magnitude and only the direction is
wrong. Leg [5] checks the zero directly instead of inferring it from a loss,
so it needs no band at all.

**A3 changed nothing, byte for byte, and is reported rather than removed.**
Nothing here caches a derived form of a weight: fp32, no AMP, no split-K
padding. The bump is inert in this fixture and is not inert in the
configuration this will eventually run in, where a stale cached cast means the
forward reads pre-update weights forever — a defect this repo has shipped
before. It stays, with the measurement written down, because "no gate covers
it" and "it does nothing" are different claims.
"""

from std.math import abs
from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.deep_agents.smolvla.text import SMOLLM_THETA
from mojo_rl.deep_agents.smolvla.expert import SmolVLAExpert
from mojo_rl.deep_agents.smolvla.kv_cache import SmolVLAKVCache
from mojo_rl.deep_agents.smolvla.fused import SmolVLADenoise
from mojo_rl.deep_agents.smolvla.train_step import SmolVLATrainStep
from mojo_rl.deep_agents.smolvla.finetune import (
    zero_trainable_grads, adam_step_trainables,
)
from mojo_rl.deep_agents.smolvla.flow_loss import build_xt_ut
from mojo_rl.deep_agents.smolvla.attn_mask import att_2d_mask, smolvla_ar

comptime P = 6
comptime CHUNK = 3
comptime B = 1
comptime L = 2
comptime EW = 8
comptime EFF = 12
comptime W = 8
comptime HEADS = 2
comptime NKV = 1
comptime HD = 4
comptime KVW = NKV * HD
comptime ADIM = 6
comptime ADIM_REAL = 3
comptime XN = B * CHUNK * ADIM
comptime PKV = B * P * KVW
comptime STEPS = 120
comptime LR = Scalar[DT](3.0e-3)

comptime Expert = SmolVLAExpert[L, EW, EFF, W, KVW, 2]
comptime Cache = SmolVLAKVCache[L, P, CHUNK, NKV, HD, B]
comptime Den = SmolVLADenoise[
    P, CHUNK, B, L, EW, EFF, W, HEADS, NKV, HD, SMOLLM_THETA, 2, KVW, True
]
comptime Step = SmolVLATrainStep[
    CHUNK, ADIM_REAL, ADIM, EW, B, L, EFF, W, HEADS, NKV, HD, SMOLLM_THETA,
    KVW,
]
comptime AIn = Linear[ADIM, EW]
comptime TIn = Linear[2 * EW, EW]
comptime TOut = Linear[EW, EW]
comptime AOut = Linear[EW, ADIM]

comptime N_GROUPS = 9


def _gname(g: Int) -> String:
    if g == 0: return String("expert.self[0].q      ")
    if g == 1: return String("expert.self[0].mlp.gate")
    if g == 2: return String("expert.self[0].in_ln   ")
    if g == 3: return String("expert.cross[0].k      ")
    if g == 4: return String("expert.norm            ")
    if g == 5: return String("action_in              ")
    if g == 6: return String("time_mlp_in            ")
    if g == 7: return String("time_mlp_out           ")
    return String("action_out             ")


def _gsize(g: Int) -> Int:
    if g == 0: return EW * W
    if g == 1: return EW * EFF
    if g == 2: return EW
    if g == 3: return KVW * KVW
    if g == 4: return EW
    if g == 5: return ADIM * EW
    if g == 6: return 2 * EW * EW
    if g == 7: return EW * EW
    return EW * ADIM


def _gval(
    g: Int, t: Int, mut e: Expert, mut ai: AIn, mut ti: TIn, mut to: TOut,
    mut ao: AOut,
) raises -> Scalar[DT]:
    if g == 0: return e.self_layers[0].q.weight.val.data[t]
    if g == 1: return e.self_layers[0].mlp.gate.weight.val.data[t]
    if g == 2: return e.self_layers[0].input_layernorm.gamma.val.data[t]
    if g == 3: return e.cross_layers[0].k.weight.val.data[t]
    if g == 4: return e.norm.gamma.val.data[t]
    if g == 5: return ai.weight.val.data[t]
    if g == 6: return ti.weight.val.data[t]
    if g == 7: return to.weight.val.data[t]
    return ao.weight.val.data[t]


def _ggrad(
    g: Int, t: Int, mut e: Expert, mut ai: AIn, mut ti: TIn, mut to: TOut,
    mut ao: AOut,
) raises -> Scalar[DT]:
    if g == 0: return e.self_layers[0].q.weight.grd.data[t]
    if g == 1: return e.self_layers[0].mlp.gate.weight.grd.data[t]
    if g == 2: return e.self_layers[0].input_layernorm.gamma.grd.data[t]
    if g == 3: return e.cross_layers[0].k.weight.grd.data[t]
    if g == 4: return e.norm.gamma.grd.data[t]
    if g == 5: return ai.weight.grd.data[t]
    if g == 6: return ti.weight.grd.data[t]
    if g == 7: return to.weight.grd.data[t]
    return ao.weight.grd.data[t]


def _snapshot(
    mut e: Expert, mut ai: AIn, mut ti: TIn, mut to: TOut, mut ao: AOut
) raises -> List[Scalar[DT]]:
    var out = List[Scalar[DT]]()
    for g in range(N_GROUPS):
        for t in range(_gsize(g)):
            out.append(_gval(g, t, e, ai, ti, to, ao))
    return out^


def main() raises:
    print("=" * 70)
    print("SmolVLA fine-tune: overfit one batch")
    print("=" * 70)
    print("  layers", L, " chunk", CHUNK, " EW", EW, " steps", STEPS,
          " lr", LR)

    var ar = smolvla_ar(3, 2, 1, CHUNK)
    var ms = att_2d_mask(ar, P, P + CHUNK, 0, P + CHUNK)
    var mc = att_2d_mask(ar, P, P + CHUNK, 0, P)

    var e = Expert.make["cpu", Deterministic]()
    var c = Cache.make["cpu"]()
    var den = Den.make["cpu"](ms, mc, None)
    var st = Step.make["cpu"](None)
    var ai = AIn.make["cpu", Deterministic]()
    var ti = TIn.make["cpu", Deterministic]()
    var to = TOut.make["cpu", Deterministic]()
    var ao = AOut.make["cpu", Deterministic]()

    var kp = Tensor.alloc(PKV)
    var vp = Tensor.alloc(PKV)
    for l in range(L):
        for i in range(PKV):
            kp.data[i] = Scalar[DT](((i * 31 + l * 7) % 13) - 6) * 0.11
            vp.data[i] = Scalar[DT](((i * 17 + l * 5) % 11) - 5) * 0.09
        c.write_prefix["cpu"](l, kp, vp)

    # ⚠ ONE frozen batch: fixed noise, fixed chunk, fixed t. Resampling any of
    # them would make the target move and the loss curve mean something else.
    var noise = Tensor.alloc(XN)
    var acts = Tensor.alloc(XN)
    for i in range(XN):
        noise.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.07
        acts.data[i] = Scalar[DT](0)
    for t in range(CHUNK):
        for d in range(ADIM_REAL):
            acts.data[t * ADIM + d] = Scalar[DT](((t * 5 + d * 3) % 7) - 3) * 0.2
    var times_t = Tensor.alloc(B)
    var tl = List[Float64]()
    for b in range(B):
        times_t.data[b] = Scalar[DT](0.37)
        tl.append(0.37)
    var x_t = Tensor.alloc(XN)
    var u_t = Tensor.alloc(XN)
    build_xt_ut["cpu", B, CHUNK * ADIM](noise, acts, times_t, x_t, u_t, None)
    # Every timestep inside its episode — this fixture has no episode edge.
    var valid = Tensor.alloc(B * CHUNK)
    for i in range(B * CHUNK):
        valid.data[i] = Scalar[DT](1.0)
    comptime N_VALID = B * CHUNK
    st.set_times["cpu"](tl, None)

    var before = _snapshot(e, ai, ti, to, ao)

    # ── [1] the loss must fall ───────────────────────────────────────────
    var opt = Adam(lr=LR)
    var first = 0.0
    var last = 0.0
    for s in range(STEPS):
        zero_trainable_grads["cpu", L, EW, EFF, W, KVW, ADIM](
            e, ai, ti, to, ao, None
        )
        var loss = st.run["cpu", P](e, c, den, ai, ti, to, ao, x_t, u_t, valid, N_VALID, None)
        if s == 0:
            first = loss
        last = loss
        if s % 30 == 0:
            print("      step", s, " loss", loss)
        adam_step_trainables["cpu", L, EW, EFF, W, KVW, ADIM](
            opt, e, ai, ti, to, ao, None
        )
    print("  [1] loss", first, "->", last, "  ratio",
          last / first)
    assert_true(first > 0.0, "the first loss is not positive")
    assert_true(
        last < first * 0.005,
        "the loss did not fall by 200x on a batch it is allowed to memorise —"
        " the gradients are correct but something in the update is not",
    )

    # ── [2] every parameter group moved ──────────────────────────────────
    # ⚠ THE leg that catches a component missing from either list in
    # `finetune.mojo`. Leg [1] cannot: `action_out` alone can fit a constant.
    var after = _snapshot(e, ai, ti, to, ao)
    assert_equal(len(after), len(before), "snapshot size changed")
    var k = 0
    var frozen = 0
    for g in range(N_GROUPS):
        var moved = 0
        var biggest = Scalar[DT](0)
        for t in range(_gsize(g)):
            var d = abs(after[k + t] - before[k + t])
            if d != Scalar[DT](0):
                moved += 1
            if d > biggest:
                biggest = d
        k += _gsize(g)
        if moved == 0:
            frozen += 1
        print("      " + _gname(g) + ": " + String(moved) + " of "
              + String(_gsize(g)) + " weights moved, largest "
              + String(biggest))
        assert_true(
            moved > 0,
            _gname(g) + " never changed — it is missing from the optimizer"
            " walk and silently did not train",
        )
    print("  [2] parameter groups that moved:", N_GROUPS - frozen, "of",
          N_GROUPS)

    # ── [4] `action_out`'s padded columns must NOT have moved ────────────
    # ⚠ Leg [2] reported 24 of 48, which looks like a partial failure and is
    # the opposite: `action_out` is [EW=8 -> ADIM=6] and the loss covers only
    # the first ADIM_REAL=3 output columns, so 8 x 3 = 24 weights can move and
    # the other 24 cannot. This is `flow_loss`'s padded-column property
    # arriving all the way at the weights, which is the only place it can be
    # observed as a CONSEQUENCE rather than as its own definition.
    var pad_moved = 0
    var real_still = 0
    var base = 0
    for g in range(N_GROUPS - 1):
        base += _gsize(g)
    for i in range(EW):
        for j in range(ADIM):
            var d = abs(after[base + i * ADIM + j] - before[base + i * ADIM + j])
            if j >= ADIM_REAL and d != Scalar[DT](0):
                pad_moved += 1
            if j < ADIM_REAL and d == Scalar[DT](0):
                real_still += 1
    print("  [4] action_out padded columns that moved:", pad_moved,
          " (of", EW * (ADIM - ADIM_REAL), ") | real columns that did NOT:",
          real_still, " (of", EW * ADIM_REAL, ")")
    assert_true(
        pad_moved == 0,
        "a padded action column's weights trained — the loss is reaching"
        " dimensions the robot does not have",
    )
    assert_true(
        real_still == 0,
        "a real action column's weights never moved — leg [4] is vacuous",
    )

    # ── [3] the control: lr = 0 must NOT fall ────────────────────────────
    var e2 = Expert.make["cpu", Deterministic]()
    var c2 = Cache.make["cpu"]()
    var den2 = Den.make["cpu"](ms, mc, None)
    var st2 = Step.make["cpu"](None)
    var ai2 = AIn.make["cpu", Deterministic]()
    var ti2 = TIn.make["cpu", Deterministic]()
    var to2 = TOut.make["cpu", Deterministic]()
    var ao2 = AOut.make["cpu", Deterministic]()
    for l in range(L):
        for i in range(PKV):
            kp.data[i] = Scalar[DT](((i * 31 + l * 7) % 13) - 6) * 0.11
            vp.data[i] = Scalar[DT](((i * 17 + l * 5) % 11) - 5) * 0.09
        c2.write_prefix["cpu"](l, kp, vp)
    st2.set_times["cpu"](tl, None)
    var opt0 = Adam(lr=Scalar[DT](0.0))
    var f0 = 0.0
    var l0 = 0.0
    for s in range(20):
        zero_trainable_grads["cpu", L, EW, EFF, W, KVW, ADIM](
            e2, ai2, ti2, to2, ao2, None
        )
        var loss = st2.run["cpu", P](
            e2, c2, den2, ai2, ti2, to2, ao2, x_t, u_t, valid, N_VALID,
            None
        )
        if s == 0:
            f0 = loss
        l0 = loss
        adam_step_trainables["cpu", L, EW, EFF, W, KVW, ADIM](
            opt0, e2, ai2, ti2, to2, ao2, None
        )
    print("  [3] control at lr=0:", f0, "->", l0, " (must not move)")
    assert_true(
        abs(l0 - f0) < 1.0e-9,
        "the loss moved at lr = 0, so leg [1] is not measuring the optimizer",
    )
    assert_true(
        abs(f0 - first) < 1.0e-9,
        "the control started from a different loss than the run — the two"
        " fixtures are not the same network",
    )

    # ── [5] `zero_trainable_grads` really zeroes ALL of them ─────────────
    # ⚠ A component missing from the ZERO list is not caught by any leg above.
    # MEASURED: dropping `action_out` from it still lets the loss fall to
    # 0.0133 (vs 0.00080 clean) and still moves every group, so legs [1] and
    # [2] both pass on a run whose gradients are the running sum of every step
    # so far. Adam's normalisation hides the magnitude; only the direction is
    # wrong, and only slowly.
    #
    # So this checks the operation directly instead of its consequences: run
    # one backward so every gradient is nonzero, zero them, and read them
    # back. No threshold, no band.
    var e3 = Expert.make["cpu", Deterministic]()
    var c3 = Cache.make["cpu"]()
    var den3 = Den.make["cpu"](ms, mc, None)
    var st3 = Step.make["cpu"](None)
    var ai3 = AIn.make["cpu", Deterministic]()
    var ti3 = TIn.make["cpu", Deterministic]()
    var to3 = TOut.make["cpu", Deterministic]()
    var ao3 = AOut.make["cpu", Deterministic]()
    for l in range(L):
        for i in range(PKV):
            kp.data[i] = Scalar[DT](((i * 31 + l * 7) % 13) - 6) * 0.11
            vp.data[i] = Scalar[DT](((i * 17 + l * 5) % 11) - 5) * 0.09
        c3.write_prefix["cpu"](l, kp, vp)
    st3.set_times["cpu"](tl, None)
    _ = st3.run["cpu", P](e3, c3, den3, ai3, ti3, to3, ao3, x_t, u_t, valid, N_VALID, None)

    var live = 0
    for g in range(N_GROUPS):
        for t in range(_gsize(g)):
            if _ggrad(g, t, e3, ai3, ti3, to3, ao3) != Scalar[DT](0):
                live += 1
    zero_trainable_grads["cpu", L, EW, EFF, W, KVW, ADIM](
        e3, ai3, ti3, to3, ao3, None
    )
    var left = 0
    var probed = 0
    for g in range(N_GROUPS):
        for t in range(_gsize(g)):
            probed += 1
            if _ggrad(g, t, e3, ai3, ti3, to3, ao3) != Scalar[DT](0):
                left += 1
    print("  [5] zero_trainable_grads: probed", probed, " nonzero before",
          live, " after", left)
    assert_true(
        live > 0,
        "no gradient was nonzero before zeroing — leg [5] proves nothing",
    )
    assert_true(
        left == 0,
        "a trainable gradient survived zero_trainable_grads: that component"
        " is missing from the zero list and its gradient is a running sum"
        " over the whole run",
    )

    # ── [6] the same loop on GPU ─────────────────────────────────────────
    # ⚠ Everything above is CPU. `Adam`'s per-parameter GPU update kernels and
    # `zero_grad["gpu"]` are therefore untouched by legs [1]-[5], and they are
    # what a real fine-tune runs. This is the same gap an ablation found in
    # `test_train_step.mojo`, where a defect planted in a GPU kernel changed
    # nothing because no leg ran on a GPU.
    #
    # Ten steps, not 120: this asks whether the GPU optimizer path takes the
    # same steps, not whether it can also memorise a batch.
    var d = DeviceContext()
    var eg = Expert.make["gpu", Deterministic](Optional(d))
    var cg = Cache.make["gpu"](Optional(d))
    var deng = Den.make["gpu"](ms, mc, Optional(d))
    var stg = Step.make["gpu"](Optional(d))
    var aig = AIn.make["gpu", Deterministic](Optional(d))
    var tig = TIn.make["gpu", Deterministic](Optional(d))
    var tog = TOut.make["gpu", Deterministic](Optional(d))
    var aog = AOut.make["gpu", Deterministic](Optional(d))
    for l in range(L):
        for i in range(PKV):
            kp.data[i] = Scalar[DT](((i * 31 + l * 7) % 13) - 6) * 0.11
            vp.data[i] = Scalar[DT](((i * 17 + l * 5) % 11) - 5) * 0.09
        kp.upload(d)
        vp.upload(d)
        cg.write_prefix["gpu"](l, kp, vp, Optional(d))
    var xg = Tensor.alloc(XN)
    var ug = Tensor.alloc(XN)
    for i in range(XN):
        xg.data[i] = x_t.data[i]
        ug.data[i] = u_t.data[i]
    xg.upload(d)
    ug.upload(d)
    var validg = Tensor.alloc(B * CHUNK)
    for i in range(B * CHUNK):
        validg.data[i] = Scalar[DT](1.0)
    validg.upload(d)
    stg.set_times["gpu"](tl, Optional(d))

    # the CPU reference for the SAME ten steps, from a fresh network
    var e4 = Expert.make["cpu", Deterministic]()
    var c4 = Cache.make["cpu"]()
    var den4 = Den.make["cpu"](ms, mc, None)
    var st4 = Step.make["cpu"](None)
    var ai4 = AIn.make["cpu", Deterministic]()
    var ti4 = TIn.make["cpu", Deterministic]()
    var to4 = TOut.make["cpu", Deterministic]()
    var ao4 = AOut.make["cpu", Deterministic]()
    for l in range(L):
        for i in range(PKV):
            kp.data[i] = Scalar[DT](((i * 31 + l * 7) % 13) - 6) * 0.11
            vp.data[i] = Scalar[DT](((i * 17 + l * 5) % 11) - 5) * 0.09
        c4.write_prefix["cpu"](l, kp, vp)
    st4.set_times["cpu"](tl, None)

    var og = Adam(lr=LR)
    var oc = Adam(lr=LR)
    var worst_rel = 0.0
    var lg = 0.0
    var lc = 0.0
    for _ in range(10):
        zero_trainable_grads["gpu", L, EW, EFF, W, KVW, ADIM](
            eg, aig, tig, tog, aog, Optional(d)
        )
        lg = stg.run["gpu", P](
            eg, cg, deng, aig, tig, tog, aog, xg, ug, validg, N_VALID,
            Optional(d)
        )
        adam_step_trainables["gpu", L, EW, EFF, W, KVW, ADIM](
            og, eg, aig, tig, tog, aog, Optional(d)
        )
        zero_trainable_grads["cpu", L, EW, EFF, W, KVW, ADIM](
            e4, ai4, ti4, to4, ao4, None
        )
        lc = st4.run["cpu", P](e4, c4, den4, ai4, ti4, to4, ao4, x_t, u_t,
                               valid, N_VALID, None)
        adam_step_trainables["cpu", L, EW, EFF, W, KVW, ADIM](
            oc, e4, ai4, ti4, to4, ao4, None
        )
        var rel = abs(lg - lc) / lc
        if rel > worst_rel:
            worst_rel = rel
    print("  [6] 10 steps GPU vs CPU: final", lg, "vs", lc,
          " worst per-step rel", worst_rel)
    assert_true(
        lc < f0 * 0.9,
        "the CPU control in leg [6] did not learn, so the comparison is"
        " between two networks that both did nothing",
    )
    assert_true(
        worst_rel < 1.0e-3,
        "the GPU training loop diverges from the CPU one",
    )

    print()
    print("PASSED — loss " + String(first) + " -> " + String(last)
          + ", all " + String(N_GROUPS) + " groups moved, control flat")
