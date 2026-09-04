# +--------------------------------------------------------------------------+ #
# | One training step, end to end: loss -> heads -> expert
# +--------------------------------------------------------------------------+ #
"""The gradient that reaches the action projections, against a central
difference of the loss it came from.

    pixi run -e apple mojo run -I . \\
        tests/deep_agents/smolvla/test_train_step.mojo

`test_denoise_backward.mojo` gates the expert's sixteen layers. This gates
what wraps them — `action_in`, the time MLP, `action_out` — and, more to the
point, the SEAMS between them, which are where a chain like this actually
goes wrong:

  * `action_out.vjp` feeding `denoise.backward`'s `grad_out`;
  * `denoise.backward`'s `grad_x` feeding `time_mlp_out.vjp`;
  * the token-concat split, where only the action half of each token carries a
    gradient anywhere and the time half must be dropped.

Every one of those is a plumbing join that produces a finite, plausible
gradient when wired wrong, and none of them is visible in a loss curve.

⚠ **The heads are gated by finite differences of the WHOLE step**, not against
the expert gate's numbers. A head weight's gradient passes through all sixteen
expert layers, so if the join between the two drivers were wrong this is the
only place it shows.

⚠ The reference is a Float64 transcription of nothing — the loss here is the
model's own forward, which is fp32. So this inherits the accuracy limits
`test_denoise_backward.mojo` measured, and uses the same remedy: Richardson
extrapolation, a step size chosen from the measurement, and `||err||/||fd||`
per parameter group rather than a per-component ratio that below-noise
components dominate.

## MEASURED — four seam defects, and the one that exposed a blind spot

    defect                                   CPU legs [2]-[3]    GPU leg [5]
    A1  token_split_a takes the TIME half
        ... in the CPU branch                action_in 1.04      —
        ... in the GPU KERNEL                nothing, 3.0e-05    0.171, 55/318
    A2  act.vjp given its OUTPUT not input   action_in 0.29      —
    A3  time_mlp_out.vjp fed GSUF,
        skipping denoise.backward            action_in 0.94      —

A1 is the useful one, and it was an accident. Planted in
`_token_split_a_kernel`, it changed **nothing** — every number identical to a
clean run — because that is the GPU kernel and legs [2]-[3] are CPU-only. The
same defect in the CPU branch four lines away moves `action_in.weight` to 1.04
norm-relative with 47 of 48 components outside the band.

Two implementations of one rule, one of them never executed, and the gate
could not tell. Leg [5] exists because of that, and re-running A1 against the
kernel now reports 0.171 with 55 of 318 outside.

⚠ Note which parameter reports every one of these. `action_in` is the FURTHEST
from the loss, so it is the only one downstream of every seam. A gate that
checked `action_out` alone — nearest the loss, easiest to reason about — would
have caught none of the four.
"""

from std.math import abs, sqrt
from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.deep_agents.smolvla.text import SMOLLM_THETA
from mojo_rl.deep_agents.smolvla.expert import SmolVLAExpert
from mojo_rl.deep_agents.smolvla.kv_cache import SmolVLAKVCache
from mojo_rl.deep_agents.smolvla.fused import SmolVLADenoise
from mojo_rl.deep_agents.smolvla.train_step import SmolVLATrainStep
from mojo_rl.deep_agents.smolvla.flow_loss import build_xt_ut, sample_noise
from mojo_rl.deep_agents.smolvla.attn_mask import att_2d_mask, smolvla_ar

comptime P = 6
comptime CHUNK = 3
comptime B = 1
comptime L = 2               # one self + one cross
comptime EW = 8
comptime EFF = 12
comptime W = 8
comptime HEADS = 2
comptime NKV = 1
comptime HD = 4
comptime KVW = NKV * HD
comptime ADIM = 6
comptime ADIM_REAL = 3
"""⚠ ADIM != EW ON PURPOSE. The suffix stream is EW-wide and the action space
is ADIM-wide, and at the checkpoint they are 720 and 32. Making them equal in
the fixture would be convenient and would turn a whole class of defect — a
gradient handed to the wrong stage of the chain — from a compile error into a
silent one, in the exact place this file exists to check."""
comptime XN = B * CHUNK * ADIM
comptime PKV = B * P * KVW

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

# ⚠ Chosen from a sweep, and this fixture shows BOTH limits at once because
# its head groups span a wide range of gradient magnitude. ||err||/||fd||:
#
#     h pair        time_mlp_out.bias      action_in.weight      TOTAL
#                   (|grad|max 1.61)       (|grad|max 0.26)
#     8e-2/4e-2         6.2e-03               2.5e-05           3.6e-03
#     4e-2/2e-2         3.3e-04               5.0e-05           1.9e-04
#     2e-2/1e-2         2.0e-05               7.8e-05           2.3e-05
#     1e-2/5e-3         1.6e-05               2.1e-04           4.3e-05
#
# The large-gradient group is TRUNCATION-limited and improves as h shrinks;
# the small-gradient one is ROUNDOFF-limited and gets worse. They pull in
# opposite directions and the total is minimised between them. A band argued
# down to fit one of these would have been fitting an artefact of the
# reference, not measuring the gradient.
comptime FD_H = 2.0e-2
comptime FD_H2 = 1.0e-2
comptime NORM_BAND = 3.0e-3
comptime N_KINDS = 8


def _pname(which: Int) -> String:
    if which == 0: return String("action_in.weight ")
    if which == 1: return String("action_in.bias   ")
    if which == 2: return String("time_mlp_in.weight ")
    if which == 3: return String("time_mlp_in.bias   ")
    if which == 4: return String("time_mlp_out.weight")
    if which == 5: return String("time_mlp_out.bias  ")
    if which == 6: return String("action_out.weight  ")
    return String("action_out.bias    ")


def _psize(which: Int) -> Int:
    if which == 0: return ADIM * EW
    if which == 1: return EW
    if which == 2: return 2 * EW * EW
    if which == 3: return EW
    if which == 4: return EW * EW
    if which == 5: return EW
    if which == 6: return EW * ADIM
    return ADIM


def _pget(
    which: Int, t: Int, mut ai: AIn, mut ti: TIn, mut to: TOut, mut ao: AOut
) raises -> Scalar[DT]:
    if which == 0: return ai.weight.val.data[t]
    if which == 1: return ai.bias.val.data[t]
    if which == 2: return ti.weight.val.data[t]
    if which == 3: return ti.bias.val.data[t]
    if which == 4: return to.weight.val.data[t]
    if which == 5: return to.bias.val.data[t]
    if which == 6: return ao.weight.val.data[t]
    return ao.bias.val.data[t]


def _pset(
    which: Int, t: Int, v: Scalar[DT], mut ai: AIn, mut ti: TIn, mut to: TOut,
    mut ao: AOut,
) raises:
    if which == 0: ai.weight.val.data[t] = v
    elif which == 1: ai.bias.val.data[t] = v
    elif which == 2: ti.weight.val.data[t] = v
    elif which == 3: ti.bias.val.data[t] = v
    elif which == 4: to.weight.val.data[t] = v
    elif which == 5: to.bias.val.data[t] = v
    elif which == 6: ao.weight.val.data[t] = v
    else: ao.bias.val.data[t] = v


def _pgrad(
    which: Int, t: Int, mut ai: AIn, mut ti: TIn, mut to: TOut, mut ao: AOut
) raises -> Scalar[DT]:
    if which == 0: return ai.weight.grd.data[t]
    if which == 1: return ai.bias.grd.data[t]
    if which == 2: return ti.weight.grd.data[t]
    if which == 3: return ti.bias.grd.data[t]
    if which == 4: return to.weight.grd.data[t]
    if which == 5: return to.bias.grd.data[t]
    if which == 6: return ao.weight.grd.data[t]
    return ao.bias.grd.data[t]


def _pdownload(
    which: Int, mut ai: AIn, mut ti: TIn, mut to: TOut, mut ao: AOut,
    d: DeviceContext,
) raises:
    """Bring one probed `.grd` back from the device."""
    if which == 0: ai.weight.grd.download(d)
    elif which == 1: ai.bias.grd.download(d)
    elif which == 2: ti.weight.grd.download(d)
    elif which == 3: ti.bias.grd.download(d)
    elif which == 4: to.weight.grd.download(d)
    elif which == 5: to.bias.grd.download(d)
    elif which == 6: ao.weight.grd.download(d)
    else: ao.bias.grd.download(d)


struct Cmp(Movable):
    """Compared / differing, plus the group's norm-relative error."""
    var n: Int
    var bad: Int
    var worst: Float64
    var floor: Float64
    var num: Float64
    var den: Float64

    def __init__(out self):
        self.n = 0
        self.bad = 0
        self.worst = 0.0
        self.floor = 1.0e-6
        self.num = 0.0
        self.den = 0.0

    def __init__(out self, *, deinit move: Self):
        self.n = move.n
        self.bad = move.bad
        self.worst = move.worst
        self.floor = move.floor
        self.num = move.num
        self.den = move.den

    def set_group(mut self, ref fd: List[Float64]):
        var mx = 0.0
        for i in range(len(fd)):
            if abs(fd[i]) > mx:
                mx = abs(fd[i])
        self.floor = mx * 1.0e-3
        if self.floor < 1.0e-6:
            self.floor = 1.0e-6

    def add(mut self, got: Float64, want: Float64):
        self.n += 1
        var sc = abs(want)
        if sc < self.floor:
            sc = self.floor
        var rel = abs(got - want) / sc
        if rel > self.worst:
            self.worst = rel
        if rel > 3.0e-2:
            self.bad += 1
        self.num += (got - want) * (got - want)
        self.den += want * want

    def rel_norm(self) -> Float64:
        if self.den <= 0.0:
            return 0.0
        return sqrt(self.num / self.den)


def main() raises:
    print("=" * 70)
    print("SmolVLA training step: loss -> heads -> expert")
    print("=" * 70)
    print("  P", P, " chunk", CHUNK, " layers", L, " EW", EW, " ADIM", ADIM,
          "(real", ADIM_REAL, ")")

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

    # A FIXED noise, action chunk and timestep — the gate differentiates a
    # deterministic function, so nothing may be resampled between evaluations.
    var noise = Tensor.alloc(XN)
    var acts = Tensor.alloc(XN)
    for i in range(XN):
        noise.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.07
        acts.data[i] = Scalar[DT](0)
    for b in range(B):
        for t in range(CHUNK):
            for d in range(ADIM_REAL):
                acts.data[b * CHUNK * ADIM + t * ADIM + d] = Scalar[DT](
                    ((t * 5 + d * 3) % 7) - 3
                ) * 0.2
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

    # ── [1] one step ─────────────────────────────────────────────────────
    var l0 = st.run["cpu", P](e, c, den, ai, ti, to, ao, x_t, u_t, valid, N_VALID, None)
    print("  [1] loss", l0)
    assert_true(l0 > 0.0, "a zero loss means the step did not run")

    var snap = List[Float64]()
    for which in range(N_KINDS):
        for t in range(_psize(which)):
            snap.append(Float64(_pgrad(which, t, ai, ti, to, ao)))

    # ── [2] the head gradients vs Richardson central differences ─────────
    # ⚠ Every evaluation re-runs the WHOLE step, which also re-accumulates
    # into every `.grd`. That is why the gradients were snapshotted above:
    # by the end of this loop the live `.grd` slabs hold the sum of hundreds
    # of perturbed backward passes, which is meaningless and would look like
    # an enormous gradient.
    print("  [2] head gradients, all", N_KINDS, "parameter kinds:")
    var total = Cmp()
    var k0 = 0
    var nonzero = 0
    for which in range(N_KINDS):
        var fdw = List[Float64]()
        for t in range(_psize(which)):
            var keep = _pget(which, t, ai, ti, to, ao)
            _pset(which, t, Scalar[DT](Float64(keep) + FD_H), ai, ti, to, ao)
            var ap = Float64(_pget(which, t, ai, ti, to, ao))
            var lp = st.run["cpu", P](e, c, den, ai, ti, to, ao, x_t, u_t, valid,
                                      N_VALID, None)
            _pset(which, t, Scalar[DT](Float64(keep) - FD_H), ai, ti, to, ao)
            var am = Float64(_pget(which, t, ai, ti, to, ao))
            var lm = st.run["cpu", P](e, c, den, ai, ti, to, ao, x_t, u_t, valid,
                                      N_VALID, None)
            _pset(which, t, Scalar[DT](Float64(keep) + FD_H2), ai, ti, to, ao)
            var ap2 = Float64(_pget(which, t, ai, ti, to, ao))
            var lp2 = st.run["cpu", P](e, c, den, ai, ti, to, ao, x_t, u_t, valid,
                                       N_VALID, None)
            _pset(which, t, Scalar[DT](Float64(keep) - FD_H2), ai, ti, to, ao)
            var am2 = Float64(_pget(which, t, ai, ti, to, ao))
            var lm2 = st.run["cpu", P](e, c, den, ai, ti, to, ao, x_t, u_t, valid,
                                       N_VALID, None)
            _pset(which, t, keep, ai, ti, to, ao)
            var d1 = (lp - lm) / (ap - am)
            var d2 = (lp2 - lm2) / (ap2 - am2)
            fdw.append((4.0 * d2 - d1) / 3.0)
        var ck = Cmp()
        ck.set_group(fdw)
        var mag = 0.0
        for t in range(_psize(which)):
            ck.add(snap[k0 + t], fdw[t])
            total.add(snap[k0 + t], fdw[t])
            if abs(snap[k0 + t]) > mag:
                mag = abs(snap[k0 + t])
        k0 += _psize(which)
        if mag > 0.0:
            nonzero += 1
        print("      " + _pname(which) + ": " + String(ck.n)
              + " compared, ||err||/||fd|| " + String(ck.rel_norm())
              + ", outside band " + String(ck.bad) + ", |grad|max "
              + String(mag))
        assert_true(
            ck.rel_norm() < NORM_BAND,
            "gradient of " + _pname(which) + " disagrees with a central"
            " difference in norm",
        )
    print("  [3] head kinds with a nonzero gradient:", nonzero, "of",
          N_KINDS)
    assert_true(
        nonzero == N_KINDS,
        "a head parameter got no gradient at all — the chain is broken"
        " somewhere above it",
    )
    print("      TOTAL: compared", total.n, " ||err||/||fd||",
          total.rel_norm(), " outside band", total.bad)
    assert_true(
        total.rel_norm() < NORM_BAND, "the head gradients disagree in norm"
    )

    # ── [4] the GPU path of the whole step ───────────────────────────────
    # ⚠ Everything above is CPU — finite differences need hundreds of cheap
    # forwards. FOUND BY AN ABLATION: a deliberate defect planted in
    # `_token_split_a_kernel` changed nothing at all, because that is the GPU
    # kernel and nothing above it runs on a GPU. The same defect in the CPU
    # branch moves `action_in.weight` to 1.04 norm-relative. Two code paths,
    # one of them unexercised, and the gate could not tell.
    var d = DeviceContext()
    var eg = Expert.make["gpu", Deterministic](Optional(d))
    var cg = Cache.make["gpu"](Optional(d))
    var deng = Den.make["gpu"](ms, mc, Optional(d))
    var stg = Step.make["gpu"](Optional(d))
    var aig = AIn.make["gpu", Deterministic](Optional(d))
    var tig = TIn.make["gpu", Deterministic](Optional(d))
    var tog = TOut.make["gpu", Deterministic](Optional(d))
    var aog = AOut.make["gpu", Deterministic](Optional(d))

    var kg = Tensor.alloc(PKV)
    var vg = Tensor.alloc(PKV)
    for l in range(L):
        for i in range(PKV):
            kg.data[i] = Scalar[DT](((i * 31 + l * 7) % 13) - 6) * 0.11
            vg.data[i] = Scalar[DT](((i * 17 + l * 5) % 11) - 5) * 0.09
        kg.upload(d)
        vg.upload(d)
        cg.write_prefix["gpu"](l, kg, vg, Optional(d))

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
    var lg = stg.run["gpu", P](eg, cg, deng, aig, tig, tog, aog, xg, ug, validg,
                               N_VALID, Optional(d))
    d.synchronize()
    print("  [4] GPU loss", lg, " vs CPU", l0, " diff", abs(lg - l0))
    assert_true(
        abs(lg - l0) < 1.0e-5,
        "the GPU forward disagrees with the CPU one, so leg [5] cannot"
        " attribute a gradient difference to the backward",
    )

    var gc = Cmp()
    var gsnap = List[Float64]()
    for which in range(N_KINDS):
        _pdownload(which, aig, tig, tog, aog, d)
        for t in range(_psize(which)):
            gsnap.append(Float64(_pgrad(which, t, aig, tig, tog, aog)))
    gc.set_group(snap)
    for i in range(len(snap)):
        gc.add(gsnap[i], snap[i])
    print("  [5] GPU head gradients vs CPU: compared", gc.n,
          " ||err||/||cpu||", gc.rel_norm(), " outside band", gc.bad)
    assert_equal(gc.n, len(snap), "the GPU leg must compare every component")
    assert_true(
        gc.rel_norm() < 1.0e-4,
        "the GPU training step disagrees with the CPU one",
    )

    print()
    print("PASSED — " + String(total.n) + " head-gradient components through"
          " the whole step, " + String(gc.n) + " against the GPU")
