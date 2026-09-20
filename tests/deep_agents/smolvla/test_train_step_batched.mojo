# +--------------------------------------------------------------------------+ #
# | B observations in one forward reproduce B separate ones
# +--------------------------------------------------------------------------+ #
"""The batched training step against the B = 1 step it replaces.

    pixi run -e apple mojo run -I . \\
        tests/deep_agents/smolvla/test_train_step_batched.mojo

`B` is a comptime parameter of every container in the port — the prefill,
the KV cache, the denoiser and its tape, the train step, the prefix tail —
and until this file only the KV cache and the block attention had a B > 1
leg. A batched path that is wrong is not loud: a batch element conditioned
on its neighbour's prefix, a state token written into the wrong row, a
gradient that sums over rows where it should not, all give finite numbers
and a policy that trains worse than it should. This gate says the batched
step IS the B = 1 step, four times over:

  [1] prefill: four observations through a B = 1 policy, then the same four
      through a B = 4 policy with identical weights — row k of the batched
      prefill output equals run k's, to a tolerance set by the GEMM's tiling
      (the M = 4x140 matmul may sum in a different order than M = 140).
  [2] loss: the batched loss equals the sum of the four losses, all under one
      denominator (`n_terms` is the group's total either way).
  [3] gradients: every trainable parameter — the expert's two layers, the
      four action projections — accumulated over four B = 1 backwards equals
      the ONE B = 4 backward, norm-relative per parameter.

      ⚠ THE BAND IS MEASURED, NOT ASSERTED. The first run of this file
      printed 4e-4 on every parameter upstream of the cross layer's
      attention and 1e-7 on every parameter downstream of it, with dV clean
      and dQ/dK off — the signature of `dS = p (dP − Σ p dP)` cancelling
      catastrophically under a near-uniform softmax, which is what
      `Deterministic` weights at real widths produce. A 1e-7 difference in
      the attention's inputs (the M = 4x50 GEMM tiles differently from
      M = 50) becomes 4e-4 in dS. That is a property of the fixture, not
      of the batch — but a band of 3e-3 written to pass it would also pass
      a real defect. So the gate measures the B = 1 path's OWN sensitivity:
      a third pass, four B = 1 steps with the inputs AND every weight
      perturbed by one ulp (inputs alone under-estimate it ~10x), and the
      batched run must sit within 10x of that per parameter — and under
      1e-3 outright. A batched path that mixed rows, dropped a row's
      gradient or double-counted one is off by O(1), not O(sensitivity).
  [4] NOT VACUOUS: one row's image segment perturbed, the batched run
      repeated — prefill row 2 moves, rows 0, 1 and 3 do not (bit for bit),
      and the gradients move. [1] already needs the rows to differ (it also
      asserts each row is NOT its neighbour's); what [4] adds is row
      INDEPENDENCE — a batched prefill in which row 2's tokens leaked into
      row 3's attention would pass [1] at a loose band and fail here at
      zero bits.

⚠ A SHALLOW FIXTURE: 2 VLM / 2 expert / 2 vision layers, deterministic
weights, TWO policies (B = 1 and B = 4) — affordable only because it is
shallow (`test_policy.mojo` explains the 3.2 GB a published-depth policy
costs). The image segments are synthetic: the vision tower is not in the
batched path at all, the cache is, and a cache row is just floats.

⚠ The bands are GPU-vs-GPU on ONE device, so they measure summation order
only: ~1e-6 on Metal's fp32 GEMM, up to ~1e-3 under TF32 on CUDA. Set from
the CUDA case; the Metal run prints the number it actually saw.
"""

from std.math import abs, sqrt
from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.core.param import ParamVisitorRT, walk_params
from mojo_rl.deep_agents.smolvla.policy import SmolVLAPolicy
from mojo_rl.deep_agents.smolvla.normalize import SmolVLAStats
from mojo_rl.deep_agents.smolvla.tasks import TaskTokens
from mojo_rl.deep_agents.smolvla.train_step import SmolVLATrainStep
from mojo_rl.deep_agents.smolvla.flow_loss import build_xt_ut
from mojo_rl.deep_agents.smolvla.heads import (
    SMOLVLA_ACTION_DIM, SMOLVLA_EXPERT_W,
)
from mojo_rl.deep_agents.smolvla.text import SMOLLM_DIM
from mojo_rl.deep_agents.smolvla.expert import EXPERT_FF

comptime TABLE = "tools/vla/smolvla_tasks_record-test_20260828_092736.tsv"
comptime N_CAM = 2
comptime N_LANG = 6
comptime CHUNK = 50
comptime STEPS = 10
comptime VLM_LAYERS = 2
comptime VIS_LAYERS = 2
comptime RDIM = 6
comptime PAD = SMOLVLA_ACTION_DIM
comptime NB = 4
comptime Pol1 = SmolVLAPolicy[
    N_CAM, N_LANG, CHUNK, STEPS, 1, VLM_LAYERS, VIS_LAYERS, True
]
comptime PolB = SmolVLAPolicy[
    N_CAM, N_LANG, CHUNK, STEPS, NB, VLM_LAYERS, VIS_LAYERS, True
]
comptime Step1 = SmolVLATrainStep[
    CHUNK, RDIM, PAD, SMOLVLA_EXPERT_W, 1, VLM_LAYERS, EXPERT_FF, SMOLLM_DIM
]
comptime StepB = SmolVLATrainStep[
    CHUNK, RDIM, PAD, SMOLVLA_EXPERT_W, NB, VLM_LAYERS, EXPERT_FF, SMOLLM_DIM
]
comptime SEG = Pol1.Prefix.IMG_SEG
comptime ROW = CHUNK * PAD
comptime PW = Pol1.P * Pol1.W
comptime PRE_BAND = 1.0e-5
"""Prefill rows, batched vs single: GEMM tiling only. Metal printed 0.0."""
comptime SENS_MULT = 10.0
"""The batched error may be this many times the measured sensitivity."""
comptime SENS_FLOOR = 1.0e-6
"""...or below this outright — the level of one differently-ordered sum."""
comptime ABS_BAND = 1.0e-3
"""Whatever the sensitivity says, the batched error may never exceed this:
a dropped, doubled or mixed row is O(0.25..1)."""
comptime PERTURB = 1.5e-7
"""Relative perturbation of the segments for the sensitivity pass: one ulp,
the size of the difference a different summation order makes."""


def robot_stats() raises -> SmolVLAStats:
    var s = SmolVLAStats()
    var m: List[Float32] = [16.64, -29.97, 31.07, 73.73, 41.12, 26.27]
    var d: List[Float32] = [21.00, 54.38, 51.43, 17.93, 18.72, 9.21]
    for i in range(RDIM):
        s.state_mean.append(m[i])
        s.state_std.append(d[i])
        s.action_mean.append(m[i])
        s.action_std.append(d[i])
    return s^


struct GradSnap(ParamVisitorRT):
    """Every parameter's gradient, by name, brought to the host."""

    var names: List[String]
    var vals: List[List[Float64]]

    def __init__(out self):
        self.names = List[String]()
        self.vals = List[List[Float64]]()

    def visit_rt[target: StaticString](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        n: Int,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target != "cpu":
            grad.download(ctx.value())
        var l = List[Float64]()
        for i in range(n):
            l.append(Float64(grad.data[i]))
        self.names.append(name)
        self.vals.append(l^)


struct SharpenQK(ParamVisitorRT):
    """Re-draw the expert's q/k weights from a hashed pattern, identically in
    both policies.

    ⚠ WHY. `Deterministic` writes the same `(i % 7 - 3) * 0.1` ribbon into
    every row, so every key is nearly the same vector, every score is nearly
    the same number, and the softmax is nearly uniform. Its backward,
    `dS = p (dP − Σ p dP)`, then cancels catastrophically: the first run of
    this file saw a one-ulp input change move the upstream gradients by 3e-2,
    which made the band on those parameters 0.28 — wide enough to pass a
    dropped row. Distinct keys give a peaked softmax and a backward that is
    conditioned like the real model's, so the band can be tight.
    """

    def __init__(out self):
        pass

    def visit_rt[target: StaticString](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        n: Int,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if not (
            name.endswith("self_attn.q.weight")
            or name.endswith("self_attn.k.weight")
        ):
            return
        for i in range(n):
            var z = UInt64(i + 1) * UInt64(0x9E3779B97F4A7C15)
            z = (z ^ (z >> 31)) * UInt64(0xBF58476D1CE4E5B9)
            z = z ^ (z >> 29)
            var u = Float64(Int(z % UInt64(20001))) / 10000.0 - 1.0
            param.data[i] = Scalar[DT](u * 0.06)
        comptime if target != "cpu":
            param.upload_resident(ctx.value())
        # ⚠ The version bump: leaves that cache a derived form of the weight
        # (the bf16 cast, split-K's padded copy) refresh on it.
        param.version += 1


struct Nudge(ParamVisitorRT):
    """Every weight and bias moved by one ulp, alternating sign — the
    sensitivity pass's model of "every rounding in the path came out
    differently", which is what a batched run's re-tiled GEMMs amount to.
    Perturbing the INPUTS alone under-estimates it by ~10x (measured)."""

    def __init__(out self):
        pass

    def visit_rt[target: StaticString](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        n: Int,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        for i in range(n):
            var sgn = Scalar[DT](1.0) if (i % 2 == 0) else Scalar[DT](-1.0)
            param.data[i] = param.data[i] * (
                Scalar[DT](1.0) + sgn * Scalar[DT](PERTURB)
            )
        comptime if target != "cpu":
            param.upload_resident(ctx.value())
        param.version += 1


def is_cross_k_bias(name: String) -> Bool:
    """`expert.layers.<odd>.self_attn.k.bias` — a CROSS layer's key bias.

    Its true gradient is exactly zero: a constant added to every key shifts
    every score of a query by the same amount and the softmax does not move.
    What the backward computes there is fp32 cancellation noise, so its
    batched-vs-single ratio is noise over noise and says nothing. Skipped
    by name, with the reason here rather than in a wider band."""
    for layer in range(VLM_LAYERS):
        if layer % 2 == 1 and name == (
            "expert.layers." + String(layer) + ".self_attn.k.bias"
        ):
            return True
    return False


def snapshot(mut t: Tensor, off: Int, n: Int, d: DeviceContext) raises -> List[Float64]:
    t.download(d)
    var out = List[Float64](unsafe_uninit_length=n)
    for i in range(n):
        out[i] = Float64(t.data[off + i])
    return out^


def rel_norm(ref a: List[Float64], ref b: List[Float64]) raises -> Float64:
    assert_equal(len(a), len(b), "vectors differ in length")
    var num = 0.0
    var den = 0.0
    for i in range(len(a)):
        num += (a[i] - b[i]) * (a[i] - b[i])
        den += b[i] * b[i]
    if den <= 0.0:
        return sqrt(num)
    return sqrt(num / den)


def n_bit_diff(ref a: List[Float64], ref b: List[Float64]) -> Int:
    var n = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            n += 1
    return n


def main() raises:
    print("=" * 70)
    print("SmolVLA batched training step: B =", NB, "vs", NB, "x B = 1")
    print("=" * 70)

    var tasks = TaskTokens(String(TABLE))
    var ids = tasks.for_index(0)
    assert_equal(len(ids), N_LANG, "instruction table vs N_LANG")
    var d = DeviceContext()

    print("  building two shallow policies (", VLM_LAYERS, "VLM /",
          VLM_LAYERS, "expert /", VIS_LAYERS, "vision layers)")
    var p1 = Pol1.make["gpu", Deterministic](Optional(d))
    var pb = PolB.make["gpu", Deterministic](Optional(d))
    p1.stats = robot_stats()
    pb.stats = robot_stats()
    var s1 = Step1.make["gpu"](Optional(d))
    var sb = StepB.make["gpu"](Optional(d))
    # Same hashed q/k in both — see `SharpenQK`.
    var sh1 = SharpenQK()
    walk_params["gpu"](p1.expert, sh1, Optional(d), String("expert"))
    var shb = SharpenQK()
    walk_params["gpu"](pb.expert, shb, Optional(d), String("expert"))

    # ── the four observations ────────────────────────────────────────────
    var segs = List[Float32](length=NB * SEG, fill=Float32(0))
    for k in range(NB):
        for i in range(SEG):
            segs[k * SEG + i] = Float32(((i * 7 + k * 131) % 97) - 48) / 24.0
    var poses = List[Float32]()
    for k in range(NB):
        var base: List[Float32] = [12.0, -40.0, 22.0, 70.0, 35.0, 20.0]
        for j in range(RDIM):
            poses.append(base[j] + Float32(k * 9 - 13) * 0.7 * Float32(j + 1))
    var acts = Tensor.alloc(NB * ROW)
    var noise = Tensor.alloc(NB * ROW)
    var valid = Tensor.alloc(NB * CHUNK)
    var total_valid = 0
    for k in range(NB):
        for t in range(CHUNK):
            # observation 1 runs off its episode after step 39
            var inside = not (k == 1 and t >= 40)
            valid.data[k * CHUNK + t] = Scalar[DT](1.0) if inside else Scalar[DT](0.0)
            if inside:
                total_valid += 1
            for j in range(PAD):
                var i = k * ROW + t * PAD + j
                noise.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.11
                if j < RDIM:
                    acts.data[i] = Scalar[DT](((t * 5 + j * 3 + k * 11) % 7) - 3) * 0.3
    var times = List[Float64]()
    times.append(0.2)
    times.append(0.55)
    times.append(0.9)
    times.append(0.35)
    print("  ", NB, "observations, total valid timesteps", total_valid,
          "(observation 1 padded from step 40)")

    # ── B = 1, four times, gradients accumulating ────────────────────────
    def run_single(
        mut p1: Pol1, mut s1: Step1, ref segs: List[Float32],
        ref poses: List[Float32], ref ids: List[Int],
        mut noise: Tensor, mut acts: Tensor, mut valid: Tensor,
        ref times: List[Float64], total_valid: Int, d: DeviceContext,
        mut pre: List[List[Float64]], mut g: GradSnap,
    ) raises -> Float64:
        p1.expert.zero_grad["gpu"](Optional(d))
        p1.action_in.zero_grad["gpu"](Optional(d))
        p1.time_mlp_in.zero_grad["gpu"](Optional(d))
        p1.time_mlp_out.zero_grad["gpu"](Optional(d))
        p1.action_out.zero_grad["gpu"](Optional(d))
        var loss = 0.0
        var seg_k = List[Float32](length=SEG, fill=Float32(0))
        var pose_k = List[Float32](length=RDIM, fill=Float32(0))
        var a_k = Tensor.alloc(ROW)
        var n_k = Tensor.alloc(ROW)
        var v_k = Tensor.alloc(CHUNK)
        var t_k = Tensor.alloc(1)
        var x_k = Tensor()
        var u_k = Tensor()
        for k in range(NB):
            for i in range(SEG):
                seg_k[i] = segs[k * SEG + i]
            for j in range(RDIM):
                pose_k[j] = poses[k * RDIM + j]
            p1.build_prefix_from_segment["gpu"](seg_k, ids, pose_k, Optional(d))
            pre.append(snapshot(p1.prefill_out, 0, PW, d))
            for i in range(ROW):
                a_k.data[i] = acts.data[k * ROW + i]
                n_k.data[i] = noise.data[k * ROW + i]
            for t in range(CHUNK):
                v_k.data[t] = valid.data[k * CHUNK + t]
            t_k.data[0] = Scalar[DT](times[k])
            a_k.upload_resident(d)
            n_k.upload_resident(d)
            v_k.upload_resident(d)
            t_k.upload_resident(d)
            build_xt_ut["gpu", 1, ROW](n_k, a_k, t_k, x_k, u_k, Optional(d))
            var tl = List[Float64]()
            tl.append(times[k])
            s1.set_times["gpu"](tl, Optional(d))
            loss += s1.run["gpu", Pol1.P](
                p1.expert, p1.cache, p1.denoiser, p1.action_in,
                p1.time_mlp_in, p1.time_mlp_out, p1.action_out, x_k, u_k,
                v_k, total_valid, Optional(d),
            )
        walk_params["gpu"](p1.expert, g, Optional(d), String("expert"))
        walk_params["gpu"](p1.action_in, g, Optional(d), String("action_in"))
        walk_params["gpu"](
            p1.time_mlp_in, g, Optional(d), String("time_mlp_in")
        )
        walk_params["gpu"](
            p1.time_mlp_out, g, Optional(d), String("time_mlp_out")
        )
        walk_params["gpu"](p1.action_out, g, Optional(d), String("action_out"))
        return loss

    var pre1 = List[List[Float64]]()
    var g1 = GradSnap()
    var loss1 = run_single(
        p1, s1, segs, poses, ids, noise, acts, valid, times, total_valid, d,
        pre1, g1,
    )
    print("  B = 1 x", NB, ": loss", loss1, " parameters snapshotted",
          len(g1.names))

    # ── the same four, everything perturbed by ~1 ulp: the path's own ────
    # ── sensitivity. Inputs AND weights — see `Nudge`. p1 is not used ───
    # ── again after this, so nothing is restored. ────────────────────────
    var nudge = Nudge()
    walk_params["gpu"](p1.expert, nudge, Optional(d), String("expert"))
    walk_params["gpu"](p1.action_in, nudge, Optional(d), String("action_in"))
    walk_params["gpu"](p1.time_mlp_in, nudge, Optional(d), String("tmi"))
    walk_params["gpu"](p1.time_mlp_out, nudge, Optional(d), String("tmo"))
    walk_params["gpu"](p1.action_out, nudge, Optional(d), String("action_out"))
    # Not the tower: leg [1] shows the prefill rows are BIT-IDENTICAL
    # batched vs single, so no difference originates there.
    var segs_eps = segs.copy()
    for i in range(len(segs_eps)):
        var sgn = Float32(1.0) if (i % 2 == 0) else Float32(-1.0)
        segs_eps[i] = segs_eps[i] * (Float32(1.0) + sgn * Float32(PERTURB))
    var pre_eps = List[List[Float64]]()
    var g_eps = GradSnap()
    var loss_eps = run_single(
        p1, s1, segs_eps, poses, ids, noise, acts, valid, times, total_valid,
        d, pre_eps, g_eps,
    )
    print("  B = 1 x", NB, "perturbed by", PERTURB, ": loss", loss_eps,
          " (rel", abs(loss_eps - loss1) / abs(loss1), ")")

    # ── B = 4, once ──────────────────────────────────────────────────────
    def run_batched(
        mut pb: PolB, mut sb: StepB, ref segs: List[Float32],
        ref poses: List[Float32], ref ids: List[Int],
        mut noise: Tensor, mut acts: Tensor, mut valid: Tensor,
        ref times: List[Float64], total_valid: Int, d: DeviceContext,
    ) raises -> Float64:
        pb.expert.zero_grad["gpu"](Optional(d))
        pb.action_in.zero_grad["gpu"](Optional(d))
        pb.time_mlp_in.zero_grad["gpu"](Optional(d))
        pb.time_mlp_out.zero_grad["gpu"](Optional(d))
        pb.action_out.zero_grad["gpu"](Optional(d))
        pb.build_prefix_from_segment["gpu"](segs, ids, poses, Optional(d))
        var tb = Tensor.alloc(NB)
        for k in range(NB):
            tb.data[k] = Scalar[DT](times[k])
        noise.upload_resident(d)
        acts.upload_resident(d)
        valid.upload_resident(d)
        tb.upload_resident(d)
        var xb = Tensor()
        var ub = Tensor()
        build_xt_ut["gpu", NB, ROW](noise, acts, tb, xb, ub, Optional(d))
        sb.set_times["gpu"](times, Optional(d))
        return sb.run["gpu", PolB.P](
            pb.expert, pb.cache, pb.denoiser, pb.action_in, pb.time_mlp_in,
            pb.time_mlp_out, pb.action_out, xb, ub, valid, total_valid,
            Optional(d),
        )

    var lossb = run_batched(
        pb, sb, segs, poses, ids, noise, acts, valid, times, total_valid, d
    )
    var gb = GradSnap()
    walk_params["gpu"](pb.expert, gb, Optional(d), String("expert"))
    walk_params["gpu"](pb.action_in, gb, Optional(d), String("action_in"))
    walk_params["gpu"](pb.time_mlp_in, gb, Optional(d), String("time_mlp_in"))
    walk_params["gpu"](pb.time_mlp_out, gb, Optional(d), String("time_mlp_out"))
    walk_params["gpu"](pb.action_out, gb, Optional(d), String("action_out"))

    # [1] prefill rows
    var worst_pre = 0.0
    for k in range(NB):
        var rk = snapshot(pb.prefill_out, k * PW, PW, d)
        var r = rel_norm(rk, pre1[k])
        if r > worst_pre:
            worst_pre = r
        # and NOT another row's: the rows are distinct observations
        var other = rel_norm(rk, pre1[(k + 1) % NB])
        assert_true(other > 1.0e-2, "prefill rows are not distinct — the"
                    " fixture cannot tell a row mix-up")
    print("  [1] prefill: worst row rel-norm", worst_pre, " (band",
          PRE_BAND, ")")
    assert_true(worst_pre <= PRE_BAND, "batched prefill row != its B=1 run")

    # [2] loss
    var lrel = abs(lossb - loss1) / abs(loss1)
    print("  [2] loss: batched", lossb, " sum of four", loss1, " rel", lrel)
    assert_true(lrel <= 1.0e-5, "batched loss != sum of the four losses")

    # [3] gradients, against the measured sensitivity
    assert_equal(len(g1.names), len(gb.names), "parameter walks differ")
    assert_equal(len(g1.names), len(g_eps.names), "sensitivity walk differs")
    var worst_ratio = 0.0
    var worst_name = String("")
    var compared = 0
    var nz = 0
    var n_fail = 0
    print("  [3] per parameter: batched-vs-single | perturbed-vs-single")
    for i in range(len(g1.names)):
        assert_equal(g1.names[i], gb.names[i], "walk order differs")
        var r = rel_norm(gb.vals[i], g1.vals[i])
        var sens = rel_norm(g_eps.vals[i], g1.vals[i])
        if is_cross_k_bias(g1.names[i]):
            print("      " + g1.names[i] + "  " + String(r) + " | "
                  + String(sens) + "   (true gradient is 0 — skipped)")
            continue
        var allowed = sens * SENS_MULT
        if allowed < SENS_FLOOR:
            allowed = SENS_FLOOR
        if allowed > ABS_BAND:
            allowed = ABS_BAND
        var ratio = r / allowed
        var ok = r <= allowed
        if not ok:
            n_fail += 1
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_name = g1.names[i]
        compared += len(g1.vals[i])
        for t in range(len(g1.vals[i])):
            if g1.vals[i][t] != 0.0:
                nz += 1
        print("      " + g1.names[i] + "  " + String(r) + " | " + String(sens)
              + ("" if ok else "   <-- OUTSIDE"))
    print("      ", compared, "components in", len(g1.names),
          "parameters; nonzero", nz, "; worst batched/allowed",
          worst_ratio, "(" + worst_name + ")")
    assert_true(nz > compared // 2, "most gradients are zero — no signal")
    assert_true(n_fail == 0, "a batched gradient is outside " + String(
        SENS_MULT) + "x the path's own sensitivity")

    # [4] not vacuous: perturb row 2's segment
    var segs2 = segs.copy()
    for i in range(SEG):
        segs2[2 * SEG + i] += 0.5
    var lossp = run_batched(
        pb, sb, segs2, poses, ids, noise, acts, valid, times, total_valid, d
    )
    var moved_rows = 0
    for k in range(NB):
        var rk = snapshot(pb.prefill_out, k * PW, PW, d)
        var nd = n_bit_diff(rk, pre1[k])
        if k == 2:
            assert_true(nd > 0, "row 2's segment perturbed, its prefill did"
                        " not move")
        else:
            assert_equal(nd, 0, "row " + String(k) + " moved when only row 2"
                         " was perturbed — rows are not independent")
        if nd > 0:
            moved_rows += 1
    var gp = GradSnap()
    walk_params["gpu"](pb.expert, gp, Optional(d), String("expert"))
    var moved_g = 0.0
    for i in range(len(gp.names)):
        var r = rel_norm(gp.vals[i], gb.vals[i])
        if r > moved_g:
            moved_g = r
    print("  [4] row 2 perturbed: prefill rows moved", moved_rows, "of", NB,
          " loss", lossb, "->", lossp, " expert grads moved rel-norm", moved_g)
    assert_true(moved_g > 1.0e-3, "the gradients did not see the perturbed row")

    print("")
    print("PASSED — B =", NB, "reproduces", NB, "separate steps: prefill per"
          " row, loss, and every gradient")
