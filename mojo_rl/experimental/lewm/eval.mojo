"""LeWM nn eval — teacher-forced action-awareness / latent-planning score.

Phase F, increment 1. The foundational §10.9 check: is the learned world
model *action-aware* enough to plan? We score candidate action plans by how
well they predict the real next latents (teacher-forced), reusing the shared
`mojo_rl/planners/trajectory` CEM + random-shooter via the `ScorePlanCallback`
contract.

The score of a plan is `MSE(pred, tgt)` where, for a fixed context window:
  - `tgt`  = the encoded REAL next latents (the `tgt` graph node; it's
             action-independent → the fixed goal),
  - `pred` = the predictor's output under the candidate actions (the `pred`
             node, action-conditioned).
A model that learned action-conditioned dynamics scores the EXPERT (real)
actions below random ones; a collapsed model can't tell them apart. Lower is
better, matching the optimizer convention.

This drives the EXISTING `LeWMLossGraph` forward (encode→predict), single
shot — no latent rollout. The autoregressive MPC horizon>1 (predictor-from-
latents in latent space) is the remaining Phase F piece (see plan §2.5).
"""

from std.memory import alloc
from mojo_rl.nn.core.ptr import untracked
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major

from mojo_rl.nn.constants import DT
comptime dtype = DT                              # float32 (legacy nn.constants.dtype)
from mojo_rl.nn.core.module import Module
from mojo_rl.planners.trajectory import (
    CategoricalRandomShooter,
    CategoricalCEMOptimizer,
)
from mojo_rl.planners.trajectory.score_callback import ScorePlanCallback
from .trainer import LeWMTrainer
from .encoder import LeWMEncoder


def _mse_latent[
    ao: MutOrigin = MutAnyOrigin,
    bo: MutOrigin = MutAnyOrigin,
](
    a: UnsafePointer[Scalar[DT], ao],
    b: UnsafePointer[Scalar[DT], bo],
    n: Int,
) -> Float64:
    """Mean squared error over `n` elements. Lower is better."""
    var s: Float64 = 0.0
    for i in range(n):
        var d = Float64(a[i] - b[i])
        s += d * d
    return s / Float64(n)


struct LeWMTFScorer[
    IN_CH: Int, IMG: Int, PATCH: Int, HIDDEN: Int, ENC_HEADS: Int,
    ENC_LAYERS: Int, EMB: Int, ENC_PROJ_H: Int, ENC_FF_MULT: Int,
    T: Int, ACT: Int, SMOOTHED: Int, AE_MLP: Int,
    H: Int, N_PREDS: Int, PRED_HEADS: Int, PRED_FF: Int, DEPTH: Int,
    PRED_PROJ_H: Int, SIG_PROJ: Int, SIG_KNOTS: Int,
    BATCH: Int, target: StaticString,
](ScorePlanCallback, Movable):
    """`ScorePlanCallback` over the teacher-forced predictor.

    Holds a borrowed pointer to a trained `LeWMTrainer` + the fixed context
    window pixels (target-resident) + the precomputed goal latents. Each
    `score_plan` runs one forward with the candidate actions and returns
    `MSE(pred, goal)`.
    """
    comptime TR = LeWMTrainer[
        Self.IN_CH, Self.IMG, Self.PATCH, Self.HIDDEN, Self.ENC_HEADS,
        Self.ENC_LAYERS, Self.EMB, Self.ENC_PROJ_H, Self.ENC_FF_MULT,
        Self.T, Self.ACT, Self.SMOOTHED, Self.AE_MLP,
        Self.H, Self.N_PREDS, Self.PRED_HEADS, Self.PRED_FF, Self.DEPTH,
        Self.PRED_PROJ_H, Self.SIG_PROJ, Self.SIG_KNOTS,
        Self.BATCH, Self.target,
    ]
    comptime PIX = Self.T * Self.IN_CH * Self.IMG * Self.IMG
    comptime ACTIN = Self.T * Self.ACT
    comptime HE = Self.H * Self.EMB
    comptime NPRED = Self.BATCH * Self.HE

    var trainer: UnsafePointer[Self.TR, MutUntrackedOrigin]
    var pix: UnsafePointer[Scalar[DT], MutUntrackedOrigin]    # target-resident, fixed
    var ctx: Optional[DeviceContext]
    var act_host: UnsafePointer[Scalar[DT], MutUntrackedOrigin]   # (B, T·ACT)
    var act_dev: Optional[DeviceBuffer[DT]]
    var pred_host: List[Scalar[DT]]  # (B, H·EMB)
    var goal_host: List[Scalar[DT]]  # (B, H·EMB) fixed
    var tgt_scratch: List[Scalar[DT]]

    def __init__(
        out self,
        trainer: UnsafePointer[Self.TR, MutAnyOrigin],
        pix: UnsafePointer[Scalar[DT], MutAnyOrigin],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.trainer = untracked(trainer)
        self.pix = untracked(pix)
        self.ctx = ctx
        self.act_host = untracked(alloc[Scalar[DT]](Self.BATCH * Self.ACTIN))
        self.pred_host = List[Scalar[DT]](
            length=Self.NPRED, fill=Scalar[DT](0)
        )
        self.goal_host = List[Scalar[DT]](
            length=Self.NPRED, fill=Scalar[DT](0)
        )
        self.tgt_scratch = List[Scalar[DT]](
            length=Self.NPRED, fill=Scalar[DT](0)
        )
        self.act_dev = None
        comptime if Self.target == "gpu":
            self.act_dev = ctx.value().enqueue_create_buffer[DT](
                Self.BATCH * Self.ACTIN
            )

    def __del__(deinit self):
        self.act_host.free()

    def _act_target_ptr(self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        comptime if Self.target == "cpu":
            return self.act_host.as_unsafe_any_origin()
        else:
            return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.act_dev.value().unsafe_ptr()
            )

    def _forward(mut self, into_goal: Bool) raises:
        """Run one forward over the fixed pixels + current `act_host`,
        landing `pred` in pred_host and `tgt` in goal_host (if into_goal)
        else tgt_scratch."""
        comptime if Self.target == "gpu":
            self.ctx.value().enqueue_copy(
                self.act_dev.value(), self.act_host
            )
        var pix_t = TileTensor(self.pix, row_major[Self.BATCH, Self.PIX]())
        var act_t = TileTensor(
            self._act_target_ptr(), row_major[Self.BATCH, Self.ACTIN]()
        )
        var tgt_dst = (
            self.goal_host.unsafe_ptr() if into_goal
            else self.tgt_scratch.unsafe_ptr()
        )
        self.trainer[].forward_into(
            pix_t, act_t, self.pred_host.unsafe_ptr(), tgt_dst
        )

    def prime(mut self) raises:
        """Precompute the fixed goal latents (zero actions — `tgt` is
        action-independent)."""
        for i in range(Self.BATCH * Self.ACTIN):
            self.act_host[i] = Scalar[DT](0.0)
        self._forward(into_goal=True)

    def score_plan[
        L: TensorLayout
    ](
        mut self,
        action_plan: TileTensor[dtype, L, MutAnyOrigin],
    ) raises -> Float64:
        comptime assert action_plan.flat_rank == 3, (
            "score_plan: action_plan must be (BATCH, HORIZON, ACT)"
        )
        # HORIZON == H (only the first-H actions condition the prediction).
        # Fill act_host: first H steps from the plan, the trailing (T-H) zero.
        for i in range(Self.BATCH * Self.ACTIN):
            self.act_host[i] = Scalar[DT](0.0)
        for b in range(Self.BATCH):
            for t in range(Self.H):
                for a in range(Self.ACT):
                    self.act_host[(b * Self.T + t) * Self.ACT + a] = rebind[
                        Scalar[DT]
                    ](action_plan[b, t, a])
        self._forward(into_goal=False)
        return _mse_latent(
            self.pred_host.unsafe_ptr(), self.goal_host.unsafe_ptr(), Self.NPRED
        )


def _mean(v: List[Float64]) -> Float64:
    var s: Float64 = 0.0
    for i in range(len(v)):
        s += v[i]
    return s / Float64(len(v)) if len(v) > 0 else 0.0


def lewm_shuffled_eval[
    IN_CH: Int, IMG: Int, PATCH: Int, HIDDEN: Int, ENC_HEADS: Int,
    ENC_LAYERS: Int, EMB: Int, ENC_PROJ_H: Int, ENC_FF_MULT: Int,
    T: Int, ACT: Int, SMOOTHED: Int, AE_MLP: Int,
    H: Int, N_PREDS: Int, PRED_HEADS: Int, PRED_FF: Int, DEPTH: Int,
    PRED_PROJ_H: Int, SIG_PROJ: Int, SIG_KNOTS: Int,
    BATCH: Int, target: StaticString, PRED_DIM_HEAD: Int = 0,
    # Trailing encoder type with dims-derived default — pass LeWMEncoderCLS
    # for CLS/recipe WMs; existing callers (which omit it) stay unchanged.
    ENC: Module = LeWMEncoder[
        IN_CH, IMG, PATCH, (IMG // PATCH) * (IMG // PATCH), HIDDEN,
        ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H, ENC_FF_MULT,
    ],
](
    mut trainer: LeWMTrainer[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, BATCH, target,
        PRED_DIM_HEAD, ENC,
    ],
    pix_t: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC,
        origin=MutAnyOrigin, ...,
    ],
    expert_act_host: UnsafePointer[Scalar[DT], MutAnyOrigin],   # (B, T·ACT)
    n_shuffles: Int = 0,
    ctx: Optional[DeviceContext] = None,
    verbose: Bool = True,
) raises -> Tuple[Float64, Float64, Float64, Float64]:
    """Teacher-forced action-awareness via the legacy H6 diagnostic — needs
    NO action sampling, so it works for CONTINUOUS actions (PushT) too.
    Score = MSE(pred under some actions, tgt = encoded real next latents,
    fixed). Compare the real (expert) actions against BATCH-shuffled actions
    (cyclic row shifts, which break the action↔transition correspondence but
    keep the action distribution). An action-aware model scores expert below
    shuffled. Returns (expert, shuffled_mean, shuffled_min, frac_shuffled_worse)."""
    comptime HE = H * EMB
    comptime ACTIN = T * ACT
    comptime NP = BATCH * HE
    var act_host = alloc[Scalar[DT]](BATCH * ACTIN)
    var pred = alloc[Scalar[DT]](NP)
    var tgt = alloc[Scalar[DT]](NP)
    var act_dev: Optional[DeviceBuffer[DT]] = None
    comptime if target == "gpu":
        act_dev = ctx.value().enqueue_create_buffer[DT](BATCH * ACTIN)

    # one forward over the fixed pixels + current act_host → MSE(pred, tgt).
    # Inlined (no closure) to keep `trainer` borrows simple.
    var scores = List[Float64]()
    var worse = 0
    var expert: Float64 = 0.0
    var ns = n_shuffles if n_shuffles > 0 else BATCH - 1
    if ns > BATCH - 1:
        ns = BATCH - 1
    # round -1 = expert, rounds 0..ns-1 = cyclic shifts by (round+1)
    for rnd in range(-1, ns):
        if rnd < 0:
            for i in range(BATCH * ACTIN):
                act_host[i] = expert_act_host[i]
        else:
            var shift = rnd + 1
            for b in range(BATCH):
                var src = ((b + shift) % BATCH) * ACTIN
                for i in range(ACTIN):
                    act_host[b * ACTIN + i] = expert_act_host[src + i]
        comptime if target == "gpu":
            var c = ctx.value()
            c.enqueue_copy(act_dev.value(), act_host)
            var at = TileTensor(
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                    act_dev.value().unsafe_ptr()
                ),
                row_major[BATCH, ACTIN](),
            )
            trainer.forward_into(pix_t, at, pred, tgt)
        else:
            var at = TileTensor(act_host, row_major[BATCH, ACTIN]())
            trainer.forward_into(pix_t, at, pred, tgt)
        var s = _mse_latent(pred, tgt, NP)
        if rnd < 0:
            expert = s
        else:
            scores.append(s)
            if s > expert:
                worse += 1
    var shuffled_mean = _mean(scores)
    var shuffled_min = Float64(1e30)
    for i in range(len(scores)):
        if scores[i] < shuffled_min:
            shuffled_min = scores[i]
    var frac_worse = Float64(worse) / Float64(ns) if ns > 0 else 0.0

    if verbose:
        print("   expert        =", expert)
        print("   shuffled mean =", shuffled_mean, " min=", shuffled_min)
        print("   expert/shuffled_mean =", expert / shuffled_mean,
              "  frac_shuffled_worse =", frac_worse)
        print("   action-aware (expert < shuffled_min):",
              "yes" if expert < shuffled_min else "no")

    act_host.free(); pred.free(); tgt.free()
    _ = act_dev^
    return (expert, shuffled_mean, shuffled_min, frac_worse)


def lewm_action_awareness_eval[
    IN_CH: Int, IMG: Int, PATCH: Int, HIDDEN: Int, ENC_HEADS: Int,
    ENC_LAYERS: Int, EMB: Int, ENC_PROJ_H: Int, ENC_FF_MULT: Int,
    T: Int, ACT: Int, SMOOTHED: Int, AE_MLP: Int,
    H: Int, N_PREDS: Int, PRED_HEADS: Int, PRED_FF: Int, DEPTH: Int,
    PRED_PROJ_H: Int, SIG_PROJ: Int, SIG_KNOTS: Int,
    BATCH: Int, target: StaticString,
](
    mut trainer: LeWMTrainer[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, BATCH, target,
    ],
    pix_host: UnsafePointer[Scalar[DT], MutAnyOrigin],          # (B, T·IMG_DIM)
    expert_act_host: UnsafePointer[Scalar[DT], MutAnyOrigin],   # (B, T·ACT) one-hot
    num_random: Int = 16,
    cem_iters: Int = 3,
    cem_samples: Int = 32,
    cem_topk: Int = 8,
    cem_smoothing: Float64 = 0.5,
    ctx: Optional[DeviceContext] = None,
    verbose: Bool = True,
) raises -> Tuple[Float64, Float64, Float64, Float64]:
    """Score expert vs random vs CEM action plans by teacher-forced latent
    prediction MSE. Returns (expert, random_mean, random_min, cem). The
    §10.9 health signals: expert < random_min (action-awareness) and
    cem <= random_min (planner finds good actions). Lower is better."""
    comptime PIX = T * IN_CH * IMG * IMG
    comptime ACTIN = T * ACT
    comptime Scorer = LeWMTFScorer[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, BATCH, target,
    ]

    # Place the fixed context window on the target.
    var pix_target = pix_host
    var pix_dev: Optional[DeviceBuffer[DT]] = None
    comptime if target == "gpu":
        var c = ctx.value()
        var d = c.enqueue_create_buffer[DT](BATCH * PIX)
        c.enqueue_copy(d, pix_host)
        c.synchronize()
        pix_dev = d^
        pix_target = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            pix_dev.value().unsafe_ptr()
        )

    var scorer = Scorer(
        UnsafePointer(to=trainer).as_unsafe_any_origin(), pix_target, ctx=ctx
    )
    scorer.prime()

    # ── Expert leg: the real recorded actions (first H steps).
    var expert_plan = alloc[Scalar[DT]](BATCH * H * ACT)
    for b in range(BATCH):
        for t in range(H):
            for a in range(ACT):
                expert_plan[(b * H + t) * ACT + a] = expert_act_host[
                    (b * T + t) * ACT + a
                ]
    var expert_t = TileTensor(expert_plan, row_major[BATCH, H, ACT]())
    var expert = scorer.score_plan(expert_t)

    # ── Random shooter leg.
    var shooter = CategoricalRandomShooter[BATCH, ACT](
        horizon=H, num_samples=num_random
    )
    var rs_best = alloc[Scalar[DT]](BATCH * H * ACT)
    var random_min = shooter.optimize(
        scorer, rs_best.as_unsafe_any_origin(), verbose=False
    )
    var random_mean = _mean(shooter.sample_scores)
    rs_best.free()

    # ── CEM leg (refines from uniform; skipped if cem_iters == 0).
    var cem_score = expert
    if cem_iters > 0:
        var cem = CategoricalCEMOptimizer[BATCH, ACT](
            horizon=H,
            cem_iters=cem_iters,
            cem_samples=cem_samples,
            cem_topk=cem_topk,
            cem_smoothing=cem_smoothing,
        )
        var cem_best = alloc[Scalar[DT]](BATCH * H * ACT)
        cem_score = cem.optimize(
            scorer, cem_best.as_unsafe_any_origin(), verbose=False
        )
        cem_best.free()

    if verbose:
        print("   expert     =", expert)
        print("   random mean=", random_mean, " min=", random_min)
        print("   cem        =", cem_score)
        print("   expert/random_min =", expert / random_min,
              "  cem/random_min =", cem_score / random_min,
              "  cem/expert =", cem_score / expert)

    expert_plan.free()
    _ = scorer^
    _ = pix_dev^
    return (expert, random_mean, random_min, cem_score)
