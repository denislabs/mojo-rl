"""LeWM autoregressive MPC — latent-rollout planning eval (Phase F 2/2).

The full §10.9 planning eval: roll the predictor forward in LATENT space
under candidate actions and score the rolled-out latent against an encoded
goal. The encoder runs once (start + goal latents); the predictor
(`LeWMPredictor`, weights name-synced from the trainer) then drives the
rollout via the ported window-slide kernels — no pixels in the loop.

  emb_seq[:, 0:H] = replicate(emb_start)                 (init context)
  for k in 0..horizon-1:
      latent_ctx = emb_seq[:, k:k+H]                     (slide)
      actions    = plan[:, k:k+H] (zero-padded to T)     (slide)
      pred       = predictor(latent_ctx, actions)        (forward)
      emb_seq[:, k+H] = pred[:, H-1]                      (append)
  score = MSE(emb_seq[:, H+horizon-1], emb_goal)

`LeWMMPCScorer` implements `ScorePlanCallback`, so the shared CEM /
random-shooter optimize action plans of length NEEDED = H+horizon-1.
`lewm_mpc_eval` encodes a window's start/goal and reports expert vs
random vs CEM. Gate (§10.9): cem < random_min, expert < random_min on a
non-collapsed model.
"""

from std.memory import alloc
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout, row_major

from mojo_rl.nn.constants import DT
comptime dtype = DT                              # float32 (legacy nn.constants.dtype)
from mojo_rl.planners.trajectory import (
    CategoricalRandomShooter,
    CategoricalCEMOptimizer,
)
from mojo_rl.planners.trajectory.score_callback import ScorePlanCallback
from .trainer import LeWMTrainer
from .predict_graph import LeWMPredictor
from .mpc_kernels import (
    mpc_replicate_start,
    mpc_slide_latent_ctx,
    mpc_slide_actions,
    mpc_store_pred_last,
    mpc_score,
)


def _tp[
    target: StaticString,
    ho: MutOrigin = MutAnyOrigin,
](
    h: UnsafePointer[Scalar[DT], ho],
    d: Optional[DeviceBuffer[DT]],
) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    comptime if target == "cpu":
        return h.as_unsafe_any_origin()
    else:
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            d.value().unsafe_ptr()
        )


struct LeWMMPCScorer[
    EMB: Int, T: Int, ACT: Int, SMOOTHED: Int, AE_MLP: Int,
    H: Int, PRED_HEADS: Int, PRED_FF: Int, DEPTH: Int, PRED_PROJ_H: Int,
    BATCH: Int, MPC_HORIZON: Int, target: StaticString, PRED_DIM_HEAD: Int = 0,
](ScorePlanCallback, Movable):
    # PRED_DIM_HEAD (default 0 ⇒ EMB/PRED_HEADS) added last so the Pong
    # categorical eval is unchanged; >0 matches a paper-width WM's expanded
    # predictor attention so the name-synced weights align.
    comptime ROLL_T = Self.H + Self.MPC_HORIZON
    comptime NEEDED = Self.H + Self.MPC_HORIZON - 1
    comptime HE = Self.H * Self.EMB
    comptime ACTIN = Self.T * Self.ACT
    comptime GOAL_POS = Self.H + Self.MPC_HORIZON - 1
    comptime Predictor = LeWMPredictor[
        Self.EMB, Self.T, Self.ACT, Self.SMOOTHED, Self.AE_MLP,
        Self.H, Self.PRED_HEADS, Self.PRED_FF, Self.DEPTH, Self.PRED_PROJ_H,
        Self.BATCH, Self.target, Self.PRED_DIM_HEAD,
    ]

    var pred_net: Self.Predictor
    var ctx: Optional[DeviceContext]
    # host buffers (always) + device mirrors (gpu)
    var emb_start_h: List[Scalar[DT]]
    var emb_goal_h: List[Scalar[DT]]
    var emb_seq_h: List[Scalar[DT]]
    var latent_ctx_h: List[Scalar[DT]]
    var actions_buf_h: List[Scalar[DT]]
    var pred_h: List[Scalar[DT]]
    var plan_h: List[Scalar[DT]]
    var emb_start_d: Optional[DeviceBuffer[DT]]
    var emb_goal_d: Optional[DeviceBuffer[DT]]
    var emb_seq_d: Optional[DeviceBuffer[DT]]
    var latent_ctx_d: Optional[DeviceBuffer[DT]]
    var actions_buf_d: Optional[DeviceBuffer[DT]]
    var pred_d: Optional[DeviceBuffer[DT]]
    var plan_d: Optional[DeviceBuffer[DT]]

    def __init__(out self, var pred_net: Self.Predictor,
                 ctx: Optional[DeviceContext] = None) raises:
        self.pred_net = pred_net^
        self.ctx = ctx
        comptime BE = Self.BATCH * Self.EMB
        self.emb_start_h = List[Scalar[DT]](length=BE, fill=Scalar[DT](0))
        self.emb_goal_h = List[Scalar[DT]](length=BE, fill=Scalar[DT](0))
        self.emb_seq_h = List[Scalar[DT]](
            length=Self.BATCH * Self.ROLL_T * Self.EMB, fill=Scalar[DT](0)
        )
        self.latent_ctx_h = List[Scalar[DT]](
            length=Self.BATCH * Self.HE, fill=Scalar[DT](0)
        )
        self.actions_buf_h = List[Scalar[DT]](
            length=Self.BATCH * Self.ACTIN, fill=Scalar[DT](0)
        )
        self.pred_h = List[Scalar[DT]](
            length=Self.BATCH * Self.HE, fill=Scalar[DT](0)
        )
        self.plan_h = List[Scalar[DT]](
            length=Self.BATCH * Self.NEEDED * Self.ACT, fill=Scalar[DT](0)
        )
        self.emb_start_d = None; self.emb_goal_d = None
        self.emb_seq_d = None; self.latent_ctx_d = None
        self.actions_buf_d = None; self.pred_d = None; self.plan_d = None
        comptime if Self.target == "gpu":
            var c = ctx.value()
            self.emb_start_d = c.enqueue_create_buffer[DT](BE)
            self.emb_goal_d = c.enqueue_create_buffer[DT](BE)
            self.emb_seq_d = c.enqueue_create_buffer[DT](
                Self.BATCH * Self.ROLL_T * Self.EMB
            )
            self.latent_ctx_d = c.enqueue_create_buffer[DT](Self.BATCH * Self.HE)
            self.actions_buf_d = c.enqueue_create_buffer[DT](
                Self.BATCH * Self.ACTIN
            )
            self.pred_d = c.enqueue_create_buffer[DT](Self.BATCH * Self.HE)
            self.plan_d = c.enqueue_create_buffer[DT](
                Self.BATCH * Self.NEEDED * Self.ACT
            )

    def set_start_goal[
        so: MutOrigin = MutAnyOrigin,
        go: MutOrigin = MutAnyOrigin,
    ](
        mut self,
        start_src: UnsafePointer[Scalar[DT], so],
        goal_src: UnsafePointer[Scalar[DT], go],
    ) raises:
        """Set the fixed encoded start + goal latents (each (B, EMB))."""
        comptime BE = Self.BATCH * Self.EMB
        for i in range(BE):
            self.emb_start_h[i] = start_src[i]
            self.emb_goal_h[i] = goal_src[i]
        comptime if Self.target == "gpu":
            var c = self.ctx.value()
            c.enqueue_copy(
                self.emb_start_d.value(), self.emb_start_h.unsafe_ptr()
            )
            c.enqueue_copy(
                self.emb_goal_d.value(), self.emb_goal_h.unsafe_ptr()
            )
            c.synchronize()

    def score_plan[
        L: TensorLayout
    ](
        mut self,
        action_plan: TileTensor[dtype, L, MutAnyOrigin],
    ) raises -> Float64:
        comptime assert action_plan.flat_rank == 3, (
            "score_plan: action_plan must be (BATCH, NEEDED, ACT)"
        )
        # stage plan → host (then upload on gpu)
        for b in range(Self.BATCH):
            for t in range(Self.NEEDED):
                for a in range(Self.ACT):
                    self.plan_h[(b * Self.NEEDED + t) * Self.ACT + a] = rebind[
                        Scalar[DT]
                    ](action_plan[b, t, a])
        comptime if Self.target == "gpu":
            self.ctx.value().enqueue_copy(
                self.plan_d.value(), self.plan_h.unsafe_ptr()
            )

        var es = _tp[Self.target](self.emb_seq_h.unsafe_ptr(), self.emb_seq_d)
        var start = _tp[Self.target](
            self.emb_start_h.unsafe_ptr(), self.emb_start_d
        )
        var goal = _tp[Self.target](
            self.emb_goal_h.unsafe_ptr(), self.emb_goal_d
        )
        var lc = _tp[Self.target](
            self.latent_ctx_h.unsafe_ptr(), self.latent_ctx_d
        )
        var ab = _tp[Self.target](
            self.actions_buf_h.unsafe_ptr(), self.actions_buf_d
        )
        var pr = _tp[Self.target](self.pred_h.unsafe_ptr(), self.pred_d)
        var plan = _tp[Self.target](self.plan_h.unsafe_ptr(), self.plan_d)

        mpc_replicate_start[
            Self.target, Self.BATCH, Self.H, Self.EMB, Self.ROLL_T
        ](start, es, ctx=self.ctx)

        for k in range(Self.MPC_HORIZON):
            mpc_slide_latent_ctx[
                Self.target, Self.BATCH, Self.H, Self.EMB, Self.ROLL_T
            ](es, lc, k, ctx=self.ctx)
            mpc_slide_actions[
                Self.target, Self.BATCH, Self.T, Self.H, Self.ACT, Self.NEEDED
            ](plan, ab, k, ctx=self.ctx)
            var lc_t = TileTensor(lc, row_major[Self.BATCH, Self.HE]())
            var ab_t = TileTensor(ab, row_major[Self.BATCH, Self.ACTIN]())
            var pr_t = TileTensor(pr, row_major[Self.BATCH, Self.HE]())
            self.pred_net.forward(lc_t, ab_t, pr_t)
            mpc_store_pred_last[
                Self.target, Self.BATCH, Self.H, Self.EMB, Self.ROLL_T
            ](pr, es, k, ctx=self.ctx)

        return mpc_score[Self.target, Self.BATCH, Self.EMB, Self.ROLL_T](
            es, goal, Self.GOAL_POS, ctx=self.ctx
        )


def _mean(v: List[Float64]) -> Float64:
    var s: Float64 = 0.0
    for i in range(len(v)):
        s += v[i]
    return s / Float64(len(v)) if len(v) > 0 else 0.0


def lewm_mpc_eval[
    IN_CH: Int, IMG: Int, PATCH: Int, HIDDEN: Int, ENC_HEADS: Int,
    ENC_LAYERS: Int, EMB: Int, ENC_PROJ_H: Int, ENC_FF_MULT: Int,
    T: Int, ACT: Int, SMOOTHED: Int, AE_MLP: Int,
    H: Int, N_PREDS: Int, PRED_HEADS: Int, PRED_FF: Int, DEPTH: Int,
    PRED_PROJ_H: Int, SIG_PROJ: Int, SIG_KNOTS: Int,
    BATCH: Int, MPC_HORIZON: Int, target: StaticString,
](
    mut trainer: LeWMTrainer[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, BATCH, target,
    ],
    pix_t: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC,
        origin=MutAnyOrigin, ...,
    ],
    act_t: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC,
        origin=MutAnyOrigin, ...,
    ],
    expert_act_host: UnsafePointer[Scalar[DT], MutAnyOrigin],   # (B, T·ACT)
    num_random: Int = 16,
    cem_iters: Int = 3,
    cem_samples: Int = 32,
    cem_topk: Int = 8,
    cem_smoothing: Float64 = 0.5,
    ctx: Optional[DeviceContext] = None,
    verbose: Bool = True,
) raises -> Tuple[Float64, Float64, Float64, Float64]:
    """Autoregressive MPC eval. `pix_t`/`act_t` are a target-resident window
    used to encode the start (frame 0) + goal (frame T-1) latents. Returns
    (expert, random_mean, random_min, cem)."""
    comptime TE = T * EMB
    comptime BE = BATCH * EMB
    comptime NEEDED = H + MPC_HORIZON - 1
    comptime Scorer = LeWMMPCScorer[
        EMB, T, ACT, SMOOTHED, AE_MLP, H, PRED_HEADS, PRED_FF, DEPTH,
        PRED_PROJ_H, BATCH, MPC_HORIZON, target,
    ]
    comptime Predictor = LeWMPredictor[
        EMB, T, ACT, SMOOTHED, AE_MLP, H, PRED_HEADS, PRED_FF, DEPTH,
        PRED_PROJ_H, BATCH, target,
    ]

    # ── encode the window: read the emb node, slice start + goal latents.
    var pred_scratch = alloc[Scalar[DT]](BATCH * H * EMB)
    var tgt_scratch = alloc[Scalar[DT]](BATCH * H * EMB)
    trainer.forward_into(pix_t, act_t, pred_scratch, tgt_scratch)
    var emb_host = alloc[Scalar[DT]](BATCH * TE)
    trainer.read_node_into["emb"](emb_host, BATCH * TE)
    var start_host = alloc[Scalar[DT]](BE)
    var goal_host = alloc[Scalar[DT]](BE)
    for b in range(BATCH):
        for d in range(EMB):
            start_host[b * EMB + d] = emb_host[b * TE + d]
            goal_host[b * EMB + d] = emb_host[b * TE + (T - 1) * EMB + d]

    # ── predictor: name-synced from the trainer.
    var pred_net = Predictor.make(ctx=ctx)
    pred_net.sync_from_named(trainer.export_named_params())
    var scorer = Scorer(pred_net^, ctx=ctx)
    scorer.set_start_goal(start_host, goal_host)

    # ── expert plan: first NEEDED recorded actions.
    var expert_plan = alloc[Scalar[DT]](BATCH * NEEDED * ACT)
    for b in range(BATCH):
        for t in range(NEEDED):
            for a in range(ACT):
                expert_plan[(b * NEEDED + t) * ACT + a] = expert_act_host[
                    (b * T + t) * ACT + a
                ]
    var expert_t = TileTensor(expert_plan, row_major[BATCH, NEEDED, ACT]())
    var expert = scorer.score_plan(expert_t)

    # ── random shooter.
    var shooter = CategoricalRandomShooter[BATCH, ACT](
        horizon=NEEDED, num_samples=num_random
    )
    var rs_best = alloc[Scalar[DT]](BATCH * NEEDED * ACT)
    var random_min = shooter.optimize(
        scorer, rs_best.as_unsafe_any_origin(), verbose=False
    )
    var random_mean = _mean(shooter.sample_scores)
    rs_best.free()

    # ── CEM.
    var cem_score = expert
    if cem_iters > 0:
        var cem = CategoricalCEMOptimizer[BATCH, ACT](
            horizon=NEEDED, cem_iters=cem_iters, cem_samples=cem_samples,
            cem_topk=cem_topk, cem_smoothing=cem_smoothing,
        )
        var cem_best = alloc[Scalar[DT]](BATCH * NEEDED * ACT)
        cem_score = cem.optimize(
            scorer, cem_best.as_unsafe_any_origin(), verbose=False
        )
        cem_best.free()

    if verbose:
        print("   [MPC horizon=", MPC_HORIZON, "] expert=", expert)
        print("   random mean=", random_mean, " min=", random_min)
        print("   cem=", cem_score)
        print("   expert/random_min=", expert / random_min,
              "  cem/random_min=", cem_score / random_min)

    pred_scratch.free(); tgt_scratch.free(); emb_host.free()
    start_host.free(); goal_host.free(); expert_plan.free()
    _ = scorer^
    return (expert, random_mean, random_min, cem_score)
