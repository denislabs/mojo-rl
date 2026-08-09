"""LeWM continuous-action MPC eval (PushT — paper's main benchmark).

The continuous-action analogue of `lewm_mpc_eval`: same autoregressive
latent rollout (`LeWMMPCScorer`, action-representation agnostic), but the
plan is optimized with the **Gaussian** `ContinuousCEMOptimizer` (paper:
300 samples, 30 iters, top-30 elites, init variance 1, horizon 5) and the
baseline is a Gaussian `ContinuousRandomShooter`.

Encodes a window's start (frame 0) + goal (frame T-1) latents, then reports
expert vs random vs CEM scores (MSE-to-goal-latent). Gate: cem < random_min
(the planner finds action sequences that drive the latent toward the goal
better than random) on a non-collapsed model; expert is the dataset-action
reference floor.
"""

from std.memory import alloc
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.planners.trajectory import (
    ContinuousCEMOptimizer,
    ContinuousRandomShooter,
)
from .trainer import LeWMTrainer
from .predict_graph import LeWMPredictor
from .mpc import LeWMMPCScorer


def _meanf(v: List[Float64]) -> Float64:
    var s: Float64 = 0.0
    for i in range(len(v)):
        s += v[i]
    return s / Float64(len(v)) if len(v) > 0 else 0.0


def lewm_mpc_eval_continuous[
    IN_CH: Int, IMG: Int, PATCH: Int, HIDDEN: Int, ENC_HEADS: Int,
    ENC_LAYERS: Int, EMB: Int, ENC_PROJ_H: Int, ENC_FF_MULT: Int,
    T: Int, ACT: Int, SMOOTHED: Int, AE_MLP: Int,
    H: Int, N_PREDS: Int, PRED_HEADS: Int, PRED_FF: Int, DEPTH: Int,
    PRED_PROJ_H: Int, SIG_PROJ: Int, SIG_KNOTS: Int,
    BATCH: Int, MPC_HORIZON: Int, target: StaticString,
    PRED_DIM_HEAD: Int = 0,
](
    mut trainer: LeWMTrainer[
        IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
        ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS,
        PRED_FF, DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, BATCH, target,
        PRED_DIM_HEAD,
    ],
    pix_t: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC,
        origin=MutAnyOrigin, ...,
    ],
    act_t: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC,
        origin=MutAnyOrigin, ...,
    ],
    expert_act_host: Pointer[Scalar[DT], MutAnyOrigin],   # (B, T·ACT)
    num_random: Int = 300,
    cem_iters: Int = 30,
    cem_samples: Int = 300,
    cem_topk: Int = 30,
    init_std: Float64 = 1.0,
    ctx: Optional[DeviceContext] = None,
    verbose: Bool = True,
) raises -> Tuple[Float64, Float64, Float64, Float64]:
    """Continuous-action autoregressive MPC eval. `pix_t`/`act_t` are a
    target-resident window encoding start (frame 0) + goal (frame T-1)
    latents. Returns (expert, random_mean, random_min, cem)."""
    comptime TE = T * EMB
    comptime BE = BATCH * EMB
    comptime NEEDED = H + MPC_HORIZON - 1
    comptime Scorer = LeWMMPCScorer[
        EMB, T, ACT, SMOOTHED, AE_MLP, H, PRED_HEADS, PRED_FF, DEPTH,
        PRED_PROJ_H, BATCH, MPC_HORIZON, target, PRED_DIM_HEAD,
    ]
    comptime Predictor = LeWMPredictor[
        EMB, T, ACT, SMOOTHED, AE_MLP, H, PRED_HEADS, PRED_FF, DEPTH,
        PRED_PROJ_H, BATCH, target, PRED_DIM_HEAD,
    ]

    # encode the window → emb node → slice start (frame 0) + goal (frame T-1)
    var pred_scratch = alloc[Scalar[DT]](BATCH * H * EMB)
    var tgt_scratch = alloc[Scalar[DT]](BATCH * H * EMB)
    trainer.forward_into(pix_t, act_t, pred_scratch, tgt_scratch)
    var emb_host = alloc[Scalar[DT]](BATCH * TE)
    trainer.read_node_into["emb"](emb_host, BATCH * TE)
    var start_host = alloc[Scalar[DT]](BE)
    var goal_host = alloc[Scalar[DT]](BE)
    for b in range(BATCH):
        for d in range(EMB):
            start_host[unsafe_offset=b * EMB + d] = emb_host[unsafe_offset=b * TE + d]
            goal_host[unsafe_offset=b * EMB + d] = emb_host[unsafe_offset=b * TE + (T - 1) * EMB + d]

    var pred_net = Predictor.make(ctx=ctx)
    pred_net.sync_from_named(trainer.export_named_params())
    var scorer = Scorer(pred_net^, ctx=ctx)
    scorer.set_start_goal(start_host, goal_host)

    # expert plan: first NEEDED recorded (continuous) action vectors.
    var expert_plan = alloc[Scalar[DT]](BATCH * NEEDED * ACT)
    for b in range(BATCH):
        for t in range(NEEDED):
            for a in range(ACT):
                expert_plan[unsafe_offset=(b * NEEDED + t) * ACT + a] = expert_act_host[unsafe_offset=
                    (b * T + t) * ACT + a
                ]
    var expert_t = TileTensor(expert_plan, row_major[BATCH, NEEDED, ACT]())
    var expert = scorer.score_plan(expert_t)

    # Gaussian random shooter baseline.
    var shooter = ContinuousRandomShooter[BATCH, ACT](
        horizon=NEEDED, num_samples=num_random, init_std=init_std
    )
    var rs_best = alloc[Scalar[DT]](BATCH * NEEDED * ACT)
    var random_min = shooter.optimize(
        scorer, rs_best.as_unsafe_any_origin(), verbose=False
    )
    var random_mean = _meanf(shooter.sample_scores)
    rs_best.unsafe_free()

    # Gaussian CEM (paper config).
    var cem_score = expert
    if cem_iters > 0:
        var cem = ContinuousCEMOptimizer[BATCH, ACT](
            horizon=NEEDED, cem_iters=cem_iters, cem_samples=cem_samples,
            cem_topk=cem_topk, init_std=init_std,
        )
        var cem_best = alloc[Scalar[DT]](BATCH * NEEDED * ACT)
        cem_score = cem.optimize(
            scorer, cem_best.as_unsafe_any_origin(), verbose=False
        )
        cem_best.unsafe_free()

    if verbose:
        print("   [continuous MPC horizon=", MPC_HORIZON, "] expert=", expert)
        print("   random mean=", random_mean, " min=", random_min)
        print("   cem=", cem_score)
        print("   expert/random_min=", expert / random_min,
              "  cem/random_min=", cem_score / random_min)

    pred_scratch.unsafe_free(); tgt_scratch.unsafe_free(); emb_host.unsafe_free()
    start_host.unsafe_free(); goal_host.unsafe_free(); expert_plan.unsafe_free()
    _ = scorer^
    return (expert, random_mean, random_min, cem_score)
