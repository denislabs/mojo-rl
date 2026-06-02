"""MBPO training on HalfCheetah (GPU) via the new `MBPOAgent` facade.

GPU sibling of `mbpo_half_cheetah_nn2_agent.mojo` (CPU). Same deep_agents2
surface, but `train_target="gpu"`: the SAC sub-update + dynamics-ensemble
training + synthetic rollouts run on-device (the env is still stepped on
CPU — MBPO uses the single-env driver path). Because the GPU can afford the
legacy MBPO data regime, this example uses the LEGACY hyperparameters
(num_rollouts=100k, sac_updates=40, real_ratio=0.05, target_entropy=-3),
which is what gives the original `mbpo_half_cheetah_training_gpu.mojo` its
smooth convergence — and which measurably HURT on CPU (the CPU example keeps
a conservative regime; see its comments).

Carries the same convergence fixes as the CPU path:
  * Critic LayerNorm (REDQ/SR-SAC stability) — bounds Q under high UTD.
  * Dynamics input normalization (per-DYN_IN z-score, refit each model-train
    round) — essential for HalfCheetah's unbounded obs.
  * Elite ranking (holdout-scored members) after each dynamics-train round.

DynNet output layout: `2 * (1 + OBS_DIM)` = `[r_mean, r_logvar,
Δobs_mean[OBS_DIM], Δobs_logvar[OBS_DIM]]`. Logvar clamped to
`[LOGVAR_MIN, LOGVAR_MAX]`.

HalfCheetah (Physics3dEnv): 17D obs, 6D action, reward ≈ forward velocity −
0.1·||action||², no early termination.

NOTE: the GPU scaler-fit D2Hs the real buffer each model-train round; the
real buffer is capped at 200k (not the legacy's 1M) to keep that copy cheap.
A device-side reduction would let it grow — a future optimization.

Run:
    pixi run -e apple mojo run -I . examples/half_cheetah/mbpo_half_cheetah_nn2_gpu.mojo    # Apple
    pixi run -e nvidia mojo run -I . examples/half_cheetah/mbpo_half_cheetah_nn2_gpu.mojo   # NVIDIA
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.layer_norm import LayerNorm
from mojo_rl.nn2.primitives.elementwise import Elementwise
from mojo_rl.nn2.primitives.ops.swish_op import SwishOp
from mojo_rl.deep_agents2.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents2.mbpo import MBPOAgent
from mojo_rl.envs.half_cheetah import HalfCheetah, HalfCheetahConfig


# =============================================================================
# Architecture
# =============================================================================

comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACT_DIM = HalfCheetahConfig.ACTION_DIM  #  6
comptime HIDDEN = 256
comptime DYN_HIDDEN = 200
comptime BATCH = 128  # legacy MBPO batch size
comptime REPLAY_CAPACITY = 200_000  # capped (vs legacy 1M) for cheap scaler D2H
comptime SYNTH_CAPACITY = 400_000
comptime N_ENSEMBLE = 7
comptime NUM_ELITES = 5
# Legacy GPU value: lean hard on the (now-normalized) world model. Affordable
# here ONLY because num_rollouts=100k keeps the synthetic buffer FRESH and
# in-distribution. Do NOT copy this to the CPU example.
comptime REAL_RATIO_PCT = 5
comptime LOGVAR_MIN_F = -10.0
# Match legacy MBPO's dynamics uncertainty bound (max std ≈ 0.37). The prior
# −5.0 capped std at ≈0.08 → an over-confident model the policy could exploit
# (predicted 0.81/step vs 0.20/step real). Legacy's wider bound injects more
# rollout noise (regularizes the policy) and lets the model hedge on OOD
# rollout actions instead of emitting sharp optimistic means.
comptime LOGVAR_MAX_F = -2.0

comptime NUM_STEPS = 300_000  # MBPO needs ~10× fewer real steps than SAC
comptime PRINT_EVERY = 10_000
comptime DIAG_EVERY = 5_000
comptime CHECKPOINT_EVERY = 50_000

comptime CHECKPOINT_PATH = "mbpo_half_cheetah_nn2_gpu.ckpt"

# ─── A/B: entropy-temperature (alpha) ablation ───────────────────────────────
# The nn2-MBPO vs legacy overlay showed nn2's auto-tuned alpha equilibrates
# 2–4× BELOW legacy (0.035–0.086 vs ~0.12), correlating with ~4× slower mean_q
# growth + a climbing critic loss + a timid (low mean_abs_action) policy.
#   FIX_ALPHA = False → arm A: auto-tuned alpha (alpha_lr live, init 0.2).
#   FIX_ALPHA = True  → arm B: alpha PINNED at legacy's level (alpha_lr=0 so the
#                       ScalarAdam update is a no-op → alpha frozen at init).
# If arm B tracks legacy's mean_q / reward, alpha is confirmed as THE lever.
# Reverted to auto-α: the α A/B was REFUTED (fixed α=0.12 left reward flat at
# ~200 vs auto's ~210; the policy wasn't timid — mean_abs_action≈0.48). The
# real lever is the dynamics uncertainty bound (LOGVAR_MAX above).
comptime FIX_ALPHA = False
comptime FIXED_ALPHA: Scalar[DT] = 0.12  # legacy's stable equilibrium
comptime INIT_ALPHA: Scalar[DT] = FIXED_ALPHA if FIX_ALPHA else 0.2
comptime ALPHA_LR: Scalar[DT] = 0.0 if FIX_ALPHA else 3e-4
comptime RUN_NAME = (
    "MBPO HalfCheetah NN2 (GPU) — fixed alpha=0.12"
    if FIX_ALPHA
    else "MBPO HalfCheetah NN2 (GPU) — auto alpha, logvar_max=-2"
)


comptime ActorNet = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
]
# Critic with pre-activation LayerNorm (REDQ/SR-SAC stability fix; mirrors
# the legacy MBPO critic). Pattern: Linear → LayerNorm → ReLU, repeated.
# Bounds the critic's feature magnitudes so Q can't diverge under high-UTD
# synthetic-batch pressure (the Q-explosion we diagnosed on this surface).
comptime CriticNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN],
    LayerNorm[HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    LayerNorm[HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]
# Dynamics output = 2 * (1 + OBS_DIM) = 2 * 18 = 36
# Layout: [r_mean, r_logvar, Δobs_mean[OBS_DIM], Δobs_logvar[OBS_DIM]]
comptime DynNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, DYN_HIDDEN],
    Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, DYN_HIDDEN],
    Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, DYN_HIDDEN],
    Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, DYN_HIDDEN],
    Elementwise[DYN_HIDDEN, SwishOp],
    Linear[DYN_HIDDEN, 2 * (1 + OBS_DIM)],
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("MBPO (deep_agents2) — HalfCheetah GPU (legacy hyperparams)")
    print("=" * 70)
    print("  OBS_DIM            =", OBS_DIM)
    print("  ACT_DIM            =", ACT_DIM)
    print("  HIDDEN (SAC)       =", HIDDEN)
    print("  DYN_HIDDEN         =", DYN_HIDDEN)
    print("  BATCH              =", BATCH)
    print("  REPLAY_CAPACITY    =", REPLAY_CAPACITY)
    print("  SYNTH_CAPACITY     =", SYNTH_CAPACITY)
    print("  N_ENSEMBLE/ELITES  =", N_ENSEMBLE, "/", NUM_ELITES)
    print("  REAL_RATIO_PCT     =", REAL_RATIO_PCT)
    print("  NUM_STEPS          =", NUM_STEPS)
    print("=" * 70)

    with DeviceContext() as ctx:
        # ─── Logger (remote) ─────────────────────────────────────────────
        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name=RUN_NAME,
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("algorithm", "MBPO")
        logger.set_config("env", "HalfCheetah")
        logger.set_config("target", "gpu")
        logger.set_config("alpha_mode", "fixed_0.12" if FIX_ALPHA else "auto")
        logger.set_config("logvar_max", String(LOGVAR_MAX_F))
        logger.set_config("hidden", String(HIDDEN))
        logger.set_config("dyn_hidden", String(DYN_HIDDEN))
        logger.set_config("batch", String(BATCH))
        logger.set_config("ensemble", String(N_ENSEMBLE))
        logger.set_config("real_ratio_pct", String(REAL_RATIO_PCT))

        var logger_ptr = UnsafePointer(to=logger)

        # ─── Agent + env ─────────────────────────────────────────────────
        var agent = MBPOAgent[
            "gpu",
            ActorNet,
            CriticNet,
            DynNet,
            OBS_DIM,
            ACT_DIM,
            BATCH,
            REPLAY_CAPACITY,
            SYNTH_CAPACITY,
            N_ENSEMBLE,
            NUM_ELITES,
            REAL_RATIO_PCT,
            LOGVAR_MIN_F,
            LOGVAR_MAX_F,
        ](
            ctx=ctx,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=ALPHA_LR,  # A/B: 0.0 freezes alpha (arm B), 3e-4 = auto
            model_lr=1e-3,
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            init_alpha=INIT_ALPHA,  # A/B: 0.12 (arm B) vs 0.2 (arm A)
            target_entropy=-3.0,  # legacy MBPO value
            learning_starts=5_000,  # legacy warmup
            window_size=100,
            initial_episode_fill=0.0,
            # Legacy GPU cadences — affordable on-device; the large fresh
            # synthetic buffer is what keeps the high-UTD critic stable
            # (together with the LayerNorm critic above).
            model_train_freq=250,
            dyn_epochs_per_round=4,
            rollout_length=1,
            num_rollouts_per_step=100_000,
            sac_updates_per_step=40,
            dyn_batch_size=256,
        )
        var env = HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False]()

        # ─── Train ───────────────────────────────────────────────────────
        var t_start = perf_counter_ns()
        _ = agent.train_single[
            HalfCheetah[DT, TERMINATE_ON_UNHEALTHY=False],
            L=RemoteLogger,
        ](
            env,
            NUM_STEPS,
            print_every=PRINT_EVERY,
            verbose=True,
            logger=logger_ptr,
            diag_every=DIAG_EVERY,
            checkpoint_path=CHECKPOINT_PATH,
            checkpoint_every=CHECKPOINT_EVERY,
        )
        var elapsed_s = Float64(perf_counter_ns() - t_start) / 1e9
        logger.close()
        _ = logger  # lifetime extender for logger_ptr

        # ─── Summary ─────────────────────────────────────────────────────
        print("=" * 70)
        print("Training complete")
        print("  total env_steps        =", NUM_STEPS)
        print("  elapsed                =", elapsed_s, "s")
        print("  mean ep return (last 100) =", agent.mean_return())
        print("  episodes completed     =", agent.ep_count())
        print("  remote points sent     =", logger.total_logged())
        print("=" * 70)

        var final_avg = Float64(agent.mean_return())
        if final_avg > 4000.0:
            print("EXCELLENT — running fast (mean > 4000).")
        elif final_avg > 1000.0:
            print("STRONG — learned locomotion (mean > 1000).")
        elif final_avg > 100.0:
            print("PROGRESS — early locomotion (mean > 100).")
        elif final_avg > 0.0:
            print("LEARNING — positive return (mean > 0).")
        else:
            print("EARLY — still exploring (mean < 0).")
        print("=" * 70)
