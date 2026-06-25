"""MuZero batched GPU training on Pong — Pixel Observations (deep_agents).

The MuZero counterpart of `rainbow_pong_pixel_training_gpu.mojo`: learns the
model (h/g/f) from raw pixels (4×84×84 stacked grayscale frames) and plans with
Gumbel-MCTS, stepping `N_ENVS` `PongPixelEnv` environments in parallel on the GPU
via `BatchedGpuDiscreteEnv` while the three nets train on the same device.

Pieces (Phases 0–2 of `docs/MUZERO_PIXEL_PONG_PLAN.md`):
  * `MuZeroCNNConfig` — Nature-CNN representation (`MZRepNetCNN`, 84→20→9→7) →
    latent; the latent-space dynamics/prediction nets are the same as the MLP
    config (only the obs encoder differs).
  * `MuZeroBatchedAgent` — facade over the batched device-replay self-play
    driver: N parallel GPU envs + a single batched Gumbel search over
    `[N_ENVS, OBS]` (the rep CNN runs at batch=N_ENVS at the root) + a
    device-resident `GPUMCTSSequenceReplay`. Obs are stored device→device from
    `env.obs_ptr()` and the training obs slab is gathered device→device into the
    train step, so no full `[N_ENVS, OBS]` pixel observation crosses the bus on
    the collection path.
  * `OBS_STORE_DT = DType.uint8` — the obs ring quantizes `round(x·255)` and
    dequantizes `k/255`, bit-lossless for the arcade pixel pipeline and 4× the
    resident steps of a float ring.

Tuning note (Phase 3): the value support [V_MIN, V_MAX] is in MuZero h-space and
must bracket the *discounted* return, not the raw ±21 episode score — the same
lever that made Rainbow converge on Pong ([-2, 2] not [-21, 21]). `N_ENVS` is the
primary throughput/cost knob (Rainbow used 64; MuZero's per-env tree search is
heavier, so start smaller and scale up on large-VRAM cards).

Run:
    pixi run -e apple  mojo run -I . examples/arcade_games/muzero_pong_pixel_training_gpu.mojo  # compile/smoke
    pixi run -e nvidia mojo run -I . examples/arcade_games/muzero_pong_pixel_training_gpu.mojo  # training
"""

from std.memory import UnsafePointer
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.muzero import MuZeroCNNConfig, MuZeroBatchedAgent
from mojo_rl.deep_agents.training import BatchedGpuDiscreteEnv
from mojo_rl.envs.arcade_games.pong import PongPixelEnv


# =============================================================================
# Constants  (Phase 3 tunes these for convergence)
# =============================================================================

# Fast-validate switch. False = full convergence run (paper-faithful budget).
# True = screening profile for quickly checking whether a config change lifts
# the return curve (~1.6–2× faster iterations): fewer sims, lighter UTD, and
# reanalyze only every 4th iter. Use True to A/B ideas in ~30 min, then flip
# back to False for the real run. CAP/N_ENVS are unchanged (memory-only / not a
# speed lever — see notes below).
comptime FAST_VALIDATE = False

comptime FRAMES = 4
comptime ACT = 3  # NOOP, UP, DOWN
comptime LATENT = 128
comptime HIDDEN = 512  # Nature-CNN projection width
comptime BINS = 51  # categorical reward/value support bins

# Parallel envs — the primary throughput knob. Each env owns a pixel
# render/frame-stack workspace AND a root tree, so MuZero scales this lower than
# Rainbow's 64. Raise on large-VRAM cards.
comptime N_ENVS = 32

# Gumbel search budget. Fast-validate trims sims (still plenty for 3 actions).
comptime NUM_SIMS = 32 if FAST_VALIDATE else 50
comptime MAX_NODES = 128
comptime MAX_K = 3  # Gumbel root candidates (= ACT for Pong)

# Device replay (Phase 2): uint8 obs ring on the GPU → 28224 bytes/step, no
# per-step obs D2H. Constraint: CAP must be a multiple of N_ENVS AND exceed
# N_ENVS·MAX_EP_STEPS (else an in-flight episode self-overwrites). With N_ENVS=32
# and MAX_EP_STEPS=2000 → CAP ≥ 64000. CAP is memory-only (no per-iteration cost)
# so we size it generously on a 32 GB card: 262144 = 8192·32 → 262144·28224 B ≈
# 7.4 GB device, leaving ~24 GB free. Bigger ring = more target diversity and
# steadier PER than the old 64000 (1.8 GB), which rode its cap and cycled fast.
comptime CAP = 262_144
comptime OBS_STORE_DT = DType.uint8

# Unroll / training.
comptime B = 256  # unroll batch (windows) — bigger batch, fewer steps
comptime K = 5  # BPTT unroll length
comptime N = 5  # n-step value bootstrap horizon
# Gradient steps per iteration. UTD 1:1 (= N_ENVS) — one grad step per env step.
# An earlier experiment at train_per_iter=4 (UTD 0.125) was faster but
# UNDERTRAINED on Pong, so the convergence run keeps the UTD-1:1 default; sample
# efficiency comes from the reanalyze coverage + target-net stabiliser below,
# not from trimming gradient steps. Fast-validate halves UTD (16) purely to
# screen configs faster — do NOT trust it for final convergence (see above).
comptime TRAIN_PER_ITER = 16 if FAST_VALIDATE else N_ENVS

# Overfitting guards from the muzero-general Atari recipe (games/atari.py): an
# unregularized model trained to its full update budget at constant LR overfits
# late — greedy eval regresses while search-augmented behavior masks it. Both are
# OFF by default in the agent; we turn them on here.
#   * weight_decay 1e-4 — decoupled L2 on weight Params (biases/norms excluded).
#   * exponential LR decay lr = lr_init·RATE^(grad_step/STEPS); reference uses
#     0.1 over ~⅓ of the total grad-step budget (= NUM_ITERS·TRAIN_PER_ITER).
comptime WEIGHT_DECAY = Scalar[DT](1e-4)
comptime LR_DECAY_RATE = Scalar[DT](0.1)
comptime LR_DECAY_STEPS = (NUM_ITERS * TRAIN_PER_ITER) // 3

# Reanalyze cadence: every iter (paper-faithful, max fresh targets) for the real
# run; every 4th iter under fast-validate (reanalyze fires `reanalyze_batch //
# N_ENVS` = 8 batched searches per trigger, so this is a large wall-clock lever).
comptime REANALYZE_EVERY = 4 if FAST_VALIDATE else 1
comptime LR = Scalar[DT](1e-4)

# Value support in MuZero h-space — bracket the DISCOUNTED return (γ=0.997 over
# sparse ±1 rewards), NOT the raw ±21 score. The Rainbow-on-Pong lever.
comptime V_MIN = Scalar[DT](-2.0)
comptime V_MAX = Scalar[DT](2.0)
comptime GAMMA = Scalar[DT](0.997)

# 5M env steps ≈ iterations · N_ENVS. iterations = 5_000_000 / 32 ≈ 156_250.
comptime NUM_ITERS = 156_250
comptime LEARNING_STARTS = 2_000  # stored steps before training
# Bounded so CAP ≥ N_ENVS·MAX_EP_STEPS holds (device ring); long Pong games are
# truncated (n-step bootstrap), which is fine for MuZero.
comptime MAX_EP_STEPS = 2_000


comptime Cfg = MuZeroCNNConfig[FRAMES, ACT, LATENT, HIDDEN, BINS]
comptime OBS = Cfg.OBS  # 4*84*84 = 28224
comptime PongPixelBatched = BatchedGpuDiscreteEnv[
    PongPixelEnv[DT], N_ENVS, OBS, 1
]
comptime Agent = MuZeroBatchedAgent[
    PongPixelBatched,
    Cfg.Rep,
    Cfg.Dyn,
    Cfg.Pred,
    N_ENVS,
    OBS,
    ACT,
    LATENT,
    BINS,
    NUM_SIMS,
    MAX_NODES,
    MAX_K,
    CAP,
    B,
    K,
    N,
    OBS_STORE_DT=OBS_STORE_DT,
]


def main() raises:
    print("=" * 70)
    print("MuZero batched GPU training on Pong — Pixel (deep_agents)")

    comptime if FAST_VALIDATE:
        print(
            "  PROFILE: FAST-VALIDATE (screening; ~2× faster, do NOT trust for"
            " final convergence)"
        )
    else:
        print("  PROFILE: CONVERGE (full paper-faithful run)")
    print("=" * 70)

    var ctx = DeviceContext()

    var agent = Agent(
        ctx=ctx,
        lr=LR,
        gamma=GAMMA,
        v_min=V_MIN,
        v_max=V_MAX,
        value_coef=Scalar[DT](0.25),  # paper recommendation (vs 1.0)
        weight_decay=WEIGHT_DECAY,  # muzero-general Atari L2 (overfitting guard)
    )

    var env = PongPixelBatched(ctx)
    var eval_env = PongPixelBatched(ctx)

    print("Environment: Pong (GPU-batched Pixel,", N_ENVS, "envs)")
    print("Agent: MuZero (Gumbel-MCTS, learned model from pixels)")
    print("  Observation: 4 × 84 × 84 =", OBS)
    print("  Actions:", ACT, "(NOOP, UP, DOWN)")
    print("  Rep: Nature-CNN → latent", LATENT, " H", HIDDEN, " BINS", BINS)
    print(
        "  Search: Gumbel sims",
        NUM_SIMS,
        "MAX_K",
        MAX_K,
        "MAX_NODES",
        MAX_NODES,
    )
    print("  Value support [", V_MIN, ",", V_MAX, "] (h-space)  γ", GAMMA)
    print("  Replay CAP", CAP, "(device, uint8 obs ring)")
    print(
        "  Unroll B",
        B,
        "K",
        K,
        "N",
        N,
        " lr",
        LR,
        " train/iter",
        TRAIN_PER_ITER,
        "(replay ratio",
        Float64(TRAIN_PER_ITER) / Float64(N_ENVS),
        ")",
    )
    print(
        "  Reanalyze every",
        REANALYZE_EVERY,
        "batch",
        B,
        "(",
        B // N_ENVS,
        "search chunks/trigger, target-net sync 200)",
    )
    print("  Total env steps ≈", NUM_ITERS * N_ENVS)
    print()

    # ── metrics logger (silent no-op without RL_MONITOR_URL in env/.env) ──
    var env_vars = load_dotenv()
    var logger = RemoteLogger(
        server_url=env_vars.get("RL_MONITOR_URL", ""),
        run_name="MuZero Pong Pixel GPU (deep_agents)",
        buffer_size=64,
        api_key=env_vars.get("RL_MONITOR_API_KEY", ""),
    )
    logger.set_config("agent", "GumbelMuZero CNN")
    logger.set_config("env", "Pong (Pixel)")
    logger.set_config("obs", "4x84x84")
    logger.set_config("framework", "deep_agents/nn")
    logger.set_config("n_envs", String(N_ENVS))
    logger.set_config("num_sims", String(NUM_SIMS))
    logger.set_config("obs_store_dtype", "uint8")
    logger.set_config("batch_size", String(B))
    logger.set_config("train_per_iter", String(TRAIN_PER_ITER))
    logger.set_config("reanalyze_batch", String(B))
    logger.set_config("target_sync_interval", "200")
    logger.set_config("value_coef", "0.25")

    var start = perf_counter_ns()
    var loss = agent.train[L=RemoteLogger](
        env,
        iterations=NUM_ITERS,
        learning_starts=LEARNING_STARTS,
        train_per_iter=TRAIN_PER_ITER,
        max_ep_steps=MAX_EP_STEPS,
        temperature_decay_steps=NUM_ITERS,
        reanalyze_every=REANALYZE_EVERY,
        reanalyze_batch=B,  # ≈ training batch: most targets stay fresh
        target_sync_interval=200,  # target-net reanalyze (A/B-validated stabiliser)
        eval_every=10_000,
        eval_episodes=10,  # mean of 10 complete greedy games
        eval_env=UnsafePointer(to=eval_env),
        diag_every=200,
        report_every=500,
        logger=UnsafePointer(to=logger),
        seed=42,
        verbose=True,
        # Prioritized Experience Replay (device sum-tree over the strided obs
        # ring): sample stored positions ∝ root value-error, IS-weight the
        # grads, write back priorities. False reproduces the converged uniform
        # baseline (the working Pong path); flip True to focus training on the
        # high-error frames. alpha/beta default to EZ Atari's 1.0/1.0.
        use_per=True,
        # Exponential LR decay (muzero-general): lr 1e-4 → ×0.1 per ⅓ of the
        # grad-step budget. Pairs with weight_decay to curb the late-run
        # overfitting that regressed greedy eval in the constant-LR runs.
        lr_decay_rate=LR_DECAY_RATE,
        lr_decay_steps=LR_DECAY_STEPS,
    )
    var elapsed_s = Float64(perf_counter_ns() - start) / 1e9
    logger.close()

    print("-" * 70)
    print("MuZero Pong Pixel training complete")
    print("  final loss:", loss)
    print("  training time:", String(elapsed_s)[byte=:8], "seconds")
    print("=" * 70)
