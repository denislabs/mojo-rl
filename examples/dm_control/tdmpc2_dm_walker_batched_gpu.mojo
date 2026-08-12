"""TD-MPC2 on dm_control `walker`, N_ENVS BATCHED (GPU env + GPU nets).

The batched counterpart of `tdmpc2_dm_walker_gpu.mojo`. Same agent, same task,
same reward — the difference is that the env is `Phyics3dBatchedEnv` (N walkers
stepped in one kernel grid, the same env SAC uses here) and the driver is
`agent.train_batched`, which did not exist until now.

Why it matters: the single-env script measured ~2.9 env-steps/s on Apple with
MPC, because one CPU-stepped walker at a time cannot fill a GPU. Batching moves
three things onto the batch axis at once:

  * the physics — one `step_batch` for all N envs;
  * acting — ONE `encoder → policy → rsample` pass over [N, ·], or ONE
    `plan_gpu` whose grid is N × (MPC_SAMPLES + MPC_PI_TRAJS) MPPI candidates;
  * the host sync — one D2H per ITERATION (N env-steps) instead of per step.

`train_step` is untouched: it samples B windows from replay, which never cared
how the data was collected.

## The replay subtlety this driver handles for you

N envs write into one sequence ring round-robin, so slot p and p+1 are
DIFFERENT envs while p and p+N are consecutive frames of the SAME env. A
contiguous window would be a world model trained on transitions that never
happened — and it would still train, and the loss would still fall. The driver
calls `replay.set_env_stride(N_ENVS)` so the sampler walks lanes of stride N.
`tests/deep_agents/test_tdmpc2_batched_smoke.mojo` asserts that windows are
single-env directly, because the loss curve cannot.

## What has actually been measured (2026-08-11)

Apple/Metal, 4 envs, MPC on, B=64: the loop runs end to end and the WM loss
falls (0.27 → 0.15 over 400 iterations). That is a FUNCTIONAL result, not a
throughput one — there is no clean apples-to-apples speedup number yet, because
the single-env baseline runs one gradient step per env-step while the batched
run does one per ITERATION. Benchmark this on NVIDIA (the standing rule for
this repo: bench on NVIDIA, Apple is for parity), holding
`updates_per_step / N_ENVS` fixed between the two, before quoting any factor.

Expect the win to come from collection and acting, not from `train_step` —
that one does identical work either way, and on Metal it dominates.

## Sizing

`N_ENVS` multiplies the MPPI grid: BATCH_TOTAL = N_ENVS × (MPC_SAMPLES +
MPC_PI_TRAJS) rows through dynamics/reward/Q per horizon step per iteration.
At the defaults below that is 8 × 268 = 2144 — comfortable. Raising N_ENVS to
32 with the reference 512/24 budget puts 17 152 rows through a 512-wide MLP
twelve times per env-step; feasible on an NVIDIA card, not on Apple. Scale
N_ENVS up and the MPPI budget down together, or run MPC-off.

`updates_per_step` is per ITERATION. `updates_per_step=N_ENVS` reproduces the
single-env ratio of one gradient step per env-step (the reference ratio);
lower it to trade sample-efficiency for wall-clock.

## UTD — what this file is currently configured to test (2026-08-12)

Every walker run before today used `updates_per_step=1`, i.e. UTD=0.125 —
**one eighth of the reference ratio**. The diagnostics say that is the binding
constraint: over a clean 220k-step run, `consistency_loss` fell all the way
(0.045 → 0.011) while `value_loss` and `reward_loss` went FLAT at ~0.028 from
50k onward. The dynamics model kept learning; the two heads that decide control
stopped. `q_mean` tracks `td_target_mean` to within 0.5% and sits 7-18% under
realized returns, so the critic is calibrated — under-trained, not broken.

⚠ Price a UTD change in GRADIENT STEPS, not env-steps. Holding TOTAL fixed and
raising the ratio 8x reads as "7x slower" and is the wrong comparison: at
UTD=1, 150k env-steps costs ~2.2h and buys 150k updates, where the 220k-step
control run took ~20 min and bought 27.5k. Fewer steps, far more learning.

Measured controls to compare against (single-task walk, N_ENVS=8, UTD=0.125,
post-`1cc6f779`; `mean_ret(100)`):

| env-steps | MPC off | MPC on |
|-----------|---------|--------|
| 120k      | 196     | 181    |
| 210k      | 302     | 529    |

The two controllers are indistinguishable below ~120k and only then does the
planner pull away (final evals ~300 vs ~880). That is why the UTD test runs
MPC-OFF: below the crossover the planner buys nothing, so leaving it on would
cost 1.47x wall-clock to confound the one variable under test.

⚠⚠ Both control curves above are only valid post-`1cc6f779`. Runs built in the
`517084c2`..`baeaa9bc` window had FROZEN target Q nets (a version-gated weight
cache the polyak write never invalidated) and scored 74.7 at 150k where the
fixed build scores 230.5. Do not compare against a number from that window.

Run:
    pixi run -e nvidia mojo run -I . examples/dm_control/tdmpc2_dm_walker_batched_gpu.mojo
    pixi run -e apple  mojo run -I . examples/dm_control/tdmpc2_dm_walker_batched_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.tdmpc2.config import TDMPC2
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.envs.dm_control.walker.walker_xml import DMWalkerModel
from mojo_rl.envs.dm_control.walker.walker_config import DMWalkerConfig


# ── pick ONE ─────────────────────────────────────────────────────────────
comptime TASK: StaticString = "walk"  # "stand" | "walk" | "run"

comptime MOVE_SPEED: Float64 = 0.0 if TASK == "stand" else (
    1.0 if TASK == "walk" else 8.0
)

comptime TARGET = "gpu"        # batched driver requires env target == this
# MPC-off for the UTD test: the two controllers are indistinguishable below
# ~120k env-steps (measured — see the UTD block below), so leaving the planner
# off isolates the update ratio and costs 1.47x less wall-clock.
comptime USE_MPC = False
comptime MPC_SAMPLES = 256
comptime MPC_PI_TRAJS = 12
comptime MPC_ELITES = 32
comptime MPC_ITERS = 4

comptime N_ENVS = 8            # training envs stepped in lockstep
comptime EVAL_ENVS = 8         # isolated eval env batch (see `eval_env`)

comptime OBS = DMWalkerModel.OBS_DIM       # 24
comptime ACT = DMWalkerModel.ACTION_DIM    #  6
comptime ENC = 256
comptime LATENT = 512
comptime MLP = 512
comptime BINS = 101
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime B = 256
comptime H = 3
comptime CAP = 1_000_000       # MUST be a multiple of N_ENVS (driver asserts)

comptime LR = 3e-4
comptime ACTION_SCALE = 1.0
# All step counts below are TOTAL env-steps ACROSS ALL ENVS (SAC's convention):
# the driver runs `TOTAL // N_ENVS` iterations.
comptime LEARN_START = 5_000
# ⚠ TOTAL is deliberately SHORT here. At UTD=1 the run is priced in GRADIENT
# STEPS, not env-steps: 150k env-steps buys 150k updates, against the 27.5k
# that a 220k-step run at UPDATES_PER_STEP=1 delivered. Do not "restore" this
# to 1M without also dropping UPDATES_PER_STEP — that is a ~20h run.
comptime TOTAL = 150_000
# Per ITERATION, and an iteration is N_ENVS env-steps — so this value IS the
# UTD numerator: N_ENVS gives the reference ratio of 1 update per env-step,
# 1 gives 0.125. Every walker run before 2026-08-12 used 1, i.e. 1/8 of the
# published recipe, which is why the reward and value heads flatlined at 50k
# while consistency_loss kept falling.
comptime UPDATES_PER_STEP = N_ENVS
# Halved vs the control run's 25k: eval spread on this task is ~±65, so the
# curve needs points. Every SECOND point still lands on a 25k multiple and so
# lines up exactly with the UPDATES_PER_STEP=1 control.
comptime EVAL_EVERY = 12_500
comptime EP_LEN = 1_000        # dm_control's own limit
comptime DIAG_EVERY = 1_000
comptime PRINT_EVERY = 10_000
comptime CHECKPOINT_EVERY = 50_000

comptime MAX_RETURN = 1000.0

# TERMINATE_ON_UNHEALTHY=False: dm_control never terminates early, so the
# driver records terminated=0 throughout and the value bootstrap survives the
# 1000-step truncation.
comptime Env = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[MOVE_SPEED], N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]
comptime EvalEnv = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[MOVE_SPEED], EVAL_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]


def main() raises:
    comptime assert (
        TASK == "stand" or TASK == "walk" or TASK == "run"
    ), "TASK must be 'stand', 'walk' or 'run'"

    var mode = "MPC" if USE_MPC else "MPC-off"
    print("=" * 70)
    print("TD-MPC2 — dm_control walker", TASK, "— BATCHED (", mode, ")")
    print("=" * 70)
    print("  N_ENVS =", N_ENVS, " OBS =", OBS, " ACT =", ACT)
    print("  latent =", LATENT, " B =", B, " H =", H)
    print("  total env-steps =", TOTAL, " (", TOTAL // N_ENVS, "iterations )")
    print("  updates/iteration =", UPDATES_PER_STEP)
    comptime if USE_MPC:
        print(
            "  MPPI =", MPC_SAMPLES, "+", MPC_PI_TRAJS, "trajs x", MPC_ITERS,
            "iters  → grid", N_ENVS * (MPC_SAMPLES + MPC_PI_TRAJS), "rows",
        )
    print("=" * 70)
    seed(0)
    var ctx = DeviceContext()

    var env = Env(ctx)
    var eval_env = EvalEnv(ctx)
    var eval_env_ptr = Pointer(to=eval_env).as_unsafe_any_origin()

    # ⚠ The UTD tag is part of the name on purpose: without it this run
    # overwrites the UPDATES_PER_STEP=1 checkpoint, which is the CONTROL for
    # the comparison it exists to make.
    var ckpt = (
        String("tdmpc2_dm_walker_batched_") + String(TASK)
        + ("_mpc" if USE_MPC else "_mpcoff")
        + "_utd" + String(UPDATES_PER_STEP) + ".ckpt"
    )

    var ag = TDMPC2[
        TARGET, OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
        MPC_SAMPLES, MPC_PI_TRAJS, MPC_ELITES, MPC_ITERS,
    ](
        ctx=ctx, lr=Scalar[DT](LR),
        action_scale=Scalar[DT](ACTION_SCALE), learning_starts=LEARN_START,
    )

    var env_vars = load_dotenv()
    var logger = RemoteLogger(
        server_url=env_vars.get("RL_MONITOR_URL", ""),
        run_name=(
            String("TD-MPC2 dm_control walker ") + String(TASK) + " x"
            + String(N_ENVS)
        ),
        buffer_size=64,
        api_key=env_vars.get("RL_MONITOR_API_KEY", ""),
    )
    logger.set_config("algorithm", "TD-MPC2")
    logger.set_config("env", String("dm_control/walker-") + String(TASK))
    logger.set_config("target", TARGET)
    logger.set_config("mpc", String("1") if USE_MPC else String("0"))
    logger.set_config("n_envs", String(N_ENVS))
    logger.set_config("latent", String(LATENT))
    logger.set_config("batch", String(B))
    var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()
    if env_vars.get("RL_MONITOR_URL", "").byte_length() > 0:
        print("  logger: ENABLED → streaming every", DIAG_EVERY, "steps")
    else:
        print("  logger: DISABLED — RL_MONITOR_URL not in .env")

    print("Starting training...")
    print("-" * 70)
    var t_start = perf_counter_ns()
    var best = ag.train_batched[
        Env, N_ENVS, RemoteLogger, USE_MPC, EvalEnv, EVAL_ENVS
    ](
        env,
        TOTAL,
        rng_seed=UInt64(42),
        updates_per_step=UPDATES_PER_STEP,
        print_every=PRINT_EVERY,
        verbose=True,
        logger=logger_ptr,
        diag_every=DIAG_EVERY,
        checkpoint_path=ckpt,
        checkpoint_every=CHECKPOINT_EVERY,
        eval_env=eval_env_ptr,
        eval_every=EVAL_EVERY,
        eval_max_steps=EP_LEN,
    )
    _ = eval_env  # lifetime extender for eval_env_ptr
    var elapsed = Float64(perf_counter_ns() - t_start) / 1e9

    logger.close()
    _ = logger  # lifetime extender for logger_ptr

    print("-" * 70)
    print("=" * 70)
    print("Training complete —", TASK, "(", mode, ", ", N_ENVS, "envs )")
    print("  total env_steps  =", TOTAL)
    print("  elapsed          =", elapsed, "s")
    print("  env-steps/s      =", Float64(TOTAL) / elapsed)
    print("  best eval return =", best)
    print("  checkpoint       =", ckpt)
    print("=" * 70)

    var frac = Float64(best) / MAX_RETURN
    if frac > 0.8:
        print("EXCELLENT — near the dm_control ceiling (>0.8 x 1000).")
    elif frac > 0.5:
        print("STRONG — solved the task (>0.5 x 1000).")
    elif frac > 0.2:
        print("PROGRESS — partial competence (>0.2 x 1000).")
    else:
        print("WEAK — check `wm=` in the progress lines first.")
    print("")
    print("Single-env baseline (same task, same reward):")
    print("  examples/dm_control/tdmpc2_dm_walker_gpu.mojo")
    print("SAC on the same task:")
    print("  examples/dm_control/sac_dm_walker_training_gpu.mojo")
    print("=" * 70)
