"""TD-MPC2 MULTI-TASK on dm_control `walker` — stand + walk + run, one agent.

The point of the whole TD-MPC2 arc: ONE task-conditioned world model, one
Q-ensemble and one policy trained on three rewards at once, against the
single-task runs (`tdmpc2_dm_walker_batched_gpu.mojo`) as the baseline. Multi-
task TD-MPC2 has never been validated here — the pieces existed
(`TDMPC2MultiTaskAgent`, the `*MT` nets, the task-embedding table) but nothing
trained them, because there was no multi-task env to train on until dm_control
landed.

## Why walker is the right first multi-task target

All three tasks share ONE model, ONE observation (24) and ONE action space (6);
only the reward differs (MOVE_SPEED 0 / 1 / 8). So the obs/action PADDING that
multi-task TD-MPC2 normally needs is a no-op here, and what is left under test
is exactly the interesting part: does the task embedding let one world model
serve three reward functions? A negative result here is about conditioning, not
about plumbing dims — which is what makes it a clean first experiment.

That is also a hard requirement today: `train_batched_mt` asserts
`E.OBS_DIM == MAX_OBS` and `E.ACT_DIM == MAX_ACT`. Nothing pads an env's slabs
yet, so heterogeneous bodies need a padding `BatchedEnv` wrapper first.

## How the three tasks share the agent

Each dm_control task is a distinct `Phyics3dEnvConfig`, hence a distinct env
TYPE, so they cannot be iterated at runtime — the three `train_batched_mt`
calls below are written out, one per task, and each is its own instantiation.

Training alternates SEGMENTS: task 0 collects for `SEGMENT_STEPS`, then task 1,
then task 2, and round again. This is still joint multi-task learning, not
sequential fine-tuning, because the replay keeps every task and `train_step`
samples windows uniformly across all of them — during run's segment, most of
each batch is stand and walk data. What alternates is only which task the
ACTING policy is driving.

⚠ Keep `SEGMENT_STEPS` well above `EPISODE_LEN * N_ENVS` (8000 here = one
episode per env) or a segment cannot finish an episode and its `mean_ret` is
stale from the previous round.

⚠ MPC is NOT available on the multi-task agent (no planner is built for it), so
this run is MPC-off on both acting and eval. The single-task comparison should
therefore be read against an MPC-off single-task run, not against the 932 that
the MPC run reached — the planner is worth a large fraction of that number.

Run:
    pixi run -e nvidia mojo run -I . examples/dm_control/tdmpc2_dm_walker_multitask_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.tdmpc2.config_mt import TDMPC2MultiTask
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.envs.dm_control.walker.walker_xml import DMWalkerModel
from mojo_rl.envs.dm_control.walker.walker_config import DMWalkerConfig

comptime TARGET = "gpu"

comptime N_ENVS = 8
comptime EVAL_ENVS = 8

comptime MAX_OBS = DMWalkerModel.OBS_DIM      # 24 — shared by all three
comptime MAX_ACT = DMWalkerModel.ACTION_DIM   #  6 — shared by all three
comptime NUM_TASKS = 3
# Reference multi-task TD-MPC2 uses 96 for MT80. Three tasks over one body need
# far less; this is the knob to raise if the tasks start interfering.
comptime TASK_EMB = 32

comptime ENC = 256
comptime LATENT = 512
comptime MLP = 512
comptime BINS = 101
comptime SN = 8
comptime VMIN = -10
comptime VMAX = 10
comptime B = 256
comptime H = 3
comptime CAP = 1_000_000       # MUST be a multiple of N_ENVS

comptime LR = 3e-4
comptime ACTION_SCALE = 1.0
comptime LEARN_START = 5_000   # replay frames before the policy takes over
comptime UPDATES_PER_STEP = 1  # per ITERATION (= N_ENVS env-steps)
comptime EPISODE_LEN = 1_000
comptime SEGMENT_STEPS = EPISODE_LEN * N_ENVS   # 8 000 — one episode per env
comptime N_ROUNDS = 40         # 3 x 40 x 8 000 = 960 k env-steps total
comptime EVAL_EVERY = SEGMENT_STEPS   # once per segment, on that task
comptime DIAG_EVERY = 1_000
comptime PRINT_EVERY = 4_000
comptime CKPT = "tdmpc2_dm_walker_multitask.ckpt"

comptime MAX_RETURN = 1000.0

# One env TYPE per task — the reason the three calls below are written out.
comptime StandEnv = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[0.0], N_ENVS, TERMINATE_ON_UNHEALTHY=False
]
comptime WalkEnv = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[1.0], N_ENVS, TERMINATE_ON_UNHEALTHY=False
]
comptime RunEnv = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[8.0], N_ENVS, TERMINATE_ON_UNHEALTHY=False
]
comptime StandEval = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[0.0], EVAL_ENVS, TERMINATE_ON_UNHEALTHY=False
]
comptime WalkEval = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[1.0], EVAL_ENVS, TERMINATE_ON_UNHEALTHY=False
]
comptime RunEval = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[8.0], EVAL_ENVS, TERMINATE_ON_UNHEALTHY=False
]

# Task ids — the index into the embedding table. ⚠ These must stay stable
# across a resume: a checkpoint's embedding row 1 means "walk" only because
# this table says so.
comptime T_STAND = 0
comptime T_WALK = 1
comptime T_RUN = 2


def main() raises:
    print("=" * 70)
    print("TD-MPC2 MULTI-TASK — dm_control walker stand + walk + run")
    print("=" * 70)
    print("  tasks    =", NUM_TASKS, " task_emb =", TASK_EMB)
    print("  N_ENVS   =", N_ENVS, " OBS =", MAX_OBS, " ACT =", MAX_ACT)
    print("  latent   =", LATENT, " B =", B, " H =", H)
    print("  segment  =", SEGMENT_STEPS, "env-steps x", N_ROUNDS, "rounds")
    print("  total    =", SEGMENT_STEPS * N_ROUNDS * NUM_TASKS, "env-steps")
    print("  MPC      = OFF (no planner on the multi-task agent)")
    print("=" * 70)
    seed(0)
    var ctx = DeviceContext()

    var stand = StandEnv(ctx)
    var walk = WalkEnv(ctx)
    var run_e = RunEnv(ctx)
    var stand_ev = StandEval(ctx)
    var walk_ev = WalkEval(ctx)
    var run_ev = RunEval(ctx)
    var stand_ev_p = Pointer(to=stand_ev).as_unsafe_any_origin()
    var walk_ev_p = Pointer(to=walk_ev).as_unsafe_any_origin()
    var run_ev_p = Pointer(to=run_ev).as_unsafe_any_origin()

    var ag = TDMPC2MultiTask[
        TARGET, MAX_OBS, MAX_ACT, NUM_TASKS, TASK_EMB, B, CAP,
        ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
    ](
        ctx=ctx, lr=Scalar[DT](LR),
        action_scale=Scalar[DT](ACTION_SCALE), learning_starts=LEARN_START,
    )

    var env_vars = load_dotenv()
    var logger = RemoteLogger(
        server_url=env_vars.get("RL_MONITOR_URL", ""),
        run_name="TD-MPC2 dm_control walker MULTI-TASK",
        buffer_size=64,
        api_key=env_vars.get("RL_MONITOR_API_KEY", ""),
    )
    logger.set_config("algorithm", "TD-MPC2-MT")
    logger.set_config("env", "dm_control/walker-stand+walk+run")
    logger.set_config("target", TARGET)
    logger.set_config("n_envs", String(N_ENVS))
    logger.set_config("num_tasks", String(NUM_TASKS))
    logger.set_config("task_emb", String(TASK_EMB))
    logger.set_config("segment_steps", String(SEGMENT_STEPS))
    var lg = Pointer(to=logger).as_unsafe_any_origin()
    if env_vars.get("RL_MONITOR_URL", "").byte_length() > 0:
        print("  logger: ENABLED → eval/<task> + avg_reward/<task>")
    else:
        print("  logger: DISABLED — RL_MONITOR_URL not in .env")

    print("Starting multi-task training ...")
    print("-" * 70)
    var t_start = perf_counter_ns()

    var best_stand: Scalar[DT] = -1.0e30
    var best_walk: Scalar[DT] = -1.0e30
    var best_run: Scalar[DT] = -1.0e30
    var at = 0

    for rnd in range(N_ROUNDS):
        # ⚠ `base_step` must keep counting ACROSS tasks and rounds, or the
        # dashboard shows N_ROUNDS x 3 overlapping runs. The warmup gate reads
        # the replay count, not this, so it is a logging axis only.
        var b0 = ag.train_batched_mt[
            StandEnv, N_ENVS, RemoteLogger, StandEval, EVAL_ENVS
        ](
            stand, T_STAND, SEGMENT_STEPS,
            rng_seed=UInt64(100 + rnd), updates_per_step=UPDATES_PER_STEP,
            print_every=PRINT_EVERY, verbose=True, logger=lg,
            diag_every=DIAG_EVERY, base_step=at,
            checkpoint_path=CKPT, checkpoint_every=0,
            eval_env=stand_ev_p, eval_every=EVAL_EVERY,
            eval_max_steps=EPISODE_LEN, task_label=String("stand"),
        )
        at += SEGMENT_STEPS
        if b0 > best_stand:
            best_stand = b0

        var b1 = ag.train_batched_mt[
            WalkEnv, N_ENVS, RemoteLogger, WalkEval, EVAL_ENVS
        ](
            walk, T_WALK, SEGMENT_STEPS,
            rng_seed=UInt64(200 + rnd), updates_per_step=UPDATES_PER_STEP,
            print_every=PRINT_EVERY, verbose=True, logger=lg,
            diag_every=DIAG_EVERY, base_step=at,
            checkpoint_path=CKPT, checkpoint_every=0,
            eval_env=walk_ev_p, eval_every=EVAL_EVERY,
            eval_max_steps=EPISODE_LEN, task_label=String("walk"),
        )
        at += SEGMENT_STEPS
        if b1 > best_walk:
            best_walk = b1

        var b2 = ag.train_batched_mt[
            RunEnv, N_ENVS, RemoteLogger, RunEval, EVAL_ENVS
        ](
            run_e, T_RUN, SEGMENT_STEPS,
            rng_seed=UInt64(300 + rnd), updates_per_step=UPDATES_PER_STEP,
            print_every=PRINT_EVERY, verbose=True, logger=lg,
            diag_every=DIAG_EVERY, base_step=at,
            checkpoint_path=CKPT, checkpoint_every=0,
            eval_env=run_ev_p, eval_every=EVAL_EVERY,
            eval_max_steps=EPISODE_LEN, task_label=String("run"),
        )
        at += SEGMENT_STEPS
        if b2 > best_run:
            best_run = b2

        # One checkpoint per ROUND — after all three tasks have collected, so
        # the file is never a mid-round snapshot biased to the last task.
        ag.save_state(CKPT)
        print(
            "  ── round", rnd + 1, "/", N_ROUNDS, " @", at, "env-steps",
            "  best: stand", best_stand, " walk", best_walk,
            " run", best_run,
        )

    _ = stand_ev
    _ = walk_ev
    _ = run_ev
    var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
    logger.close()
    _ = logger

    print("-" * 70)
    print("=" * 70)
    print("Multi-task training complete")
    print("  total env_steps =", at)
    print("  elapsed         =", elapsed, "s")
    print("  best eval — stand:", best_stand, " walk:", best_walk,
          " run:", best_run)
    print("  checkpoint      =", CKPT)
    print("=" * 70)
    print("Read it against the SINGLE-TASK, MPC-OFF baseline per task —")
    print("  examples/dm_control/tdmpc2_dm_walker_batched_gpu.mojo (USE_MPC=False)")
    print("  MPC is off here, so the 932 from the MPC run is NOT the baseline.")
    print("")
    print("  3 tasks at parity with single-task  → conditioning works.")
    print("  stand fine, run collapsed           → the shared model is being")
    print("    dominated by the easy task; raise TASK_EMB or rebalance the")
    print("    segment lengths before concluding anything about TD-MPC2.")
    print("=" * 70)
