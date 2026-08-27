"""TD-MPC2 dm_control `walker` — short fixed run for profiling. No logger.

The TD-MPC2 counterpart of `examples/half_cheetah/sac_half_cheetah_profile_graph_nn.mojo`:
same agent, env, dims and driver as `tdmpc2_dm_walker_batched_gpu.mojo`, but
sized to finish in minutes and stripped of everything that makes a training run
non-comparable — no `RemoteLogger`, no checkpoint writes, no periodic eval.
What is left is a fixed amount of work and a wall-clock number.

    pixi run -e nvidia mojo run -I . examples/dm_control/tdmpc2_dm_walker_profile_gpu.mojo
    pixi run -e nvidia nsys profile --stats=true mojo run -I . \
        examples/dm_control/tdmpc2_dm_walker_profile_gpu.mojo

## It reports TWO numbers, because the costs move independently

TD-MPC2's wall-clock splits into acting (MPPI planning, or the policy prior)
and `train_step` (the world-model gradient step). An optimization usually moves
ONE of them, and a single env-steps/s figure hides which:

  * PREFILL fills replay past `learning_starts` so the nets engage. UNTIMED.
  * PHASE 1 runs with `updates_per_step=0` — acting runs, no gradient step.
  * PHASE 2 runs with updates enabled. Subtracting phase 1's per-step cost
    attributes the remainder to `train_step`.

⚠⚠ `learning_starts` IS NOT AN UPDATES SWITCH — it gates ACTING TOO.
`agent.mojo:1484` takes the `warmup_uniform_batched` branch (uniform random
actions, no encoder, no policy, no MPPI) whenever
`replay.count() < learning_starts`, and `agent.mojo:1586` gates `train_step` on
the SAME condition. The first version of this file set
`learning_starts = COLLECT_STEPS` to suppress updates in phase 1 and thereby
suppressed ACTING for the whole phase: an nsys capture showed physics3d kernels
and nothing else — no matmul, no MPPI — and looked like a broken profiler
rather than a broken harness. `updates_per_step=0` is the correct separator
(`agent.mojo:1587` is a plain `for _ in range(updates_per_step)`).

This script now ASSERTS it left warmup before timing anything, so that failure
cannot recur silently.

⚠ Phase 2's per-step cost is NOT `train_step` alone — `updates_per_step`
gradient steps run per ITERATION (i.e. per N_ENVS env-steps), so read the
printed per-update figure, not the per-env-step one.

## What this is meant to measure

`benchmarks/bench_matmul_k_alignment.mojo` found `max_matmul` falls off a ~10x
cliff on a misaligned contraction dim, which TD-MPC2 hits constantly because
`ZA = LATENT + ACT` = 518. `Linear` now pads K to a multiple of 32 (517084c2),
measured at 2.3-2.9x on the dynamics/reward/Q trunks and 1.89x on a
1024-sample MPPI plan (M1 Pro, isolated microbenchmarks) and 7.6-7.8x on the
same GEMM shapes on an RTX 5090. This script exists to turn that into an
end-to-end training number, which the microbenchmarks consistently
over-predicted — the MPPI planner is partly bound by its ~700-launch dependent
kernel chain, and the chain absorbs GEMM savings.

⚠ NO VALID BASELINE HAS BEEN RECORDED YET. The first nsys capture measured the
warmup bug described above — physics3d kernels only — and is not a baseline.
Record a fresh run before quoting any before/after; the run should show
`max_matmul` and MPPI kernels dominating, and if it does not, read the PREFILL
assertion output first.

⚠ MPC ON is the expensive path by design — it runs the full MPPI budget per
env-step. `USE_MPC = False` profiles the policy-prior path instead, which is
~40x cheaper per action and makes `train_step` the dominant term. Profile both;
they are different workloads, not a fast and a slow version of one.

⚠ This is a THROUGHPUT harness, not a learning run. `TOTAL_STEPS` is far too
small to learn anything, and the return it prints is noise. Use
`tdmpc2_dm_walker_batched_gpu.mojo` to actually train.
"""

from std.random import seed
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.core.fmt import fit
from mojo_rl.deep_agents.tdmpc2.config import TDMPC2
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.envs.dm_control.walker.walker_xml import DMWalkerModel
from mojo_rl.envs.dm_control.walker.walker_config import DMWalkerConfig


# ─── Profiling knobs ──────────────────────────────────────────────────────
# ⚠ SIZED FOR NVIDIA. MPC acting at N_ENVS=8 pushes an 8 x 268 = 2144-row MPPI
# grid through the world model 12+ times per env-step; on an M1 Pro that is
# ~10 minutes at these counts, which is why this is not an Apple script. To
# make it Apple-viable, drop to N_ENVS=4 and 400/800 steps, or set USE_MPC=False.
comptime USE_MPC = True        # False → policy-prior acting (~40x cheaper)
comptime N_ENVS = 8
comptime LEARN_START = 512       # prefill: replay must pass this before the
                                 # driver stops taking uniform-random actions
comptime COLLECT_STEPS = 800     # phase 1: acting only, no gradient steps
comptime TRAIN_STEPS = 800       # phase 2: acting + updates
comptime UPDATES_PER_STEP = 1    # per ITERATION, not per env-step

# ⚠ KEEP THE TOTAL SMALL. The first two attempts at this file used
# 1_000/4_000/8_000 = 13_000 env-steps = 1_625 iterations, and both nsys
# captures contained ~104 instances of each physics kernel — i.e. ~104
# iterations, ~832 env-steps. The run was being cut off inside the PREFILL, so
# acting never engaged and the second trace looked identical to the first.
# 512/800/800 = 2_112 env-steps = 264 iterations, and the prefill clears in 64.
# Averaging over 100 iterations per phase is plenty for a throughput number;
# scale up only once a trace confirms the phases are doing what they claim.

# ─── MPPI budget (mirrors tdmpc2_dm_walker_batched_gpu.mojo) ──────────────
comptime MPC_SAMPLES = 256
comptime MPC_PI_TRAJS = 12
comptime MPC_ELITES = 32
comptime MPC_ITERS = 4

# ─── Sizing (mirrors tdmpc2_dm_walker_batched_gpu.mojo exactly) ───────────
comptime TASK: StaticString = "walk"
comptime MOVE_SPEED: Float64 = 0.0 if TASK == "stand" else (
    1.0 if TASK == "walk" else 8.0
)
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
# CAP must be a multiple of N_ENVS (the driver asserts) and hold prefill +
# both phases (1_000 + 4_000 + 8_000 = 13_000).
comptime CAP = 64_000
comptime LR = 3e-4

comptime Env = Phyics3dBatchedEnv[
    DMWalkerModel, DMWalkerConfig[MOVE_SPEED], N_ENVS,
    TERMINATE_ON_UNHEALTHY=False,
]

comptime AgentT = TDMPC2[
    "gpu", OBS, ACT, B, CAP, ENC, LATENT, MLP, BINS, SN, VMIN, VMAX, H,
    MPC_SAMPLES, MPC_PI_TRAJS, MPC_ELITES, MPC_ITERS,
]


def main() raises:
    comptime assert (
        COLLECT_STEPS % N_ENVS == 0 and TRAIN_STEPS % N_ENVS == 0
        and LEARN_START % N_ENVS == 0
    ), "step counts must be multiples of N_ENVS"
    comptime assert (
        CAP >= LEARN_START + COLLECT_STEPS + TRAIN_STEPS
    ), "CAP must hold prefill + both phases"
    comptime assert CAP % N_ENVS == 0, "CAP must be a multiple of N_ENVS"

    var mode = "MPC" if USE_MPC else "prior"
    print("=" * 70)
    print("TD-MPC2 dm_control walker", TASK, "— PROFILE (", mode, ")")
    print("=" * 70)
    print("  N_ENVS =", N_ENVS, " OBS =", OBS, " ACT =", ACT)
    print("  latent =", LATENT, " MLP =", MLP, " B =", B, " H =", H)
    print("  ZA = latent+act =", LATENT + ACT, " (K%32 =", (LATENT + ACT) % 32,
          "→ Linear pads it)")
    comptime if USE_MPC:
        print(
            "  MPPI =", MPC_SAMPLES, "+", MPC_PI_TRAJS, "trajs x", MPC_ITERS,
            "iters → grid", N_ENVS * (MPC_SAMPLES + MPC_PI_TRAJS), "rows",
        )
    print("  prefill:", LEARN_START, "env-steps (untimed; learning_starts)")
    print("  phase 1:", COLLECT_STEPS, "env-steps, updates_per_step=0")
    print("  phase 2:", TRAIN_STEPS, "env-steps,", UPDATES_PER_STEP,
          "update(s)/iteration")
    print("=" * 70)

    seed(42)
    var ctx = DeviceContext()
    var env = Env(ctx)

    # ⚠ `learning_starts` gates ACTING, not just updates — keep it SMALL and
    # suppress updates with `updates_per_step=0` instead. See the header.
    var ag = AgentT(
        ctx=ctx,
        lr=Scalar[DT](LR),
        action_scale=Scalar[DT](1.0),
        learning_starts=LEARN_START,
    )

    # ── prefill: get replay past `learning_starts`. UNTIMED. ─────────────
    # Until `replay.count() >= learning_starts` the driver takes uniform-random
    # actions and never touches a net, so timing this would measure physics
    # alone.
    print("PREFILL —", LEARN_START, "env-steps (untimed, random actions)")
    _ = ag.train_batched[Env, N_ENVS, USE_MPC=USE_MPC](
        env, LEARN_START, rng_seed=UInt64(41),
        updates_per_step=0, print_every=LEARN_START, verbose=False,
    )
    if ag.replay.count() < LEARN_START:
        raise Error(
            "prefill did not leave warmup: replay.count()="
            + String(ag.replay.count()) + " < learning_starts="
            + String(LEARN_START) + " — every timed phase below would be"
            " uniform-random actions with no network in the trace. Raise"
            " LEARN_START's prefill or check the replay's counting unit."
        )
    print("  replay.count() =", ag.replay.count(), ">= learning_starts —"
          " acting is live")

    # ── phase 1: env + acting only (updates_per_step=0) ──────────────────
    print("PHASE 1 — collect (acting only)")
    var t0 = perf_counter_ns()
    _ = ag.train_batched[Env, N_ENVS, USE_MPC=USE_MPC](
        env, COLLECT_STEPS, rng_seed=UInt64(42),
        updates_per_step=0,
        print_every=COLLECT_STEPS, verbose=False,
        base_step=LEARN_START,
    )
    var t1 = perf_counter_ns()
    var collect_s = Float64(t1 - t0) / 1e9
    var collect_per_step_ms = collect_s * 1000.0 / Float64(COLLECT_STEPS)

    # ── phase 2: env + acting + updates ──────────────────────────────────
    print(
        "  phase 1 done —", COLLECT_STEPS // N_ENVS,
        "iterations of ACTING ran; a trace of this phase must contain matmul"
        " / MPPI kernels",
    )
    print("PHASE 2 — collect + train")
    var t2 = perf_counter_ns()
    _ = ag.train_batched[Env, N_ENVS, USE_MPC=USE_MPC](
        env, TRAIN_STEPS, rng_seed=UInt64(43),
        updates_per_step=UPDATES_PER_STEP,
        print_every=TRAIN_STEPS, verbose=False,
        base_step=LEARN_START + COLLECT_STEPS,
    )
    var t3 = perf_counter_ns()
    var train_s = Float64(t3 - t2) / 1e9
    var train_per_step_ms = train_s * 1000.0 / Float64(TRAIN_STEPS)

    # Phase 2 minus phase 1's per-step acting cost = the gradient-step cost.
    # `updates_per_step` updates run per ITERATION (N_ENVS env-steps), so
    # convert to a per-update figure before comparing against anything.
    var delta_ms = train_per_step_ms - collect_per_step_ms
    var updates_done = (TRAIN_STEPS // N_ENVS) * UPDATES_PER_STEP
    var per_update_ms = (
        (delta_ms * Float64(TRAIN_STEPS) / Float64(updates_done))
        if updates_done > 0 else 0.0
    )

    print()
    print("=" * 70)
    print("RESULTS —", TASK, "(", mode, ",", N_ENVS, "envs )")
    print("=" * 70)
    print(
        "  phase 1  collect      ", fit(String(collect_s), 7), "s   ",
        fit(String(collect_per_step_ms), 6), "ms/env-step   ",
        fit(String(Float64(COLLECT_STEPS) / collect_s), 7), "env-steps/s",
    )
    print(
        "  phase 2  collect+train", fit(String(train_s), 7), "s   ",
        fit(String(train_per_step_ms), 6), "ms/env-step   ",
        fit(String(Float64(TRAIN_STEPS) / train_s), 7), "env-steps/s",
    )
    print("  ---")
    print(
        "  acting  (phase 1)      ", fit(String(collect_per_step_ms), 6),
        "ms / env-step",
    )
    print(
        "  train_step (by delta)  ", fit(String(per_update_ms), 6),
        "ms / update   (", updates_done, "updates )",
    )
    print("=" * 70)
    print("⚠ throughput harness — the run is far too short to learn anything.")
    print("  To train:  examples/dm_control/tdmpc2_dm_walker_batched_gpu.mojo")
    print("=" * 70)
