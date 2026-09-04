"""P0 — WHICH TERM carries the quadratic? One `k` per process, for nsys.

`so101_park_budget_probe.mojo` measured that a parked slot costs 7.88x at k=9
and that the cost is QUADRATIC in the added dofs. It did not say WHICH
FUNCTION. `docs/BLOCK_DIAGONAL_MASS_MATRIX_PLAN.md` §2.2 hypothesises the
`ldl_factor` + `compute_m_inv` pair and says, in as many words, to confirm that
by attribution before changing anything — because if it is some other function
the whole plan targets the wrong one.

    pixi run -e nvidia bash scripts/p0_attrib.sh          # all four k
    pixi run python scripts/p0_attrib.py                  # the table

⚠⚠ WHY THIS FILE EXISTS AT ALL — ONE `k` PER PROCESS.
The obvious move is to run the EXISTING probe once under nsys and read the
per-kernel summary. It does not work, and the reason is worth writing down:
Mojo mangles a kernel's comptime parameters into a HASH, not a spelling
(`mojo_rl_nn_primitives_conv2d6A6A6A6A6A6A6A_5bd29d73087ee488` — the module
path survives, the `Int`s do not). The existing probe instantiates every leg in
ONE process, so `_ldl_factor_fields_mt_kernel` appears four times under four
hashes with nothing to say which is `NV=6` and which is `NV=60`. The only
in-band way to tell them apart would be to sort by duration and assign the
biggest to the biggest `nv` — which is assuming the conclusion the probe exists
to test. So: one leg, one process, `k` from argv, and nsys attributes without
ambiguity.

⚠ THE FIVE TERMS ARE FIVE SEPARATE KERNELS, which is what makes this decidable
at all: CRBA (`compute_mass_matrix`), `ldl_factor`, `compute_m_inv`, the
constraint solve (`newton_solve`), and — on Euler models, NOT this one — the
fused `_finalize_kernel`. `cuda_gpu_kern_sum` splits them for free.

⚠⚠ THIS MODEL IS **RK4**, AND THE LAUNCH COUNT IS NOT ONE PER STEP.
`So101ParkProbeConfig` does not override `INTEGRATOR`, so it inherits `"rk4"`
(and sizes `INTEGRATOR_WS_EXTRA` with `rk4_extra_workspace_size`). `ldl_factor`
and `compute_m_inv` live inside `_stage_dynamics` (`rk4.mojo:565-566`), which
runs ONCE PER STAGE — four stages — and `FRAME_SKIP = 2` puts two physics steps
in an env step. So a term's per-step cost is `avg_kernel_ns * launches_per_step`
with `launches_per_step = 8`, not 1. `scripts/p0_attrib.py` DERIVES that number
from the instance count rather than trusting this paragraph, because a launch
count read off a docstring is exactly the sort of constant that drifts.

⚠ Euler's second dense LDL (`euler.mojo:403`, the `M_hat = M + dt*diag(damping)`
re-factorisation) is NOT on this path — RK4 has no eulerdamp stage. It remains a
real second O(nv^3) site for Euler-integrated models and this probe says nothing
about it.

⚠ RUN IT ON NVIDIA. Apple cannot build this scene past k=0 (`Compute function
exceeds available stack space`) — see the budget probe's header, which
retracted an Apple number for exactly this reason.

⚠ NO REPARK LEG HERE. Leg 3 of the budget probe already showed repark is free
(0.11%, inside the spread). Adding it would double the k=9 kernel instances for
no new information and muddy the per-launch averages.
"""

from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.envs.robots.so101_park_config import So101ParkProbeConfig
from mojo_rl.envs.robots.so101_park_xml import (
    SoArm101ParkK0Model,
    SoArm101ParkK3Model,
    SoArm101ParkK6Model,
    SoArm101ParkK9Model,
    SoArm101ParkK12Model,
    SoArm101ParkK13Model,
)


# ⚠ THE SAME KNOBS AS THE BUDGET PROBE. A different lane count or step count
# would make the two tables incomparable, and the whole point of this one is to
# decompose that one's rows.
comptime N_ENVS = 1024
# ⚠⚠ 200, NOT THE BUDGET PROBE'S 50, AND IT FIXES TWO THINGS AT ONCE.
# `nsys` averages a kernel over EVERY launch — the warmup included — while
# `probe wall` covers only the timed region. If the GPU is still ramping its
# clocks during warmup, the per-launch average is inflated by launches the wall
# time never saw, and the attribution comes out with GPU total ABOVE wall time:
# a NEGATIVE residual, which is arithmetically impossible for a sound
# measurement. That is exactly what a 2026-09-03 sweep produced — residuals of
# -0.05 to -4.64 ms where every earlier run was within -0.4, and every kernel
# the run did not touch inflated 12-85%.
#
# More warmup helps both halves: the timed region starts at settled clocks, AND
# the fixed ramp becomes a smaller fraction of the launches the average is
# taken over. `scripts/p0_attrib.py` now FAILS on a negative residual rather
# than printing a table that looks fine.
comptime WARMUP_STEPS = 200
# ⚠⚠ AND A TIME FLOOR, BECAUSE A STEP COUNT IS THE WRONG UNIT FOR A CLOCK RAMP.
# 200 steps is 8 s of GPU work at k=13 and only 1.4 s at k=0, and k=0 runs FIRST
# in the sweep, from the coldest state. Measured 2026-09-03 with a 200-step
# warmup: k>=3 came within 1% of the previous clean run on every untouched
# kernel, while k=0 was +43-46% ON EVERY KERNEL with a -20.1% residual. The
# control is the denominator of the `x k=0` column, so a cold k=0 corrupts
# every ratio in the table while looking like a fast control.
comptime WARMUP_SECONDS = 5.0
comptime TIMED_STEPS = 300

# ⚠⚠ RESET EVERY STEP — THE FIX FOR A BISECT THAT FED BACK INTO ITS OWN INPUT.
# `NEWTON_STOP_AFTER` truncates the solver, which leaves `qacc_constrained` at
# the SMOOTH acceleration. That is a stable answer, and I checked it was
# bounded — but stability is not workload equivalence. An unconstrained arm
# blows through its JOINT LIMITS, and joint limits are constraint ROWS, so the
# truncated build ran a different and more expensive problem. The tell was
# arithmetic: `STOP=5` measured 18.002 ms against the FULL kernel's 16.554, and
# a PREFIX OF A PROGRAM CANNOT COST MORE THAN THE WHOLE PROGRAM ON THE SAME
# INPUT — so the input had changed.
#
# ⚠ AND THE CONTROLS DID NOT CATCH IT. `collision` sat at 1.005 throughout,
# because its cost is dominated by the geometry-pair count and not by how many
# constraint rows the state produces. A control that cannot observe the quantity
# you are worried about is not a control.
#
# With a FIXED seed, `reset_batch` makes every step start from the identical
# state, so nothing the kernel writes can reach the next step and every arm
# measures the same problem. That is the only property a bisect needs.
#
# ⚠ IT CHANGES WHAT IS ATTRIBUTED, and the number is not comparable to a
# non-reset run: this measures the RESET POSE, not a settled trajectory. Use it
# to compare arms with each other, never against the main sweep's absolutes.
#
# ⚠ The reset's own kernels enter the nsys table. They are identical across
# arms so they cannot bias a comparison, but they do inflate wall time and the
# non-`newton` rows.
comptime RESET_EVERY_STEP: Bool = False
comptime RESET_SEED: UInt64 = 42

comptime ParkCfg0 = So101ParkProbeConfig[6, 6, 0]
comptime ParkCfg3 = So101ParkProbeConfig[27, 24, 3]
comptime ParkCfg6 = So101ParkProbeConfig[48, 42, 6]
comptime ParkCfg9 = So101ParkProbeConfig[69, 60, 9]
# ⚠⚠ 12 AND 13 CROSS THE `Je` SPILL BOUNDARY, AND THAT IS NOT A COST OF THE
# DOFS. P4 made them compile by budgeting the kernel's TOTAL threadgroup
# footprint instead of `Je` alone; the consequence is that k<=9 keeps `Je` in
# shared memory and k>=10 re-reads it from GLOBAL on every Newton iteration. A
# `x k=0` column drawn straight across that boundary charges the slot count for
# a change of code path — the shape recorded as
# `feedback_the_gates_name_named_the_wrong_axis`. Read 0..9 and 12..13 as two
# curves, and say which side of the boundary any quoted ratio came from.
comptime ParkCfg12 = So101ParkProbeConfig[90, 78, 12]
comptime ParkCfg13 = So101ParkProbeConfig[97, 84, 13]


def run_leg[
    MODEL: ModelDefLike, CONFIG: Phyics3dEnvConfig
](ctx: DeviceContext, k: Int, nv: Int) raises:
    """Warm up, then time `TIMED_STEPS` steps. Mirrors the budget probe's
    `time_once` so the two are the same measurement."""
    comptime EnvT = Phyics3dBatchedEnv[MODEL, CONFIG, N_ENVS]
    var env = EnvT(ctx)
    env.reset_batch[N_ENVS](ctx, UInt64(42))

    # Warm until BOTH floors are met. The sync is inside the loop because an
    # un-synced enqueue measures host time, not GPU time, and the whole point
    # here is elapsed GPU work.
    var warm = 0
    var t_warm = perf_counter_ns()
    while True:
        for _ in range(50):
            comptime if RESET_EVERY_STEP:
                env.reset_batch[N_ENVS](ctx, RESET_SEED)
            env.step_batch[N_ENVS](ctx, UInt64(0))
        ctx.synchronize()
        warm += 50
        if (
            warm >= WARMUP_STEPS
            and Float64(perf_counter_ns() - t_warm) / 1e9 >= WARMUP_SECONDS
        ):
            break

    var t0 = perf_counter_ns()
    for _ in range(TIMED_STEPS):
        comptime if RESET_EVERY_STEP:
            env.reset_batch[N_ENVS](ctx, RESET_SEED)
        env.step_batch[N_ENVS](ctx, UInt64(0))
    # The sync is inside the timed region: it is what makes the number mean
    # "the work finished" rather than "the work was enqueued".
    ctx.synchronize()
    var secs = Float64(perf_counter_ns() - t0) / 1e9

    # ⚠⚠ THE WARMUP LAUNCHES ARE IN THE nsys TOTALS AND THE TIMED ONES ARE NOT
    # SEPARABLE FROM THEM. That is why the parser works from Avg-per-launch and
    # this header prints BOTH counts: `total_steps` is what the instance counts
    # divide by, `timed_steps` is what the wall time divides by. Getting those
    # two the wrong way round silently rescales every term at once.
    print("=== SO-101 park attribution probe ===")
    print("  k               ", k)
    print("  nv              ", nv)
    print("  n_envs          ", N_ENVS)
    # ⚠⚠ THE ACTUAL WARMUP COUNT, NOT THE COMPTIME FLOOR. `nsys` counts every
    # launch, so `total_steps` is the divisor the parser turns instance counts
    # into launches/step with. Printing the constant while the loop ran a
    # different number would make every per-step figure in the table wrong by
    # that ratio — silently, and uniformly, which is the hardest kind to spot.
    print("  warmup_steps    ", warm)
    print("  timed_steps     ", TIMED_STEPS)
    comptime if RESET_EVERY_STEP:
        print("  reset_every_step TRUE  <- state pinned; absolutes NOT comparable")
        print("                          to a normal sweep, only arm-to-arm")
    print("  total_steps     ", warm + TIMED_STEPS)
    print("  wall_s          ", secs)
    print("  ms_per_step     ", secs * 1000.0 / Float64(TIMED_STEPS))
    print("=== done ===")


def main() raises:
    # argv: [prog, k]   k in {0, 3, 6, 9}
    if len(argv()) < 2:
        print(
            "usage: mojo run -I . examples/so101/so101_park_attrib_probe.mojo"
            " <0|3|6|9|12|13>"
        )
        return
    var k = Int(atol(String(argv()[1])))
    var ctx = DeviceContext()

    # ⚠ A RUNTIME SWITCH OVER COMPTIME INSTANTIATIONS. Every arm is compiled;
    # only one runs. That is the point — the process contains one leg's kernel
    # LAUNCHES even though it contains four legs' code.
    if k == 0:
        run_leg[SoArm101ParkK0Model, ParkCfg0](ctx, 0, 6)
    elif k == 3:
        run_leg[SoArm101ParkK3Model, ParkCfg3](ctx, 3, 24)
    elif k == 6:
        run_leg[SoArm101ParkK6Model, ParkCfg6](ctx, 6, 42)
    elif k == 9:
        run_leg[SoArm101ParkK9Model, ParkCfg9](ctx, 9, 60)
    elif k == 12:
        run_leg[SoArm101ParkK12Model, ParkCfg12](ctx, 12, 78)
    elif k == 13:
        run_leg[SoArm101ParkK13Model, ParkCfg13](ctx, 13, 84)
    else:
        # ⚠ NOT A DEFAULT. A silent fallback to k=0 would produce a full,
        # plausible, wrong table. (k=12 used to belong here because it did
        # not compile; P4 fixed that and it is a real leg now.)
        print("unknown k:", k, "- expected one of 0, 3, 6, 9, 12, 13")
