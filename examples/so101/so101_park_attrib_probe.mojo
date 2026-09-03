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
comptime WARMUP_STEPS = 50
comptime TIMED_STEPS = 300

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

    for _ in range(WARMUP_STEPS):
        env.step_batch[N_ENVS](ctx, UInt64(0))
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(TIMED_STEPS):
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
    print("  warmup_steps    ", WARMUP_STEPS)
    print("  timed_steps     ", TIMED_STEPS)
    print("  total_steps     ", WARMUP_STEPS + TIMED_STEPS)
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
