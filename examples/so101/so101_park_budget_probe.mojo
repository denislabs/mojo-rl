"""P0 — what does a PARKED SLOT cost? The scene-budget probe.

`docs/TASK_LAYER_PLAN.md` §3 rests on a fixed scene budget: a task family
declares every slot it could ever use, and a task that does not use one parks
it far away. §3.3 marks the cost of that UNPRICED and says, in as many words,
**do not commit to a budget of 6 before P0 reports**. This is P0.

    pixi run -e nvidia mojo run -I . examples/so101/so101_park_budget_probe.mojo

⚠⚠ RUN IT ON NVIDIA. Two separate reasons, and the second is hard.

1. A number from Apple Silicon is not the answer, and this tree has already
   retracted one: `sac_so_arm101_reach_profile_gpu.mojo` records a 24k-step
   Apple run at 27.6 env-steps/s whose extrapolation was quoted in
   conversation and then withdrawn. Metal is the slow path for this engine.

2. **APPLE CANNOT RUN THIS PROBE AT ALL BEYOND k=0.** Measured 2026-09-02:
   k=0 (nv 6) builds and steps; k=3 (nv 24) fails at pipeline creation with

       Failed to create compute pipeline state (GPU machine code generation):
       Compute function exceeds available stack space

   The physics kernels stack-allocate per-thread arrays sized by `nv`, and
   nv=24 WITH mesh collision is past Metal's per-thread stack. `N_ENVS` moves
   the threshold too — k=0 builds at 32 lanes and fails at 64. This is the
   Metal cliff `phyics3d_env_config.NMESH_VERTS` already warns about, reached
   from the other side. It is not a bug in this probe and there is nothing to
   fix here: the budget question is an NVIDIA question.

   So on Apple this file is good for exactly one thing — checking that the
   harness compiles, steps, formats and refuses correctly.

## ⚠⚠ IT HAS RUN. THE ANSWER IS: A PARKED SLOT IS EXPENSIVE, QUADRATICALLY.

RTX 5090, 1024 lanes, 300 timed steps, 3 interleaved repeats, minimum reported.
Run 2026-09-02. Run-to-run spread 0.04–2.02 % against a 687 % effect, so this
is decidable by a wide margin.

    leg                  nq   nv   ms/step   env-steps/s   x k=0   spread
    k=0  (control)        6    6      7.94      128,908    1.00    0.04%
    k=3                  27   24     13.55       75,573    1.71    0.52%
    k=6                  48   42     29.07       35,224    3.66    0.71%
    k=9  (ceiling)       69   60     62.55       16,371    7.88    2.02%
    k=9  REPARK          69   60     62.62       16,353    7.89    1.66%

⚠ THE EXCESS IS QUADRATIC IN THE ADDED DOFS, not linear:

    k=3   d_nv 18   +5.61 ms   d/d_nv^2 = 0.0173
    k=6   d_nv 36  +21.13 ms   d/d_nv^2 = 0.0163
    k=9   d_nv 54  +54.61 ms   d/d_nv^2 = 0.0187

constant to +-7 %. That is the mass-matrix factorisation and the cooperative
Cholesky, both ~NV^2 per thread — exactly the term `docs/TASK_LAYER_PLAN.md`
§3.3 suspected and could not price.

⚠⚠ WHAT THIS MEANS FOR THE PLAN. §3.3 said "this is not obviously free" and
"do not commit to a budget of 6 before P0 reports". P0 has reported: **a budget
of 6 costs 3.66x the physics step.** The fixed scene budget, as specified —
free-jointed slots for every object a family might use — does not survive at
the sizes the design assumes. §3.6's answer to LIBERO-Object ("declare the
union, park the unused", ten free bodies with nine parked) is past both the
compile ceiling and any usable throughput.

⚠ AND REPARK IS FREE — 62.62 vs 62.55 ms, 0.11 %, inside the 2 % spread. That
is not just a happy result for Gap D, it LOCATES THE COST: pinning the pose
removes the broadphase churn and the falling bodies and changes nothing, so the
cost is in the DYNAMICS, not in collision. **You cannot park your way out of
it.** The dofs are expensive because they exist, not because they move.

⚠ THIS MEASURES THE PHYSICS STEP ALONE, deliberately — no agent. Whether 3.66x
on the step is 3.66x on a training run depends on the env:agent ratio at 1024
lanes, which is not measured here and should be before anyone quotes this as a
training slowdown.

## ⚠⚠ THE SWEEP STOPS AT k=9 BECAUSE THE SOLVER DOES

k=12 was in this sweep and DOES NOT COMPILE on an RTX 5090:

    ptxas error : Entry function 'mojo_rl_physics3d_solver_newt...' uses
                  too much shared data (0x21414 bytes, 0x18c00 max)

The GPU Newton solver holds three NV*NV matrices (M, H, L) plus `Je` (ME*NV) in
threadgroup memory. Measured, fp32, MAX_CONTACTS=16, and the formula reproduces
ptxas's number TO THE BYTE:

    k=6   nv 42   48,372 B   fits
    k=9   nv 60   86,676 B   fits      <- the ceiling
    k=10  nv 66  101,940 B   over by 564 B
    k=12  nv 78  136,212 B   over      == 0x21414, exactly what ptxas said

**So a family on this hardware cannot declare more than 9 free-jointed slots**,
and that is a budget answer P0 never had to measure — it falls out of the
solver, not out of a throughput curve. ⚠ It is also DEVICE-DEPENDENT: an H100's
227 KB would allow more. A budget the design calls FIXED is in fact fixed *per
GPU*, which §3 does not currently account for.

⚠ `solver/je_budget.je_spills` DOES NOT CATCH THIS. It compares **Je alone**
against a 64 KB constant; at k=12 Je is 54 KB so it declines to spill, while
the three NV*NV matrices (73 KB) put the block over. The gate budgets one array
rather than the total, and it was tuned on humanoid_CMU and dog — high-nv AND
high-contact models. A fixed scene budget produces the shape it was never tuned
for: HIGH nv, LOW contact count. Fixing it would let k=12 compile with Je
spilled — but then the widest point would run a DIFFERENT SOLVER PATH from the
control, and leg 1 would be comparing two code paths while calling the
difference a budget cost. One path across the sweep is what makes it a curve.

## THE THREE LEGS, AND WHY ONE IS NOT ENOUGH

A parked free body adds `nq`/`nv` (the CRBA + factorisation cost the plan
names) — but it ALSO adds a geom, hence a broadphase entry, and it would add
contacts if it touched anything. `max_contacts` bounds the PGS/Newton solve,
which is superlinear in ACTIVE contacts, a quantity with nothing to do with
`nv`. A single sweep of k with those free to move would attribute solver cost
to `nv` and the budget number would be wrong in an unknown direction — the
shape recorded as `feedback_the_gates_name_named_the_wrong_axis`, whose fix is
to ADD THE FIXED-AXIS LEG, not to reinterpret the mixed one.

  LEG 1  slot count k in {0, 3, 6, 9}, `max_contacts` PINNED at 16 for every
         k (`so101_park_xml.PARK_MAX_CONTACTS`), and every scene verified to
         have ZERO contacts at rest by the scene generator. The curve.
  LEG 2  `max_contacts` alone, at k=0. How much of leg 1 is the solver rather
         than `nv`. Without this, leg 1 is undecidable.
  LEG 3  the repark hook on/off at k=9. What keeping a parked slot ACTUALLY
         parked costs — see `so101_park_config.mojo`, and note it is a lower
         bound because the hook cannot zero velocity yet.

⚠ LEG 2 IS NOT WIRED HERE. `max_contacts` is a comptime parameter of
`ModelDefFromXML`, so a second value means a second monomorphisation of the
whole batched env; sweeping it needs its own model defs in
`so101_park_xml.mojo`. It is deliberately left until leg 1 has run, because
leg 1's shape decides whether leg 2 needs three points or two — and because
four mesh-collision monomorphisations is already a large compile. **Leg 1 and
leg 3 do not answer the budget question on their own. Do not read a budget off
this until leg 2 exists.**

## HOW IT MEASURES

* **No agent.** `reset_batch` once, then `step_batch` in a loop with the
  action buffer left at zero. Anything else would put network kernels in the
  timed region.
* **Never resets inside a timed region.** `MAX_STEPS` is a million, so no lane
  truncates mid-run. A reset is far more expensive than a step and one landing
  inside a repeat would show up as a fluke.
* **Synchronise around the timed region, and NOWHERE INSIDE IT.** The step
  enqueues asynchronously; a host timer that spans un-synchronised enqueues is
  measuring the enqueue, not the work. Equally, a `synchronize()` per step
  would measure launch latency instead of throughput.
* **REPEATS runs per leg, interleaved across legs, and the MINIMUM reported.**
  A baseline taken earlier in a session drifts (thermal, clocks, other
  tenants), so the legs are interleaved rather than run to completion one at a
  time. Single-row flukes of +8..54% are recorded on this hardware, and an n=1
  sweep here produced both a fake win and a fake regression — so REPEATS is 3
  and the spread is printed beside the minimum. **If the spread is wider than
  the effect, there is no effect.**
"""

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
    PARK_MAX_CONTACTS,
)
from mojo_rl.core.fmt import fit


# ─── knobs ────────────────────────────────────────────────────────────────
# ⚠ 1024 IS THE LANE COUNT THE PLAN SPECIFIES, and the budget question is a
# throughput question at scale — a 32-lane answer would be launch-bound and
# would flatter every k equally.
comptime N_ENVS = 1024

comptime WARMUP_STEPS = 50    # JIT, first-touch allocation, clock ramp
comptime TIMED_STEPS = 300    # one horizon's worth
comptime REPEATS = 3          # never 1; see the module docstring


comptime ParkCfg0 = So101ParkProbeConfig[6, 6, 0]
comptime ParkCfg3 = So101ParkProbeConfig[27, 24, 3]
comptime ParkCfg6 = So101ParkProbeConfig[48, 42, 6]
comptime ParkCfg9 = So101ParkProbeConfig[69, 60, 9]
comptime ParkCfg9R = So101ParkProbeConfig[69, 60, 9, REPARK=True]


def f2(x: Float64) -> String:
    """`x` to two decimals. Mojo's `String(Float64)` prints full precision,
    which in a fixed-width table truncates mid-number and reads as garbage
    (`1419.5219759740.0%` was one column and a half)."""
    var r = Float64(Int(x * 100.0 + (0.5 if x >= 0.0 else -0.5))) / 100.0
    return String(r)


def col(s: String, n: Int) -> String:
    """`s` truncated to `n` and PADDED to `n`.

    ⚠ `core.fmt.fit` TRUNCATES ONLY. Using it alone ran every column of the
    table together into one unreadable number — a formatting bug, but on a
    table whose whole job is to be read off by a human it is the difference
    between a result and a smear.
    """
    var t = fit(s, n)
    var out = t
    for _ in range(n - t.byte_length()):
        out += " "
    return out^


def time_once[
    MODEL: ModelDefLike, CONFIG: Phyics3dEnvConfig
](ctx: DeviceContext) raises -> Float64:
    """Seconds for TIMED_STEPS batched steps. Builds a fresh env each call.

    ⚠ A FRESH ENV PER REPEAT, DELIBERATELY. Reusing one would let state from
    the previous repeat (fallen slots, warmed caches) leak into the next, and
    the repeats are meant to be independent samples of the same thing.
    """
    comptime EnvT = Phyics3dBatchedEnv[MODEL, CONFIG, N_ENVS]
    var env = EnvT(ctx)
    env.reset_batch[N_ENVS](ctx, UInt64(42))

    for _ in range(WARMUP_STEPS):
        env.step_batch[N_ENVS](ctx, UInt64(0))
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(TIMED_STEPS):
        env.step_batch[N_ENVS](ctx, UInt64(0))
    # ⚠ THE SYNC IS INSIDE THE TIMED REGION ON PURPOSE — it is the only thing
    # that makes the elapsed time mean "the work finished" rather than "the
    # work was enqueued".
    ctx.synchronize()
    return Float64(perf_counter_ns() - t0) / 1e9


struct Row(Copyable, ImplicitlyCopyable, Movable):
    """One leg's samples, so the table can print a spread beside a minimum."""
    var label: String
    var nq: Int
    var nv: Int
    var best: Float64
    var worst: Float64
    # ⚠⚠ THE SAMPLE COUNT IS NOT BOOKKEEPING — IT IS THE VACUITY GUARD.
    # An earlier version of this file had no `n`, and a run in which four of
    # five legs never executed still printed a full table (`best` stuck at its
    # 1e30 sentinel, rendered as `1e+32` ms/step) AND still concluded
    # "OK: the effect is larger than the noise floor" — computed from rows
    # holding no data at all. A measurement harness that reports a verdict on
    # zero samples is worse than one that crashes. `require_samples` below
    # raises instead.
    var n: Int

    def __init__(out self, label: String, nq: Int, nv: Int):
        self.label = label
        self.nq = nq
        self.nv = nv
        self.best = 1e30
        self.worst = 0.0
        self.n = 0

    def add(mut self, secs: Float64):
        self.n += 1
        if secs < self.best:
            self.best = secs
        if secs > self.worst:
            self.worst = secs

    def steps_per_s(self) -> Float64:
        return Float64(TIMED_STEPS * N_ENVS) / self.best

    def ms_per_step(self) -> Float64:
        return self.best * 1000.0 / Float64(TIMED_STEPS)

    def spread_pct(self) -> Float64:
        """(worst - best) / best, as a percent. The decidability check."""
        if self.best <= 0.0:
            return 0.0
        return (self.worst - self.best) / self.best * 100.0


def main() raises:
    print("=== P0: the scene-budget probe — SO-ARM101 + k parked slots ===")
    print("  N_ENVS:", N_ENVS, "| warmup:", WARMUP_STEPS,
          "| timed:", TIMED_STEPS, "| repeats:", REPEATS)
    print("  max_contacts PINNED at", PARK_MAX_CONTACTS, "for every k")
    print("  ⚠ leg 2 (max_contacts alone) is NOT in this run — see the header")
    print()

    with DeviceContext() as ctx:
        var r0 = Row("leg1  k=0  (control)", 6, 6)
        var r3 = Row("leg1  k=3", 27, 24)
        var r6 = Row("leg1  k=6", 48, 42)
        var r9 = Row("leg1  k=9  (ceiling)", 69, 60)
        var r9r = Row("leg3  k=9  REPARK", 69, 60)

        # ⚠ INTERLEAVED, NOT GROUPED. Running all of k=0 then all of k=12
        # compares a cold machine against a hot one and calls the difference
        # a budget cost.
        for rep in range(REPEATS):
            print("  repeat", rep + 1, "of", REPEATS, "...")
            r0.add(time_once[SoArm101ParkK0Model, ParkCfg0](ctx))
            r3.add(time_once[SoArm101ParkK3Model, ParkCfg3](ctx))
            r6.add(time_once[SoArm101ParkK6Model, ParkCfg6](ctx))
            r9.add(time_once[SoArm101ParkK9Model, ParkCfg9](ctx))
            r9r.add(time_once[SoArm101ParkK9Model, ParkCfg9R](ctx))

        # ⚠ BEFORE ANY ARITHMETIC. A leg that did not run has `best` at its
        # sentinel, and every derived number below — ms/step, the ratio to the
        # control, the spread — would be a rendering of that sentinel rather
        # than a measurement.
        var rows_chk = [r0, r3, r6, r9, r9r]
        for i in range(len(rows_chk)):
            if rows_chk[i].n != REPEATS:
                raise Error(
                    "park probe: leg '" + rows_chk[i].label + "' has "
                    + String(rows_chk[i].n) + " samples, expected "
                    + String(REPEATS) + ". A leg did not run, and a table"
                    + " built from that is not a slow result — it is no"
                    + " result wearing one."
                )

        print()
        print("  leg                    nq   nv   ms/step   env-steps/s"
              "   vs k=0   spread")
        print("  " + "-" * 74)
        var base = r0.ms_per_step()
        var rows = [r0, r3, r6, r9, r9r]
        for i in range(len(rows)):
            var r = rows[i]
            var rel = (r.ms_per_step() / base - 1.0) * 100.0
            print(
                "  " + col(r.label, 22)
                + col(String(r.nq), 5)
                + col(String(r.nv), 5)
                + col(f2(r.ms_per_step()), 10)
                + col(f2(r.steps_per_s()), 14)
                + col(f2(rel) + "%", 9)
                + col(f2(r.spread_pct()) + "%", 8)
            )

        print()
        # ⚠ THE DECIDABILITY LINE. A sweep whose run-to-run spread exceeds the
        # effect it is measuring has not measured it, and the honest report is
        # to say so rather than to read a budget off the noise.
        var worst_spread = 0.0
        for i in range(len(rows)):
            if rows[i].spread_pct() > worst_spread:
                worst_spread = rows[i].spread_pct()
        var effect = (r9.ms_per_step() / base - 1.0) * 100.0
        print("  worst run-to-run spread:", f2(worst_spread), "%")
        print("  k=9 effect over control:", f2(effect), "%")
        if worst_spread >= effect:
            print("  ⚠⚠ SPREAD >= EFFECT — THIS RUN IS UNDECIDABLE.")
            print("     Raise REPEATS/TIMED_STEPS, or quiet the machine.")
            print("     Do NOT set a budget from these numbers.")
        else:
            print("  OK: the effect is larger than the noise floor.")
        print()
        print("  ⚠ STILL NOT A BUDGET: leg 2 has not run. See the header.")
        print("=== done ===")
