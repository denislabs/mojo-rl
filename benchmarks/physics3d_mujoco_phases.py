"""MuJoCo per-phase cost + work counts — the reference side of PERFORMANCE.md.

    pixi run python benchmarks/physics3d_mujoco_phases.py <scene.xml> [N] [key] [bvactive] [warmup]

Answers: of MuJoCo's step, how much is collision (broad / mid / narrow) vs
constraint build vs solve vs dynamics, and how much WORK it does getting there
(ncon, nefc, solver iterations). `mojo_rl/physics3d/PERFORMANCE.md` compares
our own phase profile against the output of this script.

⚠ THE TIMER UNIT IS CALIBRATED, NOT ASSUMED. `mjTimerStat.duration` is
documented as milliseconds, but through the Python bindings the value that
comes back is ~1e6 times smaller than the wall-clock truth. Rather than guess
the unit, this wraps the same loop in `perf_counter` and derives the scale from
mjTIMER_STEP. If MuJoCo ever fixes or changes the unit, this keeps working.

⚠ THERE IS NO `mju_clearTimers` IN THE PYTHON BINDINGS, and the counters
accumulate across every step since reset — including the warmup. They are
zeroed field by field below; dropping that makes every phase percentage a
blend of warmup and steady state.

⚠ `mjtTimer` IS NOT ITERABLE (pybind11 enum) — use `.__members__`.
"""
import sys
import time
import numpy as np
import mujoco

XML = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
KEY = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
if KEY is not None:
    mujoco.mj_resetDataKeyframe(m, d, m.key(KEY).id)
else:
    mujoco.mj_resetData(m, d)

print(f"model  nq={m.nq} nv={m.nv} nbody={m.nbody} ngeom={m.ngeom} "
      f"nmesh={m.nmesh} nmeshvert={m.nmeshvert}")
print(f"opt    solver={mujoco.mjtSolver(m.opt.solver).name} "
      f"iterations={m.opt.iterations} cone={mujoco.mjtCone(m.opt.cone).name} "
      f"integrator={mujoco.mjtIntegrator(m.opt.integrator).name} "
      f"timestep={m.opt.timestep}")

# ⚠⚠ TURN OFF `bvactive` OR MOST OF WHAT YOU MEASURE IS A DEBUG MEMSET.
# `mj_collision` opens with `memset(d->bvh_active, 0, m->nbvh)` whenever
# `m->vis.global.bvactive` is set -- and it is set BY DEFAULT. `nbvh` counts
# every node of every MESH BVH, so it is enormous on mesh-heavy scenes:
# 696 364 on robotstudio_so101, which is 6.17 us/step of pure memset --
# **51% of that model's step**. Measured on M1 Pro at ~115 GB/s, i.e. exactly
# memset bandwidth for that byte count. It is a VISUALISATION feature; nothing
# in the dynamics reads `bvh_active`, and our engine has no counterpart, so
# leaving it on compares our physics against MuJoCo's physics plus a memset.
# ⚠ EVERY MuJoCo NUMBER PUBLISHED IN `PERFORMANCE.md` BEFORE 2026-08-14 HAD
# THIS ON, which flattered our ratios -- badly on so101 (1.78x -> 4.16x).
BVACTIVE = int(sys.argv[4]) if len(sys.argv) > 4 else 0
# Optional: a shorter warmup / counting loop for a scene that changes character
# after a known step (the SO-101 park scenes drop their props at step 1596).
WARMUP = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
m.vis.global_.bvactive = BVACTIVE
print(f"vis    bvactive={BVACTIVE}  nbvh={m.nbvh} "
      f"({m.nbvh/1e3:.0f} kB memset/step if on)")

# The env drives every actuator with a constant; mirror that.
d.ctrl[:] = 0.1

# Warm up (page faults, first-touch, any lazily built tables).
for _ in range(WARMUP):
    mujoco.mj_step(m, d)

TIMERS = {n: t.value for n, t in mujoco.mjtTimer.__members__.items()
          if n.startswith("mjTIMER")}
# Work counts come from their OWN loop. Reading d.ncon/d.nefc through pybind11
# costs ~0.4 us/step, and inside the timed loop that inflates `wall_us`, hence
# SCALE, hence EVERY phase number below by 3-5%.
ncon = []
nefc = []
niter = []
for _ in range(min(N, WARMUP)):
    mujoco.mj_step(m, d)
    ncon.append(d.ncon)
    nefc.append(d.nefc)
    niter.append(int(d.solver_niter[0]))

for v in TIMERS.values():
    d.timer[v].duration = 0
    d.timer[v].number = 0
t0 = time.perf_counter()
for _ in range(N):
    mujoco.mj_step(m, d)

wall_us = 1e6 * (time.perf_counter() - t0) / N
T = {n: (d.timer[v].duration, d.timer[v].number) for n, v in TIMERS.items()}
raw = T["mjTIMER_STEP"][0] / max(T["mjTIMER_STEP"][1], 1)
print(f"\nWALL   {wall_us:.3f} us/step (perf_counter)")
print(f"timer  raw STEP {raw:.9f} per step -> wall/raw = {wall_us/max(raw,1e-30):.1f}")
SCALE = wall_us / max(raw, 1e-30)  # calibrate the timer unit against wall clock

step_us = SCALE * raw
print(f"\nSTEP   {step_us*1.0:.3f} us/step  over {T['mjTIMER_STEP'][1]} steps")
print(f"work   ncon mean {np.mean(ncon):.2f} max {max(ncon)}   "
      f"nefc mean {np.mean(nefc):.2f} max {max(nefc)}   "
      f"solver_niter mean {np.mean(niter):.2f} max {max(niter)}")

order = [
    "mjTIMER_POS_KINEMATICS", "mjTIMER_POS_INERTIA", "mjTIMER_POS_COLLISION",
    "mjTIMER_COL_BROAD", "mjTIMER_COL_NARROW", "mjTIMER_POS_MAKE",
    "mjTIMER_POS_PROJECT", "mjTIMER_POSITION", "mjTIMER_VELOCITY",
    "mjTIMER_ACTUATION", "mjTIMER_CONSTRAINT", "mjTIMER_FORWARD",
    "mjTIMER_ADVANCE",
]
print("\n  phase                     us/step    % of step   calls/step")
for k in order:
    dur, num = T[k]
    if num == 0:
        continue
    per_step = SCALE * dur / max(T["mjTIMER_STEP"][1], 1)
    print(f"  {k[8:]:<22} {per_step:8.4f}   {100*per_step/step_us:7.2f}%"
          f"   {num/max(T['mjTIMER_STEP'][1],1):8.2f}")
