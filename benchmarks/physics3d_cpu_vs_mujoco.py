"""MuJoCo twin of `benchmarks/physics3d_cpu/harness.mojo` — the same protocol
on the same XML, so the two `RESULT` lines are one row of one table.

    pixi run python benchmarks/physics3d_cpu_vs_mujoco.py <name> <xml> [warmup] [steps] [rounds]

Protocol (identical on both sides): `mj_resetData` (qpos0, qvel 0, warmstart
0), `ctrl = 0.1` on every actuator, WARMUP untimed steps, a short counting
loop for the work counters OUTSIDE the timed region, then STEPS steps timed
with one clock read on each side.

⚠ `bvactive = 0`. Stock MuJoCo memsets its whole BVH-active array every step
for the VISUALISER, and it is on by default; on mesh-heavy scenes that is up
to half the reference's step (`physics3d_mujoco_phases.py` has the numbers).
Our engine has no counterpart, so a comparison with it on is physics against
physics-plus-a-memset.

⚠ THE TIMED LOOP IS ONE CALL. `mj_step(m, d, nstep)` runs the whole loop in
C; a Python `for` around `mj_step` adds ~0.3 us of interpreter per step, which
on a 2 us model is 15% of the number.
"""
import sys
import time

import numpy as np
import mujoco


sys.path.insert(0, "benchmarks")
from physics3d_cpu_vs_mujoco_pose import load_pose, apply_pose  # noqa: E402

name = sys.argv[1]
xml = sys.argv[2]
warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 1
# Optional 6th arg: a pose file written by the Mojo harness's `TASK_POSE`
# rows (`QPOS`, `QVEL`, `BODY id pos quat` lines). Body poses are applied to
# JOINTLESS bodies only — that is how a task reset moves a welded prop.
pose = load_pose(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] else None
COUNT_STEPS = 2000

m = mujoco.MjModel.from_xml_path(xml)
m.vis.global_.bvactive = 0
if pose is not None:
    for b, p, q in pose["bodies"]:
        if 0 < b < m.nbody and m.body_jntnum[b] == 0:
            m.body_pos[b] = p
            m.body_quat[b] = q
d = mujoco.MjData(m)

# `rounds` > 1 repeats the whole protocol from `mj_resetData` and reports the
# MIN -- a round is a reset, not a continuation (see the Mojo harness).
us = float("inf")
for _ in range(rounds):
    apply_pose(m, d, pose)
    d.ctrl[:] = 0.1
    mujoco.mj_step(m, d, warmup)

    ncon = []
    nefc = []
    niter = []
    for _ in range(min(steps, COUNT_STEPS)):
        mujoco.mj_step(m, d)
        ncon.append(d.ncon)
        nefc.append(d.nefc)
        niter.append(int(d.solver_niter[0]))

    t0 = time.perf_counter()
    mujoco.mj_step(m, d, steps)
    t1 = time.perf_counter()
    us = min(us, 1e6 * (t1 - t0) / steps)

integ = mujoco.mjtIntegrator(m.opt.integrator).name[6:].lower()
solver = mujoco.mjtSolver(m.opt.solver).name[6:].lower()
print(
    f"RESULT side=mujoco model={name} dtype=f64 integ={integ} solver={solver}"
    f" nq={m.nq} nv={m.nv} us_per_step={us:.4f}"
    f" ncon_mean={np.mean(ncon):.3f} nefc_mean={np.mean(nefc):.3f}"
    f" niter_mean={np.mean(niter):.3f} steps={steps} rounds={rounds}"
    f" qpos0={d.qpos[0]} qpos1={d.qpos[1]}"
    f" timestep={m.opt.timestep} version={mujoco.__version__}"
)
