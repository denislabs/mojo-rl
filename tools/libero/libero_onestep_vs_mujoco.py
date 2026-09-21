#!/usr/bin/env python3
"""The one-step board row for a composed LIBERO family — L2's fidelity gate.

    pixi run mojo build -I . docs/menagerie_fidelity_harnesses/drive.mojo -o /tmp/mfh/drive
    pixi run python tools/libero/libero_onestep_vs_mujoco.py noeira/envs/libero/scenes/libero_goal.xml \
        noeira/envs/libero/families/libero_goal.family [N] [ctrl_scale]

Same protocol as `docs/menagerie_fidelity_harnesses/full.py` (both engines
step the SAME random ctrl sequence from the SAME state; max |d| of qacc /
qvel / qpos), with one difference that this family needs: the start state is
the family's `base_qpos=` (LIBERO's Panda init pose) on the first joints and
qpos0 elsewhere, handed to the probe through `DRIVE_STATE=`, because the
composed scene carries no keyframe and a Panda at all-zeros is folded onto
itself.

⚠⚠ THE QACC COLUMN IS NOT COMPARABLE ON A DAMPED JOINT, AND THE FINGERS ARE
DAMPED (100). MuJoCo's Euler integrator applies joint damping IMPLICITLY:
`mj_step` reports the EXPLICIT `qacc = M^-1 f` and then integrates with
`(M + h D)`; the probe reports the acceleration it actually integrated. On
the Panda gripper that is 17.27 vs 14.62 with IDENTICAL qvel and qpos after
the step (`scratch finger ablation, 2026-09-13`: "no damping" -> exact, every
other knob leaves the gap). Read QVEL and QPOS; QACC is printed for the
undamped dofs' sake.

⚠ The number this prints is a BOARD ROW, to be read against the board
(`docs/PHYSICS3D_CONTACT_FIDELITY_REASSEMBLE5.md` §13): the one-step rows
there are EXACT to ~1e-12 on 85/85 Menagerie scenes; an elliptic-cone scene
with 0 contacts at rest should sit there too. A row that does not is a
finding, not a tolerance to widen.
"""
import os
import subprocess
import sys

import mujoco
import numpy as np

S = os.environ.get("MFH_BIN", "/tmp/mfh")
scene = sys.argv[1]
fam = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 1
SC = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4

base_qpos = []
for line in open(fam):
    line = line.strip()
    if line.startswith("base_qpos="):
        base_qpos = [float(x) for x in line.split("=", 1)[1].split(",")]

m = mujoco.MjModel.from_xml_path(scene)
d = mujoco.MjData(m)
for j, q in enumerate(base_qpos):
    d.qpos[m.jnt_qposadr[j]] = q
mujoco.mj_forward(m, d)
state_path = os.path.join(S, "libero_state.txt")
with open(state_path, "w") as f:
    f.write(" ".join(repr(float(x)) for x in d.qpos) + "\n")
    f.write(" ".join(repr(float(x)) for x in d.qvel) + "\n")

rng = np.random.default_rng(12345)
U = (rng.random((N, max(m.nu, 1))) * 2.0 - 1.0) * SC
ctrl_path = os.path.join(S, "libero_ctrl.txt")
with open(ctrl_path, "w") as f:
    f.write("\n".join(" ".join(repr(float(x)) for x in U[t]) for t in range(N)) + "\n")

ncon0 = d.ncon
for t in range(N):
    if m.nu:
        d.ctrl[:] = U[t]
    mujoco.mj_step(m, d)

env = dict(os.environ, DRIVE_STATE=state_path)
out = subprocess.run([os.path.join(S, "drive"), scene, ctrl_path, str(N), "-1"],
                     capture_output=True, text=True, env=env)
if out.returncode != 0:
    print(out.stdout[-2000:])
    print(out.stderr[-2000:])
    sys.exit("drive probe failed")
G = {}
for l in out.stdout.split("\n"):
    for k in ("QFRC0", "QPOS", "QACC", "QVEL"):
        if l.startswith(k + " "):
            G[k] = np.array([float(x) for x in l[len(k) + 1:].split()])

print(f"--- {os.path.basename(scene)}  N={N} scale={SC} base_qpos={len(base_qpos)} joints ---   "
      f"mj ncon at start {ncon0}, after {d.ncon}, nefc {d.nefc}")
worst = 0.0
for k, ref in (("QACC", np.array(d.qacc)), ("QVEL", np.array(d.qvel)), ("QPOS", np.array(d.qpos))):
    o = G.get(k)
    if o is None or o.shape != ref.shape:
        print(f"  {k}: shape/none")
        worst = float("inf")
        continue
    dd = np.abs(o - ref)
    if k == "QACC":
        damped = m.dof_damping > 0
        dd = np.where(damped, 0.0, dd)
    if os.environ.get("ONESTEP_ALL"):
        adr = m.jnt_qposadr if k == "QPOS" else m.jnt_dofadr
        for q in np.argsort(-dd)[:12]:
            if dd[q] > 1e-10:
                jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(np.searchsorted(adr, q, side="right") - 1))
                print(f"      {k} dof {q:>3} {jn:<32} |d| {dd[q]:.3e}  ours {o[q]:+.6e}  mj {ref[q]:+.6e}")
    i = int(np.argmax(dd))
    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(np.searchsorted(m.jnt_dofadr if k != "QPOS" else m.jnt_qposadr, i, side="right") - 1))
    print(f"  {k:<5} max|d| {dd.max():.4e} @ {i:>3} ({jname})   ours {o[i]:+.8f}  mj {ref[i]:+.8f}")
    worst = max(worst, float(dd.max()))
print(f"worst {worst:.4e}   (QACC excluded on damped dofs — see the header)")
