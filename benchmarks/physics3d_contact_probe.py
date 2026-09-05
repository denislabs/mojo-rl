"""MuJoCo twin of `benchmarks/physics3d_cpu/contact_probe.mojo`: same protocol,
same line format, contacts as sorted BODY pairs.

    pixi run python benchmarks/physics3d_contact_probe.py <xml> [steps] [pose_file]
"""
import sys
import mujoco

sys.path.insert(0, "benchmarks")
from physics3d_cpu_vs_mujoco_pose import load_pose, apply_pose  # noqa: E402

m = mujoco.MjModel.from_xml_path(sys.argv[1])
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 500
pose = load_pose(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
if pose is not None:
    for b, p, q in pose["bodies"]:
        if 0 < b < m.nbody and m.body_jntnum[b] == 0:
            m.body_pos[b] = p
            m.body_quat[b] = q
d = mujoco.MjData(m)
apply_pose(m, d, pose)
d.ctrl[:] = 0.1
for n in range(steps):
    mujoco.mj_step(m, d)
    keys = []
    for c in range(d.ncon):
        a = int(m.geom_bodyid[d.contact[c].geom1]); b = int(m.geom_bodyid[d.contact[c].geom2])
        keys.append((min(a, b), max(a, b)))
    keys.sort()
    qsum = sum(float(d.qpos[i]) * (i + 1) for i in range(m.nq))
    print(f"STEP {n} ncon={d.ncon} qsum={qsum} pairs=" + ",".join(f"{a}-{b}" for a, b in keys))
