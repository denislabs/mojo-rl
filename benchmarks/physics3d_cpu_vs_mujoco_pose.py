"""The pose file the Mojo harness writes for a `TASK_POSE` row, read for MuJoCo.

`QPOS ...`, `QVEL ...`, then `BODY id px py pz qw qx qy qz` per body. Callers
apply the body poses to JOINTLESS bodies only (`m.body_jntnum[b] == 0`): that
is how a task reset moves a welded prop, and a jointed body's pose lives in
`qpos` already.
"""
import mujoco


def load_pose(path):
    out = {"qpos": None, "qvel": None, "bodies": []}
    for ln in open(path):
        t = ln.split()
        if not t:
            continue
        if t[0] == "QPOS":
            out["qpos"] = [float(x) for x in t[1:]]
        elif t[0] == "QVEL":
            out["qvel"] = [float(x) for x in t[1:]]
        elif t[0] == "BODY":
            out["bodies"].append((int(t[1]), [float(x) for x in t[2:5]], [float(x) for x in t[5:9]]))
    return out


def apply_pose(m, d, pose):
    mujoco.mj_resetData(m, d)
    if pose is not None:
        d.qpos[:] = pose["qpos"]
        d.qvel[:] = pose["qvel"]
