"""What the LIBERO demo gates share: the asset rewrite and the joint remap.

Two gates read the same HDF5 files and ask different questions of them —
`libero_demo_success.py` (does our goal fire where LIBERO says it is solved?)
and `libero_camera_gate.py` (do our pixels look like LIBERO's?). Both need the
recorded model's paths pointed at THIS tree, and both need a recorded state
rewritten into OUR joint order.

⚠⚠ THE JOINT REMAP IS WHY THIS FILE EXISTS. robosuite merges the robot first
and then the objects in `:objects` order; our composer orders by SLOT. So the
two models agree on nq and nv and disagree on which address is which joint,
and a state copied straight across is a scene with the arm's angles in the
bowl's free joint. Written twice it would drift — the shape this tree keeps
paying for (`_a_rule_written_inline_twice_drifts`) — so it is written once,
here, and both gates import it.

⚠ `inertiagrouprange="0 0"` IS RESTORED ON THE RECORDED XML. `model_file`
records the COMPILED model, and MuJoCo's writer drops the compiler option that
kept the visual meshes out of the inertia — so a straight reload gives every
LIBERO object a different mass from the one its demonstration was recorded
with.
"""

import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEMOS = os.path.join(ROOT, "references", "libero_demos")
SCENES = os.path.join(ROOT, "noeira", "tasks", "scenes")
RS14 = os.environ.get(
    "ROBOSUITE_140", os.path.join(ROOT, "references", "robosuite-1.4.0", "robosuite")
)
PACK = os.path.join(ROOT, "noeira", "tasks", "libero", "assets")


def rewrite(xml):
    """The recorded `model_file`, with this machine's paths and the compiler
    option MuJoCo's writer dropped."""
    xml = xml.replace("/Users/yifengz/workspace/robosuite-master/robosuite", RS14)
    xml = xml.replace("/Users/yifengz/workspace/libero-dev/chiliocosm/assets", PACK)
    return xml.replace(
        '<compiler angle="radian" meshdir="meshes/" autolimits="true"/>',
        '<compiler angle="radian" meshdir="meshes/" autolimits="true" inertiagrouprange="0 0"/>',
        1,
    )


def ours(name):
    """Their joint name -> ours. robosuite prefixes the arm `robot0_` and the
    hand `gripper0_`; our composer gives both the robot's slot name."""
    for p in ("robot0_", "gripper0_"):
        if name.startswith(p):
            return "robot_" + name[len(p):]
    return name


def joint_names(m):
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]


def body_names(m):
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]


def build_remap(mt, mo):
    """`[(their_qadr, our_qadr, nq_j, their_vadr, our_vadr, nv_j)]`, by NAME.

    Raises rather than guessing on a joint one model has and the other does
    not — a silently dropped joint is a state that loads and is wrong.
    """
    their_j = joint_names(mt)
    our_j = joint_names(mo)
    remap = []
    for j, jn in enumerate(their_j):
        want = ours(jn)
        if want not in our_j:
            raise SystemExit(f"joint {jn!r} -> {want!r} is not in our scene")
        oj = our_j.index(want)
        nqj = (int(mo.jnt_qposadr[oj + 1]) if oj + 1 < mo.njnt else mo.nq) - int(
            mo.jnt_qposadr[oj]
        )
        nvj = (int(mo.jnt_dofadr[oj + 1]) if oj + 1 < mo.njnt else mo.nv) - int(
            mo.jnt_dofadr[oj]
        )
        remap.append(
            (
                int(mt.jnt_qposadr[j]),
                int(mo.jnt_qposadr[oj]),
                nqj,
                int(mt.jnt_dofadr[j]),
                int(mo.jnt_dofadr[oj]),
                nvj,
            )
        )
    return remap


def convert_state(row, remap, mt, mo):
    """One flattened `states` row (time, qpos, qvel) -> our `(qpos, qvel)`."""
    qt = row[1 : 1 + mt.nq]
    vt = row[1 + mt.nq :]
    qo = np.zeros(mo.nq)
    vo = np.zeros(mo.nv)
    for ta, oa, nqj, tv, ov, nvj in remap:
        qo[oa : oa + nqj] = qt[ta : ta + nqj]
        vo[ov : ov + nvj] = vt[tv : tv + nvj]
    return qo, vo


def fixture_poses(mt, our_b):
    """This demo's own static placement: `[(our_body, pos, quat)]`.

    LIBERO redraws every fixture per episode and bakes the draw into
    `body_pos`/`body_quat` of the recorded XML, so a gate that loads our
    family's scene has the WRONG fixtures until it patches these in.
    """
    out = []
    for i, bn in enumerate(body_names(mt)):
        if not bn.endswith("_main") or mt.body_parentid[i] != 0 or mt.body_jntnum[i] != 0:
            continue
        ob = bn[:-5] + "_object"
        if ob not in our_b:
            raise SystemExit(f"no body {ob} in our scene")
        out.append((ob, np.array(mt.body_pos[i]), np.array(mt.body_quat[i])))
    return out
