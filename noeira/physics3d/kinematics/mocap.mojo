"""A mocap body's world pose, for a caller that is not an env.

⚠⚠ `forward_kinematics` DELIBERATELY SKIPS MOCAP BODIES. Their world pose is
an EXTERNAL input — `d.mocap_pos` / `d.mocap_quat` — written by whatever is
driving the target, and the parent-chain FK must not overwrite it between
substeps. `Phyics3dEnv._sync_mocap_to_fields` presets `xpos`/`xquat` from
those buffers before each step, which is the env half of the contract.

⚠⚠ A TOOL THAT IS NOT AN ENV NEVER GETS THAT FAR, AND THE FAILURE IS SILENT.
`Data` allocates `mocap_pos` zeroed, so with nobody to fill it a mocap body's
`xpos` stays (0, 0, 0) — the geoms on it are DRAWN at the world origin, the
ray-pick tests them there, and the selection outline follows. so_arm101's
`target` sphere is the case in this tree: `m.bodies` carries pos
(0.25, 0, 0.2) and `d.xpos` reads (0, 0, 0), with no error anywhere.

MuJoCo does not have this problem because `mj_resetData` SEEDS
`mocap_pos`/`mocap_quat` from `body_pos`/`body_quat` — a mocap body that
nobody drives sits where the XML put it. That is exactly what this file does,
and it is a reset-time operation, not a per-step one: calling it after a step
would drag a driven target back to its XML pose.
"""

from ..fields import Data, Model, DimsLike
from ..gpu.constants import (
    MODEL_BODY_SIZE, BODY_IDX_MOCAP,
    BODY_IDX_POS_X, BODY_IDX_POS_Y, BODY_IDX_POS_Z,
    BODY_IDX_QUAT_X, BODY_IDX_QUAT_Y, BODY_IDX_QUAT_Z, BODY_IDX_QUAT_W,
)


def reset_mocap_from_model[
    DTYPE: DType, D: DimsLike, BATCH: Int
](m: Model[DTYPE, D], mut d: Data[DTYPE, D, BATCH]) -> Int:
    """Seed every mocap body from its XML frame, then place it. Returns how
    many bodies were touched.

    ⚠ THE COUNT IS RETURNED, not logged. A caller that wants to know whether
    this model has any mocap bodies at all should not have to re-derive it,
    and a gate asserting "0 bodies were moved" on a model that HAS one is how
    a silent no-op is caught.

    ⚠ AT RESET ONLY. `mocap_pos` is an input; overwriting it every frame would
    make a driven target impossible to drive.
    """
    var nbody = m.dims.get_nbody()
    var n = 0
    for b in range(nbody):
        var o = b * MODEL_BODY_SIZE
        if m.bodies.data[o + BODY_IDX_MOCAP] == 0:
            continue
        n += 1
        var px = m.bodies.data[o + BODY_IDX_POS_X]
        var py = m.bodies.data[o + BODY_IDX_POS_Y]
        var pz = m.bodies.data[o + BODY_IDX_POS_Z]
        var qx = m.bodies.data[o + BODY_IDX_QUAT_X]
        var qy = m.bodies.data[o + BODY_IDX_QUAT_Y]
        var qz = m.bodies.data[o + BODY_IDX_QUAT_Z]
        var qw = m.bodies.data[o + BODY_IDX_QUAT_W]
        for e in range(BATCH):
            var po = (e * nbody + b) * 3
            var qo = (e * nbody + b) * 4
            d.mocap_pos.data[po + 0] = px
            d.mocap_pos.data[po + 1] = py
            d.mocap_pos.data[po + 2] = pz
            # ⚠ THE LAYOUT IS (x, y, z, w), matching `xquat` and
            # `BODY_IDX_QUAT_*`. MuJoCo's own `mocap_quat` is (w, x, y, z);
            # this tree's packed quaternions are w-LAST, and a `mocap_quat`
            # filled in MuJoCo's order would spin every driven target.
            d.mocap_quat.data[qo + 0] = qx
            d.mocap_quat.data[qo + 1] = qy
            d.mocap_quat.data[qo + 2] = qz
            d.mocap_quat.data[qo + 3] = qw
            # ⚠ AND `xipos` WITH THEM, exactly as the env facade does. The
            # inertial frame of a mocap body is used by the constraint solve;
            # leaving it at the origin puts a weld's anchor somewhere the
            # body is not.
            d.xpos.data[po + 0] = px
            d.xpos.data[po + 1] = py
            d.xpos.data[po + 2] = pz
            d.xipos.data[po + 0] = px
            d.xipos.data[po + 1] = py
            d.xipos.data[po + 2] = pz
            d.xquat.data[qo + 0] = qx
            d.xquat.data[qo + 1] = qy
            d.xquat.data[qo + 2] = qz
            d.xquat.data[qo + 3] = qw
    return n
