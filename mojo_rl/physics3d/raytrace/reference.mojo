"""`cam_pos0` / `cam_poscom0` / `cam_mat0` — the reference pose a tracking
camera translates from.

MuJoCo's COMPILER computes these, once, at `qpos0`
(`user_model.cc`; read back by `mj_camlight`, `engine_core_smooth.c:382-392`).
This engine has no model compiler, and the quantities need forward kinematics —
so they cannot be filled in `fields_build`, which never touches `Data`. They
are filled here instead, from a lane whose FK has already run.

⚠⚠ THIS IS `init_hfield_data`'S SHAPE, AND ITS FAILURE MODE. A `Data` that
never had `init_hfield_data` called has a grid of zeros, which collides and
rays perfectly happily and is simply the wrong terrain. A camera whose
reference was never taken has a zero offset and an identity orientation, which
renders perfectly happily from the body's origin looking along world -Z. Both
are pictures, not errors. `CAM_IDX_REF_SET` is what makes the second one
detectable, and `BatchedCameraRenderer` refuses to launch without it.

⚠ CALL IT AFTER A RESET'S FORWARD KINEMATICS AND SUBTREE PASS, not before.
"Reference" means "the pose the camera was authored at", so the state it reads
must be the model's rest configuration. Calling it mid-episode bakes THAT
moment in and the camera then tracks relative to wherever the robot happened to
be — a drift no gate on the picture would catch, because every individual frame
still looks correct.

⚠ IT READS ONE LANE AND WRITES A MODEL-WIDE ANSWER. `Model` is shared across
the batch by design, and `cam_pos0` is a model constant in MuJoCo too, so this
is faithful rather than a shortcut — but it does mean a task that randomises
its reset pose per lane must call this from a lane at `qpos0`, not from lane 0
after a randomised reset.
"""

from mojo_rl.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric

from ..fields import Data, Model
from ..fields.dims import DimsLike
from ..gpu.constants import (
    MAX_GPU_CAMERAS,
    MODEL_CAM_SIZE,
    CAM_IDX_ACTIVE,
    CAM_IDX_BODY,
    CAM_IDX_POS_X,
    CAM_IDX_POS_Y,
    CAM_IDX_POS_Z,
    CAM_IDX_QUAT_X,
    CAM_IDX_QUAT_Y,
    CAM_IDX_QUAT_Z,
    CAM_IDX_QUAT_W,
    CAM_IDX_POS0_X,
    CAM_IDX_POS0_Y,
    CAM_IDX_POS0_Z,
    CAM_IDX_POSCOM0_X,
    CAM_IDX_POSCOM0_Y,
    CAM_IDX_POSCOM0_Z,
    CAM_IDX_QUAT0_X,
    CAM_IDX_QUAT0_Y,
    CAM_IDX_QUAT0_Z,
    CAM_IDX_QUAT0_W,
    CAM_IDX_REF_SET,
)


def init_camera_reference[
    DTYPE: DType, D: DimsLike, BATCH: Int
](mut d: Data[DTYPE, D, BATCH], mut m: Model[DTYPE, D], env: Int = 0):
    """Fill every active camera's reference pose from lane `env`'s FK.

    ⚠ HOST-SIDE, AND IT WRITES `m.cameras.data` — the CPU mirror. The caller
    must `m.cameras.upload(ctx)` afterwards for a device render to see it.
    `BatchedCameraRenderer` does that itself; a hand-rolled caller must not
    forget, and the symptom of forgetting is a tracking camera frozen at the
    origin with `CAM_IDX_REF_SET` reading 1 on the host.
    """
    var nb = m.dims.get_nbody()
    for c in range(MAX_GPU_CAMERAS):
        var cb = c * MODEL_CAM_SIZE
        if m.cameras.data[cb + CAM_IDX_ACTIVE] == 0:
            continue
        var body = Int(m.cameras.data[cb + CAM_IDX_BODY])
        if body < 0 or body >= nb:
            continue

        var lp = Vec3Generic[DTYPE](
            m.cameras.data[cb + CAM_IDX_POS_X],
            m.cameras.data[cb + CAM_IDX_POS_Y],
            m.cameras.data[cb + CAM_IDX_POS_Z],
        )
        # (x, y, z, w) in the record, (w, x, y, z) in the constructor.
        var lq = QuatGeneric[DTYPE](
            m.cameras.data[cb + CAM_IDX_QUAT_W],
            m.cameras.data[cb + CAM_IDX_QUAT_X],
            m.cameras.data[cb + CAM_IDX_QUAT_Y],
            m.cameras.data[cb + CAM_IDX_QUAT_Z],
        )
        var bp = Vec3Generic[DTYPE](
            d.xpos.data[env * nb * 3 + body * 3 + 0],
            d.xpos.data[env * nb * 3 + body * 3 + 1],
            d.xpos.data[env * nb * 3 + body * 3 + 2],
        )
        # ⚠ `Data.xquat` is (x, y, z, w).
        var bq = QuatGeneric[DTYPE](
            d.xquat.data[env * nb * 4 + body * 4 + 3],
            d.xquat.data[env * nb * 4 + body * 4 + 0],
            d.xquat.data[env * nb * 4 + body * 4 + 1],
            d.xquat.data[env * nb * 4 + body * 4 + 2],
        )
        var com = Vec3Generic[DTYPE](
            d.subtree_com.data[env * nb * 3 + body * 3 + 0],
            d.subtree_com.data[env * nb * 3 + body * 3 + 1],
            d.subtree_com.data[env * nb * 3 + body * 3 + 2],
        )

        # `mj_local2Global` at the reference configuration.
        var wpos = bp + bq.rotate_vec(lp)
        var wq = bq * lq

        var p0 = wpos - bp
        var pc0 = wpos - com
        m.cameras.data[cb + CAM_IDX_POS0_X] = p0.x
        m.cameras.data[cb + CAM_IDX_POS0_Y] = p0.y
        m.cameras.data[cb + CAM_IDX_POS0_Z] = p0.z
        m.cameras.data[cb + CAM_IDX_POSCOM0_X] = pc0.x
        m.cameras.data[cb + CAM_IDX_POSCOM0_Y] = pc0.y
        m.cameras.data[cb + CAM_IDX_POSCOM0_Z] = pc0.z
        m.cameras.data[cb + CAM_IDX_QUAT0_X] = wq.x
        m.cameras.data[cb + CAM_IDX_QUAT0_Y] = wq.y
        m.cameras.data[cb + CAM_IDX_QUAT0_Z] = wq.z
        m.cameras.data[cb + CAM_IDX_QUAT0_W] = wq.w
        m.cameras.data[cb + CAM_IDX_REF_SET] = Scalar[DTYPE](1)
