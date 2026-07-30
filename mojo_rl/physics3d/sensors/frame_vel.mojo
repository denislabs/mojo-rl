"""Site-frame velocity sensors — MuJoCo `velocimeter` and `gyro`.

Both are thin wrappers over `mj_objectVelocity(m, d, mjOBJ_SITE, id, res, 1)`
(engine_support.c), which transports the body's spatial velocity to the site
point and rotates it into the site frame:

    mju_transformSpatial(res, d->cvel + 6*bodyid, 0,
                         d->site_xpos + 3*siteid,          <- to here
                         d->subtree_com + 3*body_rootid,   <- from here
                         d->site_xmat + 9*siteid)          <- into this frame

    mjSENS_VELOCIMETER -> res[3:6]   (linear)
    mjSENS_GYRO        -> res[0:3]   (angular)

Our `Data` carries the same content in a different parameterisation:
`xangvel[b]` is the body's world angular velocity, and `xvel[b]` is its world
linear velocity AT `xipos[b]` (the body CoM — see `_vel_body`, which propagates
`v = v_parent + w_parent x (xipos - xipos_parent)`). So the transport is a
single cross product and the rotation is one inverse quaternion rotation:

    v_site_world = xvel[b] + xangvel[b] x (site_xpos - xipos[b])
    velocimeter  = R_site^T v_site_world
    gyro         = R_site^T xangvel[b]

⚠ THE SITE FRAME IS TAKEN TO BE THE BODY FRAME. `Data` stores `site_xpos` but
no `site_xmat`, so `R_site` here is the body's `xquat`. That is exact only for
a site that declares no `quat`/`euler`/`axisangle`/`zaxis` — i.e. one whose
local orientation is identity. Every site swimmer declares is of that form
(`<site name="site_i"/>` under `class="swimmer"`, which sets only `size` and
`rgba`), and the swimmer parity test asserts `site_quat == (1,0,0,0)` for all
of them against the reference model, so a rotated site cannot slip in
unnoticed. Adding `site_xmat` is the real fix when a domain needs one.

Used by dm_control's swimmer, whose `body_velocities()` observation is the
`[vx, vy, wz]` slice of one velocimeter/gyro pair per link.
"""

from ..kinematics.quat_math import quat_rotate_inverse


def site_frame_velocity[
    DTYPE: DType
](
    xvel: List[Scalar[DTYPE]],
    xangvel: List[Scalar[DTYPE]],
    xipos: List[Scalar[DTYPE]],
    xquat: List[Scalar[DTYPE]],
    site_xpos: List[Scalar[DTYPE]],
    body: Int,
    site: Int,
) raises -> Tuple[Float64, Float64, Float64, Float64, Float64, Float64]:
    """`(velocimeter[0:3], gyro[0:3])` for `site` mounted on `body`.

    All six lists are the corresponding `Data` host buffers (`.data`), single
    env. Returns linear first, then angular — the opposite of MuJoCo's packed
    `res` ordering, because callers here read the linear part far more often.
    """
    var wx = xangvel[body * 3 + 0]
    var wy = xangvel[body * 3 + 1]
    var wz = xangvel[body * 3 + 2]

    # Transport the CoM velocity to the site point.
    var rx = site_xpos[site * 3 + 0] - xipos[body * 3 + 0]
    var ry = site_xpos[site * 3 + 1] - xipos[body * 3 + 1]
    var rz = site_xpos[site * 3 + 2] - xipos[body * 3 + 2]

    var vx = xvel[body * 3 + 0] + (wy * rz - wz * ry)
    var vy = xvel[body * 3 + 1] + (wz * rx - wx * rz)
    var vz = xvel[body * 3 + 2] + (wx * ry - wy * rx)

    var qx = xquat[body * 4 + 0]
    var qy = xquat[body * 4 + 1]
    var qz = xquat[body * 4 + 2]
    var qw = xquat[body * 4 + 3]

    var vl = quat_rotate_inverse[DTYPE](qx, qy, qz, qw, vx, vy, vz)
    var wl = quat_rotate_inverse[DTYPE](qx, qy, qz, qw, wx, wy, wz)

    return (
        Float64(vl[0]),
        Float64(vl[1]),
        Float64(vl[2]),
        Float64(wl[0]),
        Float64(wl[1]),
        Float64(wl[2]),
    )
