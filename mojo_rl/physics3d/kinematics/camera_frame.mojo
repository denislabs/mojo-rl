"""World pose of a camera — MuJoCo's `d->cam_xpos` / `d->cam_xmat`.

`mj_camlight` (`engine_core_smooth.c`) opens EVERY mode with the same two
lines, before it dispatches on `cam_mode`:

    mj_local2Global(d, d->cam_xpos+3*i, d->cam_xmat+9*i,
                    m->cam_pos+3*i, m->cam_quat+4*i, cam_bodyid[i], 0);

that is, `cam_pos`/`cam_quat` are stored **in the parent body's frame** and the
world pose is `xpos[b] + xmat[b]*cam_pos` / `xmat[b]*cam_quat`. Only after that
does `TRACK`/`TRACKCOM` overwrite the orientation and `TARGETBODY` overwrite it
with a look-at.

⚠⚠ THIS COMPOSITION WAS MISSING UNTIL 2026-08-24, and the failure was silent in
the one way that is hardest to notice: every camera in every ported model was
drawn at its LOCAL pose read as a world pose. For a `<camera>` in `<worldbody>`
— which is every camera the dm_control suite declares — the parent transform is
the identity and the two agree exactly, so nothing was ever wrong on screen.
The first body-attached camera (SO-101's `wrist_cam`) is where it separates: it
stayed welded to the origin while the wrist moved. The parent body id was not
merely unused, it was DROPPED at the `CameraData` -> `RenderFields` boundary,
so the render path could not have composed the transform even if it had wanted
to.

WHY THIS IS A FUNCTION AND NOT A `Data` FIELD. Same call as
`site_frame.site_world_quat`: MuJoCo materialises `cam_xpos`/`cam_xmat` in
`mjData`, we compose on demand. Storing them would add a `[BATCH, NCAM*12]`
tensor plus a write on every forward-kinematics path and an operand to every
kernel binding `Data`, for a quantity nothing inside the dynamics reads.
⚠ That call is worth REVISITING when batched camera observations land — a ray
tracer over `[N_ENVS, ...]` wants the pose per env on the device, which is
exactly the case `site_xmat` does not have.

Quaternions here are `math3d.Quat` (w, x, y, z), NOT the `(x, y, z, w)` packing
the site and body RECORDS use.
"""

from mojo_rl.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric

comptime _V3 = Vec3Generic[DType.float64]
comptime _Q = QuatGeneric[DType.float64]


@always_inline
def camera_world_pos(
    body_pos: _V3, body_quat: _Q, local_pos: _V3
) -> _V3:
    """`cam_xpos` — `xpos[b] + xmat[b] * cam_pos`."""
    return body_pos + body_quat.rotate_vec(local_pos)


@always_inline
def camera_world_quat(body_quat: _Q, local_quat: _Q) -> _Q:
    """`cam_xmat` as a quaternion — `xmat[b] * cam_quat`.

    ⚠ Order matters and is not symmetric: the BODY rotation is applied to the
    camera's, not the other way round. `Quat.__mul__` is right-to-left (`a * b`
    applies `b` first), which is the same convention `mju_mulQuat` uses, so
    this reads exactly like the reference.
    """
    return body_quat * local_quat


@always_inline
def camera_look_dir(cam_world_quat: _Q) -> _V3:
    """The optical axis. MuJoCo's camera looks down its own **-Z** (`mjCCamera`).
    """
    return cam_world_quat.rotate_vec(_V3(0.0, 0.0, -1.0))


@always_inline
def camera_up_dir(cam_world_quat: _Q) -> _V3:
    """The camera's up vector — its own **+Y**."""
    return cam_world_quat.rotate_vec(_V3(0.0, 1.0, 0.0))
