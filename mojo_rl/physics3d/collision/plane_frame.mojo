"""Plane pose handling for the plane-vs-* narrow phases.

Every `*_plane` primitive in `collision_primitives.mojo` — `sphere_plane`,
`ellipsoid_plane`, `capsule_plane`, `cylinder_plane`, `box_plane` — is written
for a plane that is **z = 0 with normal +z**, and takes the plane as a single
`ground_z`. Until 2026-08-01 the call sites fed them `ground_z = <the plane
geom's world z>` and hardcoded the contact normal to `(0, 0, 1)`, which means
**a plane was modelled as a horizontal floor at the height of its origin, no
matter which way it actually faced.**

That was silent for every model ported up to then, because all of them have
exactly one horizontal ground plane at z = 0 — where `ground_z = 0` and normal
+z happen to be right. dm_control's `manipulator` is the first with
non-horizontal planes: two `zaxis`-rotated 45-degree walls, and a `background`
plane at `pos="0 .2 .5" zaxis="0 -1 0"` which is a VERTICAL wall at y = 0.2.
We read that one as a floor at z = 0.5 and invented a contact with the
`upper_arm` capsule, whose bottom sits at z = 0.38 — in every pose, identical,
including poses that bend the arm.

Rather than teach five primitives about orientation, the call sites now work
in the PLANE'S OWN FRAME, where the plane really is z = 0 with normal +z and
every primitive's existing assumption holds exactly:

    local  = to_plane_frame(plane, world_point)     # then use local[2] as the
                                                    # height above the plane
    ...call the primitive with ground_z = 0...
    world  = from_plane_frame(plane, local_contact)
    normal = plane_world_normal(plane)              # instead of (0, 0, 1)

For an identity plane quaternion `rotate_by_conjugate` returns its input
bit-for-bit (the cross-product terms are exactly zero), so a model whose floor
is axis-aligned AND at the origin is unaffected down to the last bit. A floor
that is axis-aligned but NOT at the origin can differ in the last bit of the
contact point, since `(p - c) + c` is not exactly `p` in floating point — the
distance, which is what the solver actually uses, is unchanged.

Quaternions here are `(x, y, z, w)`, the order the geom records use.
"""

from std.collections import InlineArray

from ..kinematics.quat_math import gpu_quat_rotate


@always_inline
def plane_world_normal[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
) -> InlineArray[Scalar[DTYPE], 3]:
    """The plane's world normal — its local +z axis, per MuJoCo's convention.

    Unit in, unit out, so no renormalisation.
    """
    return gpu_quat_rotate[DTYPE](
        qx, qy, qz, qw, Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1)
    )


@always_inline
def to_plane_frame[
    DTYPE: DType
](
    px: Scalar[DTYPE],
    py: Scalar[DTYPE],
    pz: Scalar[DTYPE],
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    wx: Scalar[DTYPE],
    wy: Scalar[DTYPE],
    wz: Scalar[DTYPE],
) -> InlineArray[Scalar[DTYPE], 3]:
    """World POINT -> the plane's frame. `[2]` is then its height above the
    plane, which is exactly what `ground_z = 0` makes the primitives expect."""
    return gpu_quat_rotate[DTYPE](-qx, -qy, -qz, qw, wx - px, wy - py, wz - pz)


@always_inline
def dir_to_plane_frame[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    vx: Scalar[DTYPE],
    vy: Scalar[DTYPE],
    vz: Scalar[DTYPE],
) -> InlineArray[Scalar[DTYPE], 3]:
    """World DIRECTION -> the plane's frame (rotation only, no translation).

    For a geom's orientation quaternion use `quat_to_plane_frame` instead —
    composing quaternions is not the same as rotating a vector.
    """
    return gpu_quat_rotate[DTYPE](-qx, -qy, -qz, qw, vx, vy, vz)


@always_inline
def from_plane_frame[
    DTYPE: DType
](
    px: Scalar[DTYPE],
    py: Scalar[DTYPE],
    pz: Scalar[DTYPE],
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    lx: Scalar[DTYPE],
    ly: Scalar[DTYPE],
    lz: Scalar[DTYPE],
) -> InlineArray[Scalar[DTYPE], 3]:
    """Plane-frame POINT -> world. Inverse of `to_plane_frame`."""
    var r = gpu_quat_rotate[DTYPE](qx, qy, qz, qw, lx, ly, lz)
    var out = InlineArray[Scalar[DTYPE], 3](uninitialized=True)
    out[0] = px + r[0]
    out[1] = py + r[1]
    out[2] = pz + r[2]
    return out^


@always_inline
def quat_to_plane_frame[
    DTYPE: DType
](
    pqx: Scalar[DTYPE],
    pqy: Scalar[DTYPE],
    pqz: Scalar[DTYPE],
    pqw: Scalar[DTYPE],
    gqx: Scalar[DTYPE],
    gqy: Scalar[DTYPE],
    gqz: Scalar[DTYPE],
    gqw: Scalar[DTYPE],
) -> InlineArray[Scalar[DTYPE], 4]:
    """A geom's world ORIENTATION expressed in the plane's frame:
    `conj(q_plane) * q_geom`, as (x, y, z, w).

    Needed by every primitive whose answer depends on how the other geom is
    turned relative to the plane — box, cylinder, ellipsoid, capsule.
    """
    # conj(p) * g, written out rather than via gpu_quat_mul so the negation
    # of the plane's vector part stays visible at the call site.
    var ax = -pqx
    var ay = -pqy
    var az = -pqz
    var aw = pqw
    var out = InlineArray[Scalar[DTYPE], 4](uninitialized=True)
    out[0] = aw * gqx + ax * gqw + ay * gqz - az * gqy
    out[1] = aw * gqy - ax * gqz + ay * gqw + az * gqx
    out[2] = aw * gqz + ax * gqy - ay * gqx + az * gqw
    out[3] = aw * gqw - ax * gqx - ay * gqy - az * gqz
    return out^
