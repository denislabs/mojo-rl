"""MuJoCo-compatible body rotation matrices (`xmat`) from `Data.xquat`.

MuJoCo exposes each body's world orientation twice: as `data.xquat` (a
quaternion) and as `data.xmat` (the same rotation as a flattened row-major
3x3). Our `Data` stores only the quaternion, but a lot of task code — most of
the dm_control suite, which reads `named.data.xmat['torso', 'zz']` for upright
/ orientation terms — is written against the matrix form.

Rather than materialize an `xmat` tensor (another NBODY*9 field to allocate,
upload, and thread through every GPU hook signature), this derives the
element on demand. Each accessor is a handful of flops on values already in
registers, and reward hooks typically want one or two specific entries.

Column naming matches MuJoCo's `named` indexer: 'xx' 'xy' 'xz' 'yx' 'yy' 'yz'
'zx' 'zy' 'zz', row-major — so `XMAT_ZZ` is index 8, `XMAT_XZ` is index 2.

QUATERNION ORDER: `Data.xquat` is [x, y, z, w] (see `forward_kinematics`),
NOT MuJoCo's [w, x, y, z]. The accessors below take the components by name to
keep that from biting at the call site.
"""

from ..fields import Data


# Row-major indices into a flattened 3x3, matching MuJoCo's named columns.
comptime XMAT_XX: Int = 0
comptime XMAT_XY: Int = 1
comptime XMAT_XZ: Int = 2
comptime XMAT_YX: Int = 3
comptime XMAT_YY: Int = 4
comptime XMAT_YZ: Int = 5
comptime XMAT_ZX: Int = 6
comptime XMAT_ZY: Int = 7
comptime XMAT_ZZ: Int = 8


@always_inline
def quat_xmat_elem(
    qx: Float64, qy: Float64, qz: Float64, qw: Float64, idx: Int
) -> Float64:
    """One element of the row-major rotation matrix for quaternion (x,y,z,w).

    `idx` is an `XMAT_*` constant. Standard quaternion-to-matrix identities;
    gated against MuJoCo's own `data.xmat` in
    `tests/dm_control/test_xmat_vs_mujoco.mojo`.
    """
    if idx == XMAT_XX:
        return 1.0 - 2.0 * (qy * qy + qz * qz)
    if idx == XMAT_XY:
        return 2.0 * (qx * qy - qw * qz)
    if idx == XMAT_XZ:
        return 2.0 * (qx * qz + qw * qy)
    if idx == XMAT_YX:
        return 2.0 * (qx * qy + qw * qz)
    if idx == XMAT_YY:
        return 1.0 - 2.0 * (qx * qx + qz * qz)
    if idx == XMAT_YZ:
        return 2.0 * (qy * qz - qw * qx)
    if idx == XMAT_ZX:
        return 2.0 * (qx * qz - qw * qy)
    if idx == XMAT_ZY:
        return 2.0 * (qy * qz + qw * qx)
    return 1.0 - 2.0 * (qx * qx + qy * qy)  # XMAT_ZZ


@always_inline
def xmat_elem[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    MAX_CONTACTS: Int,
    NSITE: Int,
](
    d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
    body: Int,
    idx: Int,
) -> Float64:
    """`data.xmat[body, idx]` for the single-env (BATCH=1) CPU path.

    Equivalent to MuJoCo's `named.data.xmat[body_name][idx]`.
    """
    var qx = Float64(d.xquat.data[body * 4 + 0])
    var qy = Float64(d.xquat.data[body * 4 + 1])
    var qz = Float64(d.xquat.data[body * 4 + 2])
    var qw = Float64(d.xquat.data[body * 4 + 3])
    return quat_xmat_elem(qx, qy, qz, qw, idx)
