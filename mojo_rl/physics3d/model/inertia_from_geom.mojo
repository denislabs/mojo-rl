"""Compute body mass/inertia/ipos/iquat from child geoms (MuJoCo inertiafromgeom).

Implements the MuJoCo compiler's inertiafromgeom="true" functionality:
  1. For each body, collect geoms attached to it
  2. Compute per-geom mass from density * volume (or explicit mass)
  3. Single geom: copy pos/quat/mass/inertia directly
  4. Multiple geoms: accumulate via parallel axis theorem + eigendecompose

Reference: mujoco-3.3.6/src/user/user_objects.cc (mjCBody::InertiaFromGeom)
"""

from std.collections import InlineArray
from std.math import sqrt, abs as math_abs
from ..constants import (
    GEOM_PLANE,
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
    GEOM_CYLINDER,
    GEOM_ELLIPSOID,
)

# Pi constant
comptime PI: Float64 = 3.14159265358979323846

# MuJoCo default geom density (kg/m³) — used when geom has no explicit mass
comptime MJ_DEFAULT_DENSITY: Float64 = 1000.0


# =============================================================================
# Volume computation
# =============================================================================


def geom_volume[
    DTYPE: DType
](
    geom_type: Int,
    radius: Scalar[DTYPE],
    half_length: Scalar[DTYPE],
    half_x: Scalar[DTYPE],
    half_y: Scalar[DTYPE],
    half_z: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """Compute volume of a geom. Matches MuJoCo GetVolume() for VOLUME type."""
    if geom_type == GEOM_SPHERE:
        # V = (4/3) * pi * r^3
        return Scalar[DTYPE](4.0 / 3.0 * PI) * radius * radius * radius
    elif geom_type == GEOM_CAPSULE:
        # V = pi * r^2 * h + (4/3)*pi*r^3  where h = 2*half_length
        var r2 = radius * radius
        var h = Scalar[DTYPE](2.0) * half_length
        return (
            Scalar[DTYPE](PI) * r2 * h
            + Scalar[DTYPE](4.0 / 3.0 * PI) * r2 * radius
        )
    elif geom_type == GEOM_CYLINDER:
        # V = pi * r^2 * 2h (no hemisphere caps)
        var r2 = radius * radius
        var h = Scalar[DTYPE](2.0) * half_length
        return Scalar[DTYPE](PI) * r2 * h
    elif geom_type == GEOM_BOX:
        # V = 8 * hx * hy * hz
        return Scalar[DTYPE](8.0) * half_x * half_y * half_z
    elif geom_type == GEOM_ELLIPSOID:
        # V = (4/3) * pi * a * b * c, the three semi-axes from `size`.
        return (
            Scalar[DTYPE](4.0 / 3.0 * PI) * half_x * half_y * half_z
        )
    else:
        # Plane or unknown: zero volume
        return Scalar[DTYPE](0.0)


def geom_effective_mass[
    DTYPE: DType
](
    geom_type: Int,
    stored_mass: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    half_length: Scalar[DTYPE],
    half_x: Scalar[DTYPE],
    half_y: Scalar[DTYPE],
    half_z: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """Return effective geom mass: explicit if >= 0, else density * volume.

    MuJoCo stores -1.0 when mass is absent (inertiafromgeom uses density).
    """
    if stored_mass >= Scalar[DTYPE](0):
        return stored_mass
    # Compute from default density (1000 kg/m³)
    var vol = geom_volume(
        geom_type, radius, half_length, half_x, half_y, half_z
    )
    return Scalar[DTYPE](MJ_DEFAULT_DENSITY) * vol


# =============================================================================
# Per-geom diagonal inertia (in geom's local frame)
# =============================================================================


def geom_inertia[
    DTYPE: DType
](
    geom_type: Int,
    mass: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    half_length: Scalar[DTYPE],
    half_x: Scalar[DTYPE],
    half_y: Scalar[DTYPE],
    half_z: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Compute diagonal inertia (Ixx, Iyy, Izz) for a geom in its local frame.

    Matches MuJoCo SetInertia() for VOLUME type. Capsule axis is Z.
    """
    if geom_type == GEOM_SPHERE:
        # I = (2/5) * m * r^2
        var I = Scalar[DTYPE](0.4) * mass * radius * radius
        return (I, I, I)

    elif geom_type == GEOM_CAPSULE:
        # MuJoCo capsule inertia with proper hemisphere parallel axis theorem
        var r = radius
        var r2 = r * r
        var h = Scalar[DTYPE](2.0) * half_length  # cylinder height

        # Mass partition: sphere vs cylinder proportional to volume
        # sphere_mass = m * (4r) / (4r + 3h)
        var denom = Scalar[DTYPE](4.0) * r + Scalar[DTYPE](3.0) * h
        var sphere_mass = Scalar[DTYPE](0.0)
        var cylinder_mass = Scalar[DTYPE](0.0)
        if denom > Scalar[DTYPE](1e-30):
            sphere_mass = mass * Scalar[DTYPE](4.0) * r / denom
            cylinder_mass = mass - sphere_mass

        # Sphere (two hemispheres) inertia about own center
        var sphere_inertia = Scalar[DTYPE](0.4) * sphere_mass * r2

        # Transverse inertia (Ix = Iy)
        # Cylinder: m*(3r^2 + h^2)/12
        # Hemispheres: sphere_inertia + sphere_mass * h * (3r + 2h) / 8
        var Ix = (
            cylinder_mass
            * (Scalar[DTYPE](3.0) * r2 + h * h)
            / Scalar[DTYPE](12.0)
            + sphere_inertia
            + sphere_mass
            * h
            * (Scalar[DTYPE](3.0) * r + Scalar[DTYPE](2.0) * h)
            / Scalar[DTYPE](8.0)
        )

        # Axial inertia (Iz)
        # Cylinder: m*r^2/2, Hemispheres: sphere_inertia
        var Iz = cylinder_mass * r2 / Scalar[DTYPE](2.0) + sphere_inertia

        return (Ix, Ix, Iz)

    elif geom_type == GEOM_CYLINDER:
        # Solid cylinder: axis = Z, height h = 2*half_length, radius r
        # Iz = m*r^2/2
        # Ix = Iy = m*(3r^2 + h^2)/12
        var r = radius
        var r2 = r * r
        var h = Scalar[DTYPE](2.0) * half_length
        var Iz = mass * r2 / Scalar[DTYPE](2.0)
        var Ix = mass * (Scalar[DTYPE](3.0) * r2 + h * h) / Scalar[DTYPE](12.0)
        return (Ix, Ix, Iz)

    elif geom_type == GEOM_BOX:
        # I_xx = m*(sy^2 + sz^2)/3, etc. (using half-sizes)
        var sx2 = half_x * half_x
        var sy2 = half_y * half_y
        var sz2 = half_z * half_z
        var Ix = mass * (sy2 + sz2) / Scalar[DTYPE](3.0)
        var Iy = mass * (sx2 + sz2) / Scalar[DTYPE](3.0)
        var Iz = mass * (sx2 + sy2) / Scalar[DTYPE](3.0)
        return (Ix, Iy, Iz)

    elif geom_type == GEOM_ELLIPSOID:
        # I_xx = m*(b^2 + c^2)/5, etc. — the box formula with /5 instead of
        # /3, and a sphere's (2/5)mr^2 when the three semi-axes are equal.
        var ax2 = half_x * half_x
        var ay2 = half_y * half_y
        var az2 = half_z * half_z
        var Ix = mass * (ay2 + az2) / Scalar[DTYPE](5.0)
        var Iy = mass * (ax2 + az2) / Scalar[DTYPE](5.0)
        var Iz = mass * (ax2 + ay2) / Scalar[DTYPE](5.0)
        return (Ix, Iy, Iz)

    else:
        return (Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](0))


# =============================================================================
# Quaternion → rotation matrix
# =============================================================================


def quat_to_mat[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    mut mat: InlineArray[Scalar[DTYPE], 9],
):
    """Convert quaternion (x,y,z,w) to 3x3 rotation matrix (row-major).

    mat[row*3+col]. Matches MuJoCo's column-major convention reindexed.
    """
    var x2 = qx + qx
    var y2 = qy + qy
    var z2 = qz + qz
    var xx = qx * x2
    var xy = qx * y2
    var xz = qx * z2
    var yy = qy * y2
    var yz = qy * z2
    var zz = qz * z2
    var wx = qw * x2
    var wy = qw * y2
    var wz = qw * z2

    mat[0] = Scalar[DTYPE](1) - yy - zz
    mat[1] = xy - wz
    mat[2] = xz + wy
    mat[3] = xy + wz
    mat[4] = Scalar[DTYPE](1) - xx - zz
    mat[5] = yz - wx
    mat[6] = xz - wy
    mat[7] = yz + wx
    mat[8] = Scalar[DTYPE](1) - xx - yy


# =============================================================================
# globalinertia — rotate diagonal inertia to full 6-element tensor
# =============================================================================


def globalinertia[
    DTYPE: DType
](
    local_ixx: Scalar[DTYPE],
    local_iyy: Scalar[DTYPE],
    local_izz: Scalar[DTYPE],
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    mut result: InlineArray[Scalar[DTYPE], 6],
):
    """Rotate diagonal inertia tensor by quaternion to full symmetric tensor.

    result = [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    Matches mjuu_globalinertia: R * diag(I) * R^T
    """
    var mat = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    quat_to_mat(qx, qy, qz, qw, mat)

    # tmp[col][row] = R[row][col] * local[col]
    # Using row-major mat: mat[row*3+col]
    # MuJoCo uses column-major: mat_mj[col*3+row] = mat[row*3+col]
    # tmp_mj[0] = mat_mj[0]*I0 = mat[0*3+0]*I0 = mat[0]*I0
    # tmp_mj[1] = mat_mj[3]*I0 = mat[1*3+0]*I0 = mat[3]*I0
    # tmp_mj[2] = mat_mj[6]*I0 = mat[2*3+0]*I0 = mat[6]*I0
    # etc.
    var t00 = mat[0] * local_ixx
    var t10 = mat[3] * local_ixx
    var t20 = mat[6] * local_ixx
    var t01 = mat[1] * local_iyy
    var t11 = mat[4] * local_iyy
    var t21 = mat[7] * local_iyy
    var t02 = mat[2] * local_izz
    var t12 = mat[5] * local_izz
    var t22 = mat[8] * local_izz

    # global = R * diag * R^T (upper triangular: Ixx, Iyy, Izz, Ixy, Ixz, Iyz)
    # MuJoCo: global[0] = mat[0]*tmp[0] + mat[1]*tmp[3] + mat[2]*tmp[6]
    # In our notation (mat_mj[col*3+row] = mat[row*3+col]):
    # mat_mj[0]=mat[0], mat_mj[1]=mat[3], mat_mj[2]=mat[6]
    # mat_mj[3]=mat[1], mat_mj[4]=mat[4], mat_mj[5]=mat[7]
    # mat_mj[6]=mat[2], mat_mj[7]=mat[5], mat_mj[8]=mat[8]
    result[0] = mat[0] * t00 + mat[1] * t01 + mat[2] * t02  # Ixx
    result[1] = mat[3] * t10 + mat[4] * t11 + mat[5] * t12  # Iyy
    result[2] = mat[6] * t20 + mat[7] * t21 + mat[8] * t22  # Izz
    result[3] = mat[0] * t10 + mat[1] * t11 + mat[2] * t12  # Ixy
    result[4] = mat[0] * t20 + mat[1] * t21 + mat[2] * t22  # Ixz
    result[5] = mat[3] * t20 + mat[4] * t21 + mat[5] * t22  # Iyz


# =============================================================================
# offcenter — parallel axis theorem
# =============================================================================


def offcenter[
    DTYPE: DType
](
    mass: Scalar[DTYPE],
    dx: Scalar[DTYPE],
    dy: Scalar[DTYPE],
    dz: Scalar[DTYPE],
    mut result: InlineArray[Scalar[DTYPE], 6],
):
    """Parallel axis theorem correction for offset CoM.

    result = [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] (additive)
    Matches mjuu_offcenter.
    """
    result[0] = mass * (dy * dy + dz * dz)  # Ixx += m*(y^2+z^2)
    result[1] = mass * (dx * dx + dz * dz)  # Iyy += m*(x^2+z^2)
    result[2] = mass * (dx * dx + dy * dy)  # Izz += m*(x^2+y^2)
    result[3] = -mass * dx * dy  # Ixy -= m*xy
    result[4] = -mass * dx * dz  # Ixz -= m*xz
    result[5] = -mass * dy * dz  # Iyz -= m*yz


# =============================================================================
# Jacobi eigendecomposition of 3x3 symmetric matrix
# =============================================================================


def mat3_to_quat[
    DTYPE: DType
](
    mat: InlineArray[Scalar[DTYPE], 9],
) -> Tuple[
    Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]
]:
    """Convert 3x3 rotation matrix (row-major) to quaternion (x,y,z,w).

    Uses Shepperd's method for numerical stability.
    """
    var trace = mat[0] + mat[4] + mat[8]
    var qx: Scalar[DTYPE]
    var qy: Scalar[DTYPE]
    var qz: Scalar[DTYPE]
    var qw: Scalar[DTYPE]

    if trace > Scalar[DTYPE](0):
        var s = sqrt(trace + Scalar[DTYPE](1)) * Scalar[DTYPE](2)
        qw = s * Scalar[DTYPE](0.25)
        qx = (mat[7] - mat[5]) / s
        qy = (mat[2] - mat[6]) / s
        qz = (mat[3] - mat[1]) / s
    elif mat[0] > mat[4] and mat[0] > mat[8]:
        var s = sqrt(Scalar[DTYPE](1) + mat[0] - mat[4] - mat[8]) * Scalar[
            DTYPE
        ](2)
        qw = (mat[7] - mat[5]) / s
        qx = s * Scalar[DTYPE](0.25)
        qy = (mat[1] + mat[3]) / s
        qz = (mat[2] + mat[6]) / s
    elif mat[4] > mat[8]:
        var s = sqrt(Scalar[DTYPE](1) + mat[4] - mat[0] - mat[8]) * Scalar[
            DTYPE
        ](2)
        qw = (mat[2] - mat[6]) / s
        qx = (mat[1] + mat[3]) / s
        qy = s * Scalar[DTYPE](0.25)
        qz = (mat[5] + mat[7]) / s
    else:
        var s = sqrt(Scalar[DTYPE](1) + mat[8] - mat[0] - mat[4]) * Scalar[
            DTYPE
        ](2)
        qw = (mat[3] - mat[1]) / s
        qx = (mat[2] + mat[6]) / s
        qy = (mat[5] + mat[7]) / s
        qz = s * Scalar[DTYPE](0.25)

    # Normalize
    var norm = sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm > Scalar[DTYPE](1e-30):
        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm

    return (qx, qy, qz, qw)


def _mulquat[
    DTYPE: DType
](
    ax: Scalar[DTYPE], ay: Scalar[DTYPE], az: Scalar[DTYPE], aw: Scalar[DTYPE],
    bx: Scalar[DTYPE], by: Scalar[DTYPE], bz: Scalar[DTYPE], bw: Scalar[DTYPE],
) -> InlineArray[Scalar[DTYPE], 4]:
    """`mjuu_mulquat` in OUR (x, y, z, w) storage order."""
    var r = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    r[3] = aw * bw - ax * bx - ay * by - az * bz
    r[0] = aw * bx + ax * bw + ay * bz - az * by
    r[1] = aw * by - ax * bz + ay * bw + az * bx
    r[2] = aw * bz + ax * by - ay * bx + az * bw
    return r^


def eig3_symmetric[
    DTYPE: DType
](
    mut full_inertia: InlineArray[Scalar[DTYPE], 6],
) -> Tuple[
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
]:
    """`mjuu_eig3` — quaternion-accumulating Jacobi, decreasing eigenvalues.

    Input: full_inertia = [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    Returns: (eigval0, eigval1, eigval2, qx, qy, qz, qw)

    ⚠⚠ THIS IS A TRANSCRIPTION, NOT "AN" EIGENSOLVER, AND THE DIFFERENCE IS
    OBSERVABLE. The previous implementation was a perfectly valid symmetric
    Jacobi — it accumulated an eigenvector MATRIX, sorted by swapping columns,
    and repaired handedness by negating the last column. It produced the same
    EIGENVALUES as MuJoCo and a different FRAME: measured against
    `mjModel.mesh_quat` on Jaco's 9 meshes, 3 matched to 1e-13 and 6 were out
    by 0.75 to 1.0 — whole axis relabelings and sign flips, not drift.

    That is not merely cosmetic. `body_iquat` is a model constant a gate
    compares directly, and for a body with two near-equal principal moments the
    frame is genuinely underdetermined: only reproducing MuJoCo's exact
    iteration reproduces its answer. Three details do the work, and all three
    were different before:

      * **The pivot test uses strict `>`**, in MuJoCo's order. `>=` picks a
        different axis on ties, and ties are common in near-symmetric parts.
      * **The rotation is accumulated as a QUATERNION** with MuJoCo's sign:
        `tmp[rotk] = -sqrt(0.5-0.5c)` for `tau >= 0`, negated again when
        `rotk == 1`. A rotation matrix built with the opposite off-diagonal
        sign diagonalizes just as well and lands on the mirrored frame.
      * **The sort rotates the quaternion by 90 degrees** about axis
        `(j1+2)%3` rather than swapping columns, so the result stays
        right-handed by construction. The old column-swap could produce a
        left-handed triple, and the negate-last-column repair is a REFLECTION
        MuJoCo never applies.

    ⚠ The bubblesort is MuJoCo's exact three-step `j1 = j % 2` (indices
    0, 1, 0), not a full sort, and it only swaps when the difference exceeds
    `kEigEPS`. A general sort would reorder equal eigenvalues that MuJoCo
    leaves alone, which puts the frame back off by a 90-degree rotation.
    """
    comptime EPS = Scalar[DTYPE](1e-12)  # kEigEPS

    var mat = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    mat[0] = full_inertia[0]
    mat[1] = full_inertia[3]
    mat[2] = full_inertia[4]
    mat[3] = full_inertia[3]
    mat[4] = full_inertia[1]
    mat[5] = full_inertia[5]
    mat[6] = full_inertia[4]
    mat[7] = full_inertia[5]
    mat[8] = full_inertia[2]

    var qx = Scalar[DTYPE](0)
    var qy = Scalar[DTYPE](0)
    var qz = Scalar[DTYPE](0)
    var qw = Scalar[DTYPE](1)

    var eigvals = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))

    for _ in range(500):
        # D = V^T * mat * V, with V = quat2mat(quat)
        var V = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
        quat_to_mat[DTYPE](qx, qy, qz, qw, V)
        var D = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
        for i in range(3):
            for j in range(3):
                var acc = Scalar[DTYPE](0)
                for k in range(3):
                    for l in range(3):
                        acc += V[k * 3 + i] * mat[k * 3 + l] * V[l * 3 + j]
                D[i * 3 + j] = acc

        eigvals[0] = D[0]
        eigvals[1] = D[4]
        eigvals[2] = D[8]

        # ⚠ STRICT `>`, in this order — see the docstring.
        var a01 = math_abs(D[1])
        var a02 = math_abs(D[2])
        var a12 = math_abs(D[5])
        var rk: Int
        var ck: Int
        var rotk: Int
        if a01 > a02 and a01 > a12:
            rk = 0
            ck = 1
            rotk = 2
        elif a02 > a12:
            rk = 0
            ck = 2
            rotk = 1
        else:
            rk = 1
            ck = 2
            rotk = 0

        if math_abs(D[rk * 3 + ck]) < EPS:
            break

        var tau = (D[4 * ck] - D[4 * rk]) / (
            Scalar[DTYPE](2) * D[rk * 3 + ck]
        )
        var t: Scalar[DTYPE]
        if tau >= Scalar[DTYPE](0):
            t = Scalar[DTYPE](1) / (tau + sqrt(Scalar[DTYPE](1) + tau * tau))
        else:
            t = Scalar[DTYPE](-1) / (-tau + sqrt(Scalar[DTYPE](1) + tau * tau))
        var c = Scalar[DTYPE](1) / sqrt(Scalar[DTYPE](1) + t * t)
        if c > Scalar[DTYPE](1) - EPS:
            break

        var half = sqrt(Scalar[DTYPE](0.5) - Scalar[DTYPE](0.5) * c)
        var axis = -half if tau >= Scalar[DTYPE](0) else half
        if rotk == 1:
            axis = -axis
        var tx = Scalar[DTYPE](0)
        var ty = Scalar[DTYPE](0)
        var tz = Scalar[DTYPE](0)
        if rotk == 0:
            tx = axis
        elif rotk == 1:
            ty = axis
        else:
            tz = axis
        var tw = sqrt(Scalar[DTYPE](1) - axis * axis)
        var tn = sqrt(tx * tx + ty * ty + tz * tz + tw * tw)
        tx /= tn
        ty /= tn
        tz /= tn
        tw /= tn

        var q = _mulquat[DTYPE](qx, qy, qz, qw, tx, ty, tz, tw)
        var qn = sqrt(
            q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
        )
        qx = q[0] / qn
        qy = q[1] / qn
        qz = q[2] / qn
        qw = q[3] / qn

    # Decreasing-order bubblesort: lead index 0, 1, 0.
    for j in range(3):
        var j1 = j % 2
        if eigvals[j1] + EPS < eigvals[j1 + 1]:
            var tmp = eigvals[j1]
            eigvals[j1] = eigvals[j1 + 1]
            eigvals[j1 + 1] = tmp
            # 90-degree rotation about axis (j1+2)%3
            var hs = Scalar[DTYPE](0.707106781186548)
            var ax = Scalar[DTYPE](0)
            var ay = Scalar[DTYPE](0)
            var az = Scalar[DTYPE](0)
            var which = (j1 + 2) % 3
            if which == 0:
                ax = hs
            elif which == 1:
                ay = hs
            else:
                az = hs
            var q2 = _mulquat[DTYPE](qx, qy, qz, qw, ax, ay, az, hs)
            var q2n = sqrt(
                q2[0] * q2[0] + q2[1] * q2[1] + q2[2] * q2[2] + q2[3] * q2[3]
            )
            qx = q2[0] / q2n
            qy = q2[1] / q2n
            qz = q2[2] / q2n
            qw = q2[3] / q2n

    return (eigvals[0], eigvals[1], eigvals[2], qx, qy, qz, qw)


# =============================================================================
# Main: compute_inertia_from_geoms
# =============================================================================


# The Model-typed `compute_inertia_from_geoms` (+ `_buffer` variant) were
# deleted at the G4 fields sunset — the spec-direct port is
# `fields_build._inertia_from_geoms_staging`, which reuses the pure-math
# helpers above (geom_volume / geom_effective_mass / geom_inertia /
# quat_to_mat / globalinertia / offcenter / mat3_to_quat / eig3_symmetric).
