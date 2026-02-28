"""Compute body mass/inertia/ipos/iquat from child geoms (MuJoCo inertiafromgeom).

Implements the MuJoCo compiler's inertiafromgeom="true" functionality:
  1. For each body, collect geoms attached to it
  2. Compute per-geom mass from density * volume (or explicit mass)
  3. Single geom: copy pos/quat/mass/inertia directly
  4. Multiple geoms: accumulate via parallel axis theorem + eigendecompose

Reference: mujoco-3.3.6/src/user/user_objects.cc (mjCBody::InertiaFromGeom)
"""

from collections import InlineArray
from math import sqrt, abs as math_abs
from ..constants import GEOM_PLANE, GEOM_SPHERE, GEOM_CAPSULE, GEOM_BOX, GEOM_CYLINDER
from ..types import Model
from gpu.host import HostBuffer
from ..gpu.constants import (
    BODY_IDX_MASS,
    BODY_IDX_INV_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_INV_IXX,
    BODY_IDX_INV_IYY,
    BODY_IDX_INV_IZZ,
    BODY_IDX_IPOS_X,
    BODY_IDX_IPOS_Y,
    BODY_IDX_IPOS_Z,
    BODY_IDX_IQUAT_X,
    BODY_IDX_IQUAT_Y,
    BODY_IDX_IQUAT_Z,
    BODY_IDX_IQUAT_W,
    model_body_offset,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z,
    model_geom_offset,
)

# Pi constant
comptime PI: Float64 = 3.14159265358979323846

# MuJoCo default geom density (kg/m³) — used when geom has no explicit mass
comptime MJ_DEFAULT_DENSITY: Float64 = 1000.0


# =============================================================================
# Volume computation
# =============================================================================


fn geom_volume[
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
        return Scalar[DTYPE](PI) * r2 * h + Scalar[DTYPE](
            4.0 / 3.0 * PI
        ) * r2 * radius
    elif geom_type == GEOM_CYLINDER:
        # V = pi * r^2 * 2h (no hemisphere caps)
        var r2 = radius * radius
        var h = Scalar[DTYPE](2.0) * half_length
        return Scalar[DTYPE](PI) * r2 * h
    elif geom_type == GEOM_BOX:
        # V = 8 * hx * hy * hz
        return Scalar[DTYPE](8.0) * half_x * half_y * half_z
    else:
        # Plane or unknown: zero volume
        return Scalar[DTYPE](0.0)


fn geom_effective_mass[
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
    var vol = geom_volume(geom_type, radius, half_length, half_x, half_y, half_z)
    return Scalar[DTYPE](MJ_DEFAULT_DENSITY) * vol


# =============================================================================
# Per-geom diagonal inertia (in geom's local frame)
# =============================================================================


fn geom_inertia[
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
        var Ix = cylinder_mass * (Scalar[DTYPE](3.0) * r2 + h * h) / Scalar[
            DTYPE
        ](12.0) + sphere_inertia + sphere_mass * h * (
            Scalar[DTYPE](3.0) * r + Scalar[DTYPE](2.0) * h
        ) / Scalar[
            DTYPE
        ](
            8.0
        )

        # Axial inertia (Iz)
        # Cylinder: m*r^2/2, Hemispheres: sphere_inertia
        var Iz = cylinder_mass * r2 / Scalar[DTYPE](
            2.0
        ) + sphere_inertia

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

    else:
        return (Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](0))


# =============================================================================
# Quaternion → rotation matrix
# =============================================================================


fn quat_to_mat[
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


fn globalinertia[
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


fn offcenter[
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


fn mat3_to_quat[
    DTYPE: DType
](
    mat: InlineArray[Scalar[DTYPE], 9],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
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
        var s = sqrt(
            Scalar[DTYPE](1) + mat[0] - mat[4] - mat[8]
        ) * Scalar[DTYPE](2)
        qw = (mat[7] - mat[5]) / s
        qx = s * Scalar[DTYPE](0.25)
        qy = (mat[1] + mat[3]) / s
        qz = (mat[2] + mat[6]) / s
    elif mat[4] > mat[8]:
        var s = sqrt(
            Scalar[DTYPE](1) + mat[4] - mat[0] - mat[8]
        ) * Scalar[DTYPE](2)
        qw = (mat[2] - mat[6]) / s
        qx = (mat[1] + mat[3]) / s
        qy = s * Scalar[DTYPE](0.25)
        qz = (mat[5] + mat[7]) / s
    else:
        var s = sqrt(
            Scalar[DTYPE](1) + mat[8] - mat[0] - mat[4]
        ) * Scalar[DTYPE](2)
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


fn eig3_symmetric[
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
    """Jacobi eigendecomposition of symmetric 3x3 inertia tensor.

    Input: full_inertia = [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    Returns: (eigval0, eigval1, eigval2, qx, qy, qz, qw)
    Eigenvalues sorted in decreasing order (matches MuJoCo mjuu_eig3).
    """
    # Build symmetric 3x3 matrix (row-major)
    var mat = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    mat[0] = full_inertia[0]  # Ixx
    mat[1] = full_inertia[3]  # Ixy
    mat[2] = full_inertia[4]  # Ixz
    mat[3] = full_inertia[3]  # Ixy
    mat[4] = full_inertia[1]  # Iyy
    mat[5] = full_inertia[5]  # Iyz
    mat[6] = full_inertia[4]  # Ixz
    mat[7] = full_inertia[5]  # Iyz
    mat[8] = full_inertia[2]  # Izz

    # Eigenvector matrix (starts as identity, row-major)
    var eigvec = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    eigvec[0] = Scalar[DTYPE](1)
    eigvec[4] = Scalar[DTYPE](1)
    eigvec[8] = Scalar[DTYPE](1)

    # Jacobi iteration (matches MuJoCo: up to 500 iterations)
    for _ in range(500):
        # Compute D = eigvec^T * mat * eigvec
        var D = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
        for i in range(3):
            for j in range(3):
                var s = Scalar[DTYPE](0)
                for k in range(3):
                    for l in range(3):
                        s += eigvec[k * 3 + i] * mat[k * 3 + l] * eigvec[
                            l * 3 + j
                        ]
                D[i * 3 + j] = s

        # Find largest off-diagonal element
        # Off-diagonal pairs: (0,1), (0,2), (1,2)
        var max_val: Scalar[DTYPE]
        var rk: Int
        var ck: Int

        var abs01 = math_abs(D[0 * 3 + 1])
        var abs02 = math_abs(D[0 * 3 + 2])
        var abs12 = math_abs(D[1 * 3 + 2])

        if abs01 >= abs02 and abs01 >= abs12:
            max_val = abs01
            rk = 0
            ck = 1
        elif abs02 >= abs12:
            max_val = abs02
            rk = 0
            ck = 2
        else:
            max_val = abs12
            rk = 1
            ck = 2

        # Check convergence
        if max_val < Scalar[DTYPE](1e-12):
            break

        # 2x2 Schur decomposition
        var d_diff = D[ck * 3 + ck] - D[rk * 3 + rk]
        var tau = d_diff / (Scalar[DTYPE](2) * D[rk * 3 + ck])
        var t: Scalar[DTYPE]
        if tau >= Scalar[DTYPE](0):
            t = Scalar[DTYPE](1) / (
                tau + sqrt(Scalar[DTYPE](1) + tau * tau)
            )
        else:
            t = Scalar[DTYPE](-1) / (
                -tau + sqrt(Scalar[DTYPE](1) + tau * tau)
            )
        var c = Scalar[DTYPE](1) / sqrt(Scalar[DTYPE](1) + t * t)
        var s = t * c

        # Apply Jacobi rotation: eigvec = eigvec * Rot
        # Build rotation matrix for this Jacobi rotation
        var rot_mat = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
        rot_mat[0] = Scalar[DTYPE](1)
        rot_mat[4] = Scalar[DTYPE](1)
        rot_mat[8] = Scalar[DTYPE](1)
        rot_mat[rk * 3 + rk] = c
        rot_mat[ck * 3 + ck] = c
        rot_mat[rk * 3 + ck] = s
        rot_mat[ck * 3 + rk] = -s

        # new_eigvec = eigvec * rot_mat
        var new_eigvec = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
        for i in range(3):
            for j in range(3):
                var val = Scalar[DTYPE](0)
                for k in range(3):
                    val += eigvec[i * 3 + k] * rot_mat[k * 3 + j]
                new_eigvec[i * 3 + j] = val
        eigvec = new_eigvec^

    # Extract eigenvalues from D = eigvec^T * mat * eigvec
    var eigvals = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    for i in range(3):
        var s = Scalar[DTYPE](0)
        for k in range(3):
            for l in range(3):
                s += eigvec[k * 3 + i] * mat[k * 3 + l] * eigvec[l * 3 + i]
        eigvals[i] = s

    # Bubble sort eigenvalues in DECREASING order (matching MuJoCo)
    # Also swap eigenvector columns
    for _ in range(3):
        for i in range(2):
            if eigvals[i] < eigvals[i + 1]:
                # Swap eigenvalues
                var tmp = eigvals[i]
                eigvals[i] = eigvals[i + 1]
                eigvals[i + 1] = tmp
                # Swap eigenvector columns
                for r in range(3):
                    var tmp2 = eigvec[r * 3 + i]
                    eigvec[r * 3 + i] = eigvec[r * 3 + (i + 1)]
                    eigvec[r * 3 + (i + 1)] = tmp2

    # Ensure right-handed coordinate system (det(eigvec) > 0)
    # det = eigvec col0 . (col1 x col2)
    var cross_x = eigvec[1 * 3 + 1] * eigvec[2 * 3 + 2] - eigvec[
        2 * 3 + 1
    ] * eigvec[1 * 3 + 2]
    var cross_y = eigvec[2 * 3 + 1] * eigvec[0 * 3 + 2] - eigvec[
        0 * 3 + 1
    ] * eigvec[2 * 3 + 2]
    var cross_z = eigvec[0 * 3 + 1] * eigvec[1 * 3 + 2] - eigvec[
        1 * 3 + 1
    ] * eigvec[0 * 3 + 2]
    var det = (
        eigvec[0 * 3 + 0] * cross_x
        + eigvec[1 * 3 + 0] * cross_y
        + eigvec[2 * 3 + 0] * cross_z
    )
    if det < Scalar[DTYPE](0):
        # Negate last column
        for r in range(3):
            eigvec[r * 3 + 2] = -eigvec[r * 3 + 2]

    # Convert eigenvector matrix to quaternion
    var q = mat3_to_quat(eigvec)

    return (eigvals[0], eigvals[1], eigvals[2], q[0], q[1], q[2], q[3])


# =============================================================================
# Main: compute_inertia_from_geoms
# =============================================================================


fn compute_inertia_from_geoms[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    MAX_EQUALITY: Int,
    CONE_TYPE: Int,
    MAX_TENDON: Int = 0,
    NSITE: Int = 0,
](
    mut model: Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, MAX_EQUALITY, CONE_TYPE, MAX_TENDON, NSITE,
    ],
):
    """Compute body mass/inertia/ipos/iquat from child geoms.

    For each body (1..NBODY-1):
      1. Collect geoms attached to it (geom_body[g] == body_id)
      2. Compute per-geom mass from geom_mass array
      3. Single geom with mass: copy pos/quat/mass/diagonal inertia
      4. Multiple geoms: accumulate full tensor via parallel axis theorem,
         then eigendecompose to get principal axes

    Must be called AFTER Geoms.setup_model (which populates geom arrays and geom_mass).
    """
    # For each body (skip worldbody at index 0)
    for body_id in range(1, NBODY):
        # Collect geoms for this body and count those with non-trivial mass.
        # geom_mass may be -1.0 (no explicit mass) — use density*volume in that case.
        var total_mass = Scalar[DTYPE](0)
        var num_contributing = 0

        # First pass: count contributing geoms and total mass
        for g in range(NGEOM):
            if model.geom_body[g] == body_id:
                var gm = geom_effective_mass[DTYPE](
                    model.geom_type[g],
                    model.geom_mass[g],
                    model.geom_radius[g],
                    model.geom_half_length[g],
                    model.geom_half_x[g],
                    model.geom_half_y[g],
                    model.geom_half_z[g],
                )
                if gm > Scalar[DTYPE](1e-10):
                    num_contributing += 1
                    total_mass += gm

        if num_contributing == 0:
            continue

        if num_contributing == 1:
            # Single geom: find it and copy directly
            for g in range(NGEOM):
                if model.geom_body[g] == body_id:
                    var gm = geom_effective_mass[DTYPE](
                        model.geom_type[g],
                        model.geom_mass[g],
                        model.geom_radius[g],
                        model.geom_half_length[g],
                        model.geom_half_x[g],
                        model.geom_half_y[g],
                        model.geom_half_z[g],
                    )
                    if gm > Scalar[DTYPE](1e-10):
                        # Copy pos as ipos
                        model.body_ipos[body_id * 3 + 0] = model.geom_pos[
                            g * 3 + 0
                        ]
                        model.body_ipos[body_id * 3 + 1] = model.geom_pos[
                            g * 3 + 1
                        ]
                        model.body_ipos[body_id * 3 + 2] = model.geom_pos[
                            g * 3 + 2
                        ]
                        # Copy quat as iquat
                        model.body_iquat[body_id * 4 + 0] = model.geom_quat[
                            g * 4 + 0
                        ]
                        model.body_iquat[body_id * 4 + 1] = model.geom_quat[
                            g * 4 + 1
                        ]
                        model.body_iquat[body_id * 4 + 2] = model.geom_quat[
                            g * 4 + 2
                        ]
                        model.body_iquat[body_id * 4 + 3] = model.geom_quat[
                            g * 4 + 3
                        ]
                        # Set mass
                        model.body_mass[body_id] = gm
                        model.body_inv_mass[body_id] = Scalar[DTYPE](1.0) / gm
                        # Compute diagonal inertia
                        var inertia = geom_inertia[DTYPE](
                            model.geom_type[g],
                            gm,
                            model.geom_radius[g],
                            model.geom_half_length[g],
                            model.geom_half_x[g],
                            model.geom_half_y[g],
                            model.geom_half_z[g],
                        )
                        model.body_inertia[body_id * 3 + 0] = inertia[0]
                        model.body_inertia[body_id * 3 + 1] = inertia[1]
                        model.body_inertia[body_id * 3 + 2] = inertia[2]
                        model.body_inv_inertia[body_id * 3 + 0] = Scalar[DTYPE](
                            1.0
                        ) / inertia[0]
                        model.body_inv_inertia[body_id * 3 + 1] = Scalar[DTYPE](
                            1.0
                        ) / inertia[1]
                        model.body_inv_inertia[body_id * 3 + 2] = Scalar[DTYPE](
                            1.0
                        ) / inertia[2]
                        break
        else:
            # Multiple geoms: accumulate
            # Compute center of mass (mass-weighted average of geom positions)
            var com_x = Scalar[DTYPE](0)
            var com_y = Scalar[DTYPE](0)
            var com_z = Scalar[DTYPE](0)
            for g in range(NGEOM):
                if model.geom_body[g] == body_id:
                    var gm = geom_effective_mass[DTYPE](
                        model.geom_type[g],
                        model.geom_mass[g],
                        model.geom_radius[g],
                        model.geom_half_length[g],
                        model.geom_half_x[g],
                        model.geom_half_y[g],
                        model.geom_half_z[g],
                    )
                    if gm > Scalar[DTYPE](1e-10):
                        com_x += gm * model.geom_pos[g * 3 + 0]
                        com_y += gm * model.geom_pos[g * 3 + 1]
                        com_z += gm * model.geom_pos[g * 3 + 2]
            com_x /= total_mass
            com_y /= total_mass
            com_z /= total_mass

            # Set ipos = center of mass
            model.body_ipos[body_id * 3 + 0] = com_x
            model.body_ipos[body_id * 3 + 1] = com_y
            model.body_ipos[body_id * 3 + 2] = com_z

            # Set mass
            model.body_mass[body_id] = total_mass
            model.body_inv_mass[body_id] = Scalar[DTYPE](1.0) / total_mass

            # Accumulate full 6-element inertia tensor
            var toti = InlineArray[Scalar[DTYPE], 6](fill=Scalar[DTYPE](0))

            for g in range(NGEOM):
                if model.geom_body[g] == body_id:
                    var gm = geom_effective_mass[DTYPE](
                        model.geom_type[g],
                        model.geom_mass[g],
                        model.geom_radius[g],
                        model.geom_half_length[g],
                        model.geom_half_x[g],
                        model.geom_half_y[g],
                        model.geom_half_z[g],
                    )
                    if gm > Scalar[DTYPE](1e-10):
                        # Compute diagonal inertia for this geom
                        var diag = geom_inertia[DTYPE](
                            model.geom_type[g],
                            gm,
                            model.geom_radius[g],
                            model.geom_half_length[g],
                            model.geom_half_x[g],
                            model.geom_half_y[g],
                            model.geom_half_z[g],
                        )

                        # Rotate to global frame
                        var inert_global = InlineArray[Scalar[DTYPE], 6](
                            fill=Scalar[DTYPE](0)
                        )
                        globalinertia(
                            diag[0],
                            diag[1],
                            diag[2],
                            model.geom_quat[g * 4 + 0],
                            model.geom_quat[g * 4 + 1],
                            model.geom_quat[g * 4 + 2],
                            model.geom_quat[g * 4 + 3],
                            inert_global,
                        )

                        # Parallel axis theorem (offset from body CoM)
                        var dx = model.geom_pos[g * 3 + 0] - com_x
                        var dy = model.geom_pos[g * 3 + 1] - com_y
                        var dz = model.geom_pos[g * 3 + 2] - com_z
                        var inert_offset = InlineArray[Scalar[DTYPE], 6](
                            fill=Scalar[DTYPE](0)
                        )
                        offcenter(gm, dx, dy, dz, inert_offset)

                        # Accumulate
                        for j in range(6):
                            toti[j] += inert_global[j] + inert_offset[j]

            # Eigendecompose to get principal axes and moments
            var eig = eig3_symmetric(toti)
            # eig = (eigval0, eigval1, eigval2, qx, qy, qz, qw)
            model.body_inertia[body_id * 3 + 0] = eig[0]
            model.body_inertia[body_id * 3 + 1] = eig[1]
            model.body_inertia[body_id * 3 + 2] = eig[2]
            model.body_inv_inertia[body_id * 3 + 0] = Scalar[DTYPE](
                1.0
            ) / eig[0]
            model.body_inv_inertia[body_id * 3 + 1] = Scalar[DTYPE](
                1.0
            ) / eig[1]
            model.body_inv_inertia[body_id * 3 + 2] = Scalar[DTYPE](
                1.0
            ) / eig[2]
            model.body_iquat[body_id * 4 + 0] = eig[3]
            model.body_iquat[body_id * 4 + 1] = eig[4]
            model.body_iquat[body_id * 4 + 2] = eig[5]
            model.body_iquat[body_id * 4 + 3] = eig[6]


fn compute_inertia_from_geoms_buffer[
    DTYPE: DType,
    NBODY: Int,
    NJOINT: Int,
    NGEOM: Int,
](buffer: HostBuffer[DTYPE], geom_masses: InlineArray[Scalar[DTYPE], NGEOM]):
    """Compute body mass/inertia/ipos/iquat from child geoms (buffer version).

    Same algorithm as compute_inertia_from_geoms but reads geom data from
    the HostBuffer and writes body data back to the buffer.

    Args:
        buffer: GPU host buffer with body and geom data.
        geom_masses: Pre-computed geom masses (from Geoms.compute_masses).
    """
    for body_id in range(1, NBODY):
        var body_off = model_body_offset(body_id)

        # First pass: count contributing geoms and total mass
        var total_mass = Scalar[DTYPE](0)
        var num_contributing = 0

        for g in range(NGEOM):
            var goff = model_geom_offset[NBODY, NJOINT](g)
            var gbody = Int(buffer[goff + GEOM_IDX_BODY])
            if gbody == body_id:
                var gm = geom_masses[g]
                if gm > Scalar[DTYPE](1e-10):
                    num_contributing += 1
                    total_mass += gm

        if num_contributing == 0:
            continue

        if num_contributing == 1:
            # Single geom: find it and copy directly
            for g in range(NGEOM):
                var goff = model_geom_offset[NBODY, NJOINT](g)
                var gbody = Int(buffer[goff + GEOM_IDX_BODY])
                if gbody == body_id and geom_masses[g] > Scalar[DTYPE](1e-10):
                    var gm = geom_masses[g]

                    # Copy pos as ipos
                    buffer[body_off + BODY_IDX_IPOS_X] = buffer[
                        goff + GEOM_IDX_POS_X
                    ]
                    buffer[body_off + BODY_IDX_IPOS_Y] = buffer[
                        goff + GEOM_IDX_POS_Y
                    ]
                    buffer[body_off + BODY_IDX_IPOS_Z] = buffer[
                        goff + GEOM_IDX_POS_Z
                    ]
                    # Copy quat as iquat
                    buffer[body_off + BODY_IDX_IQUAT_X] = buffer[
                        goff + GEOM_IDX_QUAT_X
                    ]
                    buffer[body_off + BODY_IDX_IQUAT_Y] = buffer[
                        goff + GEOM_IDX_QUAT_Y
                    ]
                    buffer[body_off + BODY_IDX_IQUAT_Z] = buffer[
                        goff + GEOM_IDX_QUAT_Z
                    ]
                    buffer[body_off + BODY_IDX_IQUAT_W] = buffer[
                        goff + GEOM_IDX_QUAT_W
                    ]
                    # Set mass
                    buffer[body_off + BODY_IDX_MASS] = gm
                    buffer[body_off + BODY_IDX_INV_MASS] = Scalar[DTYPE](
                        1.0
                    ) / gm
                    # Compute diagonal inertia
                    var gtype = Int(buffer[goff + GEOM_IDX_TYPE])
                    var inertia = geom_inertia[DTYPE](
                        gtype,
                        gm,
                        buffer[goff + GEOM_IDX_RADIUS],
                        buffer[goff + GEOM_IDX_HALF_LENGTH],
                        buffer[goff + GEOM_IDX_HALF_X],
                        buffer[goff + GEOM_IDX_HALF_Y],
                        buffer[goff + GEOM_IDX_HALF_Z],
                    )
                    buffer[body_off + BODY_IDX_IXX] = inertia[0]
                    buffer[body_off + BODY_IDX_IYY] = inertia[1]
                    buffer[body_off + BODY_IDX_IZZ] = inertia[2]
                    buffer[body_off + BODY_IDX_INV_IXX] = Scalar[DTYPE](
                        1.0
                    ) / inertia[0]
                    buffer[body_off + BODY_IDX_INV_IYY] = Scalar[DTYPE](
                        1.0
                    ) / inertia[1]
                    buffer[body_off + BODY_IDX_INV_IZZ] = Scalar[DTYPE](
                        1.0
                    ) / inertia[2]
                    break
        else:
            # Multiple geoms: accumulate
            var com_x = Scalar[DTYPE](0)
            var com_y = Scalar[DTYPE](0)
            var com_z = Scalar[DTYPE](0)
            for g in range(NGEOM):
                var goff = model_geom_offset[NBODY, NJOINT](g)
                var gbody = Int(buffer[goff + GEOM_IDX_BODY])
                if gbody == body_id and geom_masses[g] > Scalar[DTYPE](1e-10):
                    var gm = geom_masses[g]
                    com_x += gm * buffer[goff + GEOM_IDX_POS_X]
                    com_y += gm * buffer[goff + GEOM_IDX_POS_Y]
                    com_z += gm * buffer[goff + GEOM_IDX_POS_Z]
            com_x /= total_mass
            com_y /= total_mass
            com_z /= total_mass

            # Set ipos = center of mass
            buffer[body_off + BODY_IDX_IPOS_X] = com_x
            buffer[body_off + BODY_IDX_IPOS_Y] = com_y
            buffer[body_off + BODY_IDX_IPOS_Z] = com_z

            # Set mass
            buffer[body_off + BODY_IDX_MASS] = total_mass
            buffer[body_off + BODY_IDX_INV_MASS] = Scalar[DTYPE](
                1.0
            ) / total_mass

            # Accumulate full 6-element inertia tensor
            var toti = InlineArray[Scalar[DTYPE], 6](fill=Scalar[DTYPE](0))

            for g in range(NGEOM):
                var goff = model_geom_offset[NBODY, NJOINT](g)
                var gbody = Int(buffer[goff + GEOM_IDX_BODY])
                if gbody == body_id and geom_masses[g] > Scalar[DTYPE](1e-10):
                    var gm = geom_masses[g]
                    var gtype = Int(buffer[goff + GEOM_IDX_TYPE])

                    var diag = geom_inertia[DTYPE](
                        gtype,
                        gm,
                        buffer[goff + GEOM_IDX_RADIUS],
                        buffer[goff + GEOM_IDX_HALF_LENGTH],
                        buffer[goff + GEOM_IDX_HALF_X],
                        buffer[goff + GEOM_IDX_HALF_Y],
                        buffer[goff + GEOM_IDX_HALF_Z],
                    )

                    var inert_global = InlineArray[Scalar[DTYPE], 6](
                        fill=Scalar[DTYPE](0)
                    )
                    globalinertia(
                        diag[0],
                        diag[1],
                        diag[2],
                        buffer[goff + GEOM_IDX_QUAT_X],
                        buffer[goff + GEOM_IDX_QUAT_Y],
                        buffer[goff + GEOM_IDX_QUAT_Z],
                        buffer[goff + GEOM_IDX_QUAT_W],
                        inert_global,
                    )

                    var dx = buffer[goff + GEOM_IDX_POS_X] - com_x
                    var dy = buffer[goff + GEOM_IDX_POS_Y] - com_y
                    var dz = buffer[goff + GEOM_IDX_POS_Z] - com_z
                    var inert_offset = InlineArray[Scalar[DTYPE], 6](
                        fill=Scalar[DTYPE](0)
                    )
                    offcenter(gm, dx, dy, dz, inert_offset)

                    for j in range(6):
                        toti[j] += inert_global[j] + inert_offset[j]

            # Eigendecompose to get principal axes and moments
            var eig = eig3_symmetric(toti)
            buffer[body_off + BODY_IDX_IXX] = eig[0]
            buffer[body_off + BODY_IDX_IYY] = eig[1]
            buffer[body_off + BODY_IDX_IZZ] = eig[2]
            buffer[body_off + BODY_IDX_INV_IXX] = Scalar[DTYPE](1.0) / eig[0]
            buffer[body_off + BODY_IDX_INV_IYY] = Scalar[DTYPE](1.0) / eig[1]
            buffer[body_off + BODY_IDX_INV_IZZ] = Scalar[DTYPE](1.0) / eig[2]
            buffer[body_off + BODY_IDX_IQUAT_X] = eig[3]
            buffer[body_off + BODY_IDX_IQUAT_Y] = eig[4]
            buffer[body_off + BODY_IDX_IQUAT_Z] = eig[5]
            buffer[body_off + BODY_IDX_IQUAT_W] = eig[6]
