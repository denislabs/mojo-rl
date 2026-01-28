"""3D Body utilities including inertia tensor computation.

Provides functions to compute inertia tensors for common shapes
used in MuJoCo-style environments.
"""

from math import pi
from math3d import Vec3


# =============================================================================
# Inertia Tensor Computation
# =============================================================================


fn compute_box_inertia[
    DTYPE: DType
](mass: Scalar[DTYPE], half_extents: Vec3[DTYPE]) -> Vec3[DTYPE]:
    """Compute diagonal inertia tensor for a box.

    Args:
        mass: Total mass of the box.
        half_extents: Half-extents (hx, hy, hz) of the box.

    Returns:
        Vec3 of diagonal inertia elements (Ixx, Iyy, Izz).

    The inertia tensor of a box about its center:
        Ixx = (1/12) * m * (h_y^2 + h_z^2) * 4 = (1/3) * m * (hy^2 + hz^2)
        Iyy = (1/12) * m * (h_x^2 + h_z^2) * 4 = (1/3) * m * (hx^2 + hz^2)
        Izz = (1/12) * m * (h_x^2 + h_y^2) * 4 = (1/3) * m * (hx^2 + hy^2)
    """
    var hx = half_extents.x
    var hy = half_extents.y
    var hz = half_extents.z

    # Full dimensions
    var lx = 2.0 * hx
    var ly = 2.0 * hy
    var lz = 2.0 * hz

    var scale = mass / 12.0
    return Vec3(
        scale * (ly * ly + lz * lz),
        scale * (lx * lx + lz * lz),
        scale * (lx * lx + ly * ly),
    )


fn compute_sphere_inertia[
    DTYPE: DType
](mass: Scalar[DTYPE], radius: Scalar[DTYPE]) -> Vec3[DTYPE]:
    """Compute diagonal inertia tensor for a solid sphere.

    Args:
        mass: Total mass of the sphere.
        radius: Radius of the sphere.

    Returns:
        Vec3 of diagonal inertia elements (all equal for sphere).

    I = (2/5) * m * r^2 for all axes
    """
    var I = (2.0 / 5.0) * mass * radius * radius
    return Vec3(I, I, I)


fn compute_capsule_inertia[
    DTYPE: DType
](
    mass: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    half_height: Scalar[DTYPE],
    axis: Int = 2,
) -> Vec3[DTYPE]:
    """Compute diagonal inertia tensor for a capsule.

    A capsule is a cylinder capped with hemispheres.

    Args:
        mass: Total mass of the capsule.
        radius: Radius of the capsule.
        half_height: Half-height of the cylindrical portion.
        axis: Principal axis (0=X, 1=Y, 2=Z, default Z).

    Returns:
        Vec3 of diagonal inertia elements (Ixx, Iyy, Izz).
    """
    # Volume of capsule: V_cylinder + V_sphere
    # V_cylinder = pi * r^2 * 2h
    # V_sphere = (4/3) * pi * r^3
    var h = 2.0 * half_height
    var r = radius

    var v_cyl = pi * r * r * h
    var v_sphere = (4.0 / 3.0) * pi * r * r * r
    var v_total = v_cyl + v_sphere

    # Distribute mass proportionally
    var m_cyl = mass * v_cyl / v_total
    var m_sphere = mass * v_sphere / v_total

    # Cylinder inertia (about center, aligned with axis)
    # For Z-axis aligned cylinder:
    # Ixx = Iyy = (1/12) * m * (3r^2 + h^2)
    # Izz = (1/2) * m * r^2
    var I_cyl_axial = 0.5 * m_cyl * r * r
    var I_cyl_radial = (1.0 / 12.0) * m_cyl * (3.0 * r * r + h * h)

    # Sphere inertia (two hemispheres at ±half_height)
    # Each hemisphere has inertia (2/5) * m/2 * r^2 about its center
    # Plus parallel axis theorem to shift to capsule center
    var I_hemi = (2.0 / 5.0) * (m_sphere / 2.0) * r * r
    var d = (
        half_height + (3.0 / 8.0) * r
    )  # Distance from capsule center to hemisphere CoM
    var I_hemi_shifted = I_hemi + (m_sphere / 2.0) * d * d  # Parallel axis

    # Combined inertia
    var I_axial = I_cyl_axial + 2.0 * (2.0 / 5.0) * (m_sphere / 2.0) * r * r
    var I_radial = I_cyl_radial + 2.0 * I_hemi_shifted

    # Return based on axis orientation
    if axis == 0:  # X-axis aligned
        return Vec3(I_axial, I_radial, I_radial)
    elif axis == 1:  # Y-axis aligned
        return Vec3(I_radial, I_axial, I_radial)
    else:  # Z-axis aligned (default)
        return Vec3(I_radial, I_radial, I_axial)


fn compute_cylinder_inertia[
    DTYPE: DType
](
    mass: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    half_height: Scalar[DTYPE],
    axis: Int = 2,
) -> Vec3[DTYPE]:
    """Compute diagonal inertia tensor for a solid cylinder.

    Args:
        mass: Total mass of the cylinder.
        radius: Radius of the cylinder.
        half_height: Half-height of the cylinder.
        axis: Principal axis (0=X, 1=Y, 2=Z, default Z).

    Returns:
        Vec3 of diagonal inertia elements (Ixx, Iyy, Izz).
    """
    var h = 2.0 * half_height
    var r = radius

    # Cylinder about its axis: I_axial = (1/2) * m * r^2
    # Cylinder perpendicular: I_radial = (1/12) * m * (3r^2 + h^2)
    var I_axial = 0.5 * mass * r * r
    var I_radial = (1.0 / 12.0) * mass * (3.0 * r * r + h * h)

    if axis == 0:  # X-axis aligned
        return Vec3(I_axial, I_radial, I_radial)
    elif axis == 1:  # Y-axis aligned
        return Vec3(I_radial, I_axial, I_radial)
    else:  # Z-axis aligned (default)
        return Vec3(I_radial, I_radial, I_axial)


fn compute_ellipsoid_inertia[
    DTYPE: DType
](mass: Scalar[DTYPE], radii: Vec3[DTYPE]) -> Vec3[DTYPE]:
    """Compute diagonal inertia tensor for a solid ellipsoid.

    Args:
        mass: Total mass of the ellipsoid.
        radii: Semi-principal axes (a, b, c).

    Returns:
        Vec3 of diagonal inertia elements (Ixx, Iyy, Izz).

    I_xx = (1/5) * m * (b^2 + c^2)
    I_yy = (1/5) * m * (a^2 + c^2)
    I_zz = (1/5) * m * (a^2 + b^2)
    """
    var a = radii.x
    var b = radii.y
    var c = radii.z
    var scale = mass / 5.0

    return Vec3(
        scale * (b * b + c * c),
        scale * (a * a + c * c),
        scale * (a * a + b * b),
    )


# =============================================================================
# Parallel Axis Theorem
# =============================================================================


fn parallel_axis_offset[
    DTYPE: DType
](inertia: Vec3[DTYPE], mass: Scalar[DTYPE], offset: Vec3[DTYPE]) -> Vec3[
    DTYPE
]:
    """Apply parallel axis theorem to shift inertia tensor.

    Args:
        inertia: Original diagonal inertia tensor about CoM.
        mass: Total mass.
        offset: Displacement from original axis to new axis.

    Returns:
        New diagonal inertia tensor about offset axis.

    I'_xx = I_xx + m * (dy^2 + dz^2)
    I'_yy = I_yy + m * (dx^2 + dz^2)
    I'_zz = I_zz + m * (dx^2 + dy^2)
    """
    var dx2 = offset.x * offset.x
    var dy2 = offset.y * offset.y
    var dz2 = offset.z * offset.z

    return Vec3(
        inertia.x + mass * (dy2 + dz2),
        inertia.y + mass * (dx2 + dz2),
        inertia.z + mass * (dx2 + dy2),
    )


# =============================================================================
# Combined Inertia
# =============================================================================


fn combine_inertias[
    DTYPE: DType
](
    inertias: List[Vec3[DTYPE]],
    masses: List[Scalar[DTYPE]],
    positions: List[Vec3[DTYPE]],
) -> Tuple[Vec3[DTYPE], Scalar[DTYPE], Vec3[DTYPE]]:
    """Combine multiple bodies into a single equivalent body.

    Args:
        inertias: List of diagonal inertia tensors for each body.
        masses: List of masses for each body.
        positions: List of CoM positions for each body.

    Returns:
        Tuple of (combined_inertia, total_mass, combined_com).
    """
    var n = len(masses)
    if n == 0:
        return (Vec3[DTYPE].zero(), Scalar[DTYPE](0.0), Vec3[DTYPE].zero())

    # Compute total mass and center of mass
    var total_mass = Scalar[DTYPE](0.0)
    var com = Vec3[DTYPE].zero()

    for i in range(n):
        total_mass += masses[i]
        com += positions[i] * masses[i]

    if total_mass > 1e-10:
        com = com / total_mass

    # Shift each inertia to combined CoM and sum
    var combined_inertia = Vec3[DTYPE].zero()

    for i in range(n):
        var offset = positions[i] - com
        var shifted = parallel_axis_offset(inertias[i], masses[i], offset)
        combined_inertia += shifted

    return (combined_inertia, total_mass, com)


# =============================================================================
# MuJoCo-Style Body Defaults
# =============================================================================


fn get_torso_inertia[
    DTYPE: DType
](mass: Scalar[DTYPE] = Scalar[DTYPE](10.0)) -> Tuple[
    Vec3[DTYPE], Scalar[DTYPE]
]:
    """Get default torso inertia for humanoid-like robots.

    Returns (inertia, actual_mass) for a torso-sized box.
    """
    # Approximate torso as box: 0.3m x 0.2m x 0.5m
    var half_extents = Vec3[DTYPE](0.15, 0.1, 0.25)
    var inertia = compute_box_inertia[DTYPE](mass, half_extents)
    return (inertia, mass)


fn get_limb_inertia[
    DTYPE: DType
](
    mass: Scalar[DTYPE] = Scalar[DTYPE](2.0),
    length: Scalar[DTYPE] = Scalar[DTYPE](0.4),
    radius: Scalar[DTYPE] = Scalar[DTYPE](0.04),
) -> Tuple[Vec3[DTYPE], Scalar[DTYPE]]:
    """Get default limb (thigh/shin) inertia as a capsule.

    Returns (inertia, actual_mass).
    """
    var half_height = length / 2.0 - radius
    if half_height < 0:
        half_height = 0.01

    var inertia = compute_capsule_inertia(mass, radius, half_height, axis=2)
    return (inertia, mass)


fn get_foot_inertia[
    DTYPE: DType
](mass: Scalar[DTYPE] = Scalar[DTYPE](1.0)) -> Tuple[
    Vec3[DTYPE], Scalar[DTYPE]
]:
    """Get default foot inertia as a small box.

    Returns (inertia, actual_mass).
    """
    # Foot as flat box: 0.15m x 0.06m x 0.03m
    var half_extents = Vec3[DTYPE](0.075, 0.03, 0.015)
    var inertia = compute_box_inertia[DTYPE](mass, half_extents)
    return (inertia, mass)
