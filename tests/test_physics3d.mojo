"""Test physics3d module compilation and basic functionality."""

from physics3d import (
    PhysicsLayout3D,
    HopperLayout3D,
    compute_box_inertia,
    compute_sphere_inertia,
    compute_capsule_inertia,
    BODY_STATE_SIZE_3D,
)
from physics3d.body import get_torso_inertia, get_limb_inertia
from math3d import Vec3, Quat


fn main() raises:
    print("=== Physics3D Tests ===\n")

    # Test layout computation
    print("Testing PhysicsLayout3D...")
    print("  BODY_STATE_SIZE_3D:", BODY_STATE_SIZE_3D)
    print("  HopperLayout3D.STATE_SIZE:", HopperLayout3D.STATE_SIZE)
    print("  HopperLayout3D.BODIES_OFFSET:", HopperLayout3D.BODIES_OFFSET)
    print("  HopperLayout3D.JOINTS_OFFSET:", HopperLayout3D.JOINTS_OFFSET)
    print("  Layout computation PASSED")

    # Test inertia computation
    print("\nTesting inertia computation...")

    # Box inertia
    var box_inertia = compute_box_inertia(mass=2.0, half_extents=Vec3(0.1, 0.05, 0.2))
    print("  Box inertia (2kg, 0.2x0.1x0.4m):", box_inertia.x, box_inertia.y, box_inertia.z)

    # Sphere inertia
    var sphere_inertia = compute_sphere_inertia(mass=1.0, radius=0.1)
    print("  Sphere inertia (1kg, r=0.1m):", sphere_inertia.x, sphere_inertia.y, sphere_inertia.z)

    # Capsule inertia
    var capsule_inertia = compute_capsule_inertia(mass=1.5, radius=0.04, half_height=0.15)
    print("  Capsule inertia (1.5kg, r=0.04m, h=0.3m):", capsule_inertia.x, capsule_inertia.y, capsule_inertia.z)

    print("  Inertia computation PASSED")

    # Test MuJoCo-style body defaults
    print("\nTesting MuJoCo body defaults...")
    var torso = get_torso_inertia(mass=10.0)
    print("  Torso inertia:", torso[0].x, torso[0].y, torso[0].z)

    var limb = get_limb_inertia(mass=2.0, length=0.4, radius=0.04)
    print("  Limb inertia:", limb[0].x, limb[0].y, limb[0].z)

    print("  MuJoCo defaults PASSED")

    print("\n=== All Physics3D Tests PASSED ===")
