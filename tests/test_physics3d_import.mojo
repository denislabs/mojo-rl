"""Test physics3d module imports."""

from physics3d import (
    # Constants
    dtype,
    BODY_STATE_SIZE_3D,
    IDX_PX, IDX_PY, IDX_PZ,
    IDX_QW, IDX_QX, IDX_QY, IDX_QZ,
    JOINT_HINGE, JOINT_BALL,

    # Layout
    PhysicsLayout3D,
    HopperLayout3D,
    AntLayout3D,

    # State
    PhysicsState3D,

    # Body
    compute_box_inertia,
    compute_sphere_inertia,
    compute_capsule_inertia,

    # Joints
    Hinge3D,
    Ball3D,
    Motor3D,
    PDController,

    # Collision
    Contact3D,
    ContactManifold,
    SpherePlaneCollision,
    CapsulePlaneCollision,
)

from math3d import Vec3, Quat


fn main() raises:
    print("Testing physics3d module imports...")

    # Test constants
    print("BODY_STATE_SIZE_3D:", BODY_STATE_SIZE_3D)

    # Test Vec3 and Quat
    var v = Vec3(1.0, 2.0, 3.0)
    var q = Quat.identity()
    print("Vec3:", v.x, v.y, v.z)
    print("Quat identity:", q.w, q.x, q.y, q.z)

    # Test inertia computation
    var box_inertia = compute_box_inertia(1.0, Vec3(0.5, 0.5, 0.5))
    print("Box inertia:", box_inertia.x, box_inertia.y, box_inertia.z)

    # Test Contact3D
    var contact = Contact3D()
    print("Contact body_a:", contact.body_a)

    # Test ContactManifold
    var manifold = ContactManifold(0, 1)
    print("Manifold bodies:", manifold.body_a, manifold.body_b)

    # Test SpherePlaneCollision
    var sphere_contact = SpherePlaneCollision.detect_ground(
        Vec3(0.0, 0.0, 0.5),  # center
        0.5,                   # radius
        0.0,                   # ground_height
        0,                     # body_idx
    )
    print("Sphere contact depth:", sphere_contact.depth)

    # Test PDController
    var pd = PDController(kp=100.0, kd=10.0, max_force=50.0)
    var torque = pd.compute_torque(target=0.5, current=0.0, velocity=0.0)
    print("PD torque:", torque)

    print("All physics3d imports successful!")
