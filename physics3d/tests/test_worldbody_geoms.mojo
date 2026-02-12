"""Test WorldBody geom support: SphereGeom, BoxGeom, CapsuleGeom.

Tests:
1. Geom types conform to GeomSpec and have correct field values
2. WorldBody with mixed geoms has correct N count
3. CPU detect_worldbody_contacts detects sphere/capsule/box vs body contacts
4. Per-contact friction propagates from geom to ContactInfo
5. EmptyWorldBody has N=0 (no-op detection)
6. WorldBody.copy_geoms_to_buffer writes correct data to GPU buffer
"""

from math import sqrt
from gpu.host import DeviceContext

from physics3d.model import (
    GeomSpec,
    PlaneGeom,
    SphereGeom,
    BoxGeom,
    CapsuleGeom,
    WorldBody,
    EmptyWorldBody,
    ModelDef,
    Bodies,
    Joints,
)
from physics3d.model.body_spec import SphereBody, CapsuleBody, BoxBody
from physics3d.model.joint_spec import SlideJoint
from physics3d.types import Model, Data
from physics3d.kinematics.forward_kinematics import forward_kinematics
from physics3d.collision.contact_detection import detect_ground_contacts
from physics3d.gpu.constants import (
    model_size,
    model_wgeom_offset,
    WGEOM_IDX_TYPE,
    WGEOM_IDX_POS_X,
    WGEOM_IDX_POS_Y,
    WGEOM_IDX_POS_Z,
    WGEOM_IDX_RADIUS,
    WGEOM_IDX_FRICTION,
    WGEOM_IDX_CONTYPE,
    WGEOM_IDX_CONAFFINITY,
    WGEOM_IDX_SIZE_X,
    WGEOM_IDX_SIZE_Y,
    WGEOM_IDX_SIZE_Z,
)
from physics3d.constants import GEOM_SPHERE, GEOM_BOX, GEOM_CAPSULE, GEOM_PLANE


fn test_geom_types() -> Bool:
    """Test that new geom types have correct compile-time fields."""
    print("Test: Geom type field values")
    var ok = True

    # SphereGeom
    comptime S = SphereGeom[pos_x=1.0, pos_z=0.5, radius=0.3, friction=0.7]
    if S.GEOM_TYPE != GEOM_SPHERE:
        print("  FAIL: SphereGeom.GEOM_TYPE should be", GEOM_SPHERE)
        ok = False
    if S.POS_X != 1.0 or S.POS_Z != 0.5:
        print("  FAIL: SphereGeom position wrong")
        ok = False
    if S.RADIUS != 0.3:
        print("  FAIL: SphereGeom.RADIUS should be 0.3")
        ok = False
    if S.FRICTION != 0.7:
        print("  FAIL: SphereGeom.FRICTION should be 0.7")
        ok = False

    # BoxGeom
    comptime B = BoxGeom[pos_y=2.0, half_x=0.1, half_y=0.2, half_z=0.3, friction=0.6]
    if B.GEOM_TYPE != GEOM_BOX:
        print("  FAIL: BoxGeom.GEOM_TYPE should be", GEOM_BOX)
        ok = False
    if B.SIZE_X != 0.1 or B.SIZE_Y != 0.2 or B.SIZE_Z != 0.3:
        print("  FAIL: BoxGeom sizes wrong")
        ok = False

    # CapsuleGeom
    comptime C = CapsuleGeom[pos_z=1.0, half_length=0.4, radius=0.15, friction=0.8]
    if C.GEOM_TYPE != GEOM_CAPSULE:
        print("  FAIL: CapsuleGeom.GEOM_TYPE should be", GEOM_CAPSULE)
        ok = False
    if C.SIZE_Z != 0.4:
        print("  FAIL: CapsuleGeom.SIZE_Z (half_length) should be 0.4")
        ok = False
    if C.RADIUS != 0.15:
        print("  FAIL: CapsuleGeom.RADIUS should be 0.15")
        ok = False

    if ok:
        print("  PASS")
    return ok


fn test_worldbody_n_count() -> Bool:
    """Test WorldBody N count with mixed geom types."""
    print("Test: WorldBody N count")
    var ok = True

    comptime WB0 = EmptyWorldBody
    if WB0.N != 0:
        print("  FAIL: EmptyWorldBody.N should be 0, got", WB0.N)
        ok = False

    comptime WB1 = WorldBody[PlaneGeom[]]
    if WB1.N != 1:
        print("  FAIL: WorldBody[PlaneGeom].N should be 1")
        ok = False

    comptime WB3 = WorldBody[
        PlaneGeom[friction=0.4],
        SphereGeom[pos_x=2.0, radius=0.3],
        BoxGeom[pos_z=0.5],
    ]
    if WB3.N != 3:
        print("  FAIL: WorldBody[Plane,Sphere,Box].N should be 3")
        ok = False

    if ok:
        print("  PASS")
    return ok


fn test_cpu_worldbody_sphere_contact() -> Bool:
    """Test CPU worldbody sphere contact detection.

    Setup: A single sphere body at (0, 0, 0.5) with radius 0.1.
    A worldbody SphereGeom at (0, 0, 0.5) with radius 0.3.
    They should overlap (dist < 0).
    """
    print("Test: CPU worldbody sphere contact")
    var ok = True

    # Minimal model: 1 body (sphere, r=0.1), 1 slide joint (Z axis)
    comptime TestBodies = Bodies[SphereBody[
        mass=1.0, radius=0.1, pos_z=0.5,
    ]]
    comptime TestJoints = Joints[SlideJoint[
        body_idx=0, axis_z=1.0, range_min=-10.0, range_max=10.0,
    ]]
    comptime NBODY = TestBodies.N  # 1
    comptime NJOINT = TestJoints.N  # 1
    comptime NQ = TestJoints._sum_nq()  # 1
    comptime NV = TestJoints._sum_nv()  # 1
    comptime MAX_CONTACTS = 10

    # Ground plane + sphere obstacle at (0.05, 0, 0.5)
    comptime TestWB = WorldBody[
        PlaneGeom[z=0.0, friction=0.4],
        SphereGeom[pos_x=0.05, pos_z=0.5, radius=0.3, friction=0.7],
    ]

    var model = Model[DType.float64, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    TestBodies.setup_model(model)
    TestJoints.setup_model(model)
    TestWB.setup_model(model)

    var data = Data[DType.float64, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    TestJoints.reset_data(data)
    # Body is at local pos_z=0.5, qpos[0]=0 (slide along Z from that pos)

    forward_kinematics(model, data)

    # First detect ground contacts
    detect_ground_contacts(model, data)
    var ground_contacts = data.num_contacts

    # Now detect worldbody contacts
    TestWB.detect_contacts(model, data)

    var total_contacts = data.num_contacts
    var wg_contacts = total_contacts - ground_contacts

    if wg_contacts < 1:
        print("  FAIL: Expected at least 1 worldbody contact, got", wg_contacts)
        ok = False
    else:
        # Check per-contact friction
        var wg_contact = data.contacts[ground_contacts]
        if wg_contact.friction != 0.7:
            print(
                "  FAIL: worldbody contact friction should be 0.7, got",
                wg_contact.friction,
            )
            ok = False
        if wg_contact.body_b != -1:
            print(
                "  FAIL: worldbody contact body_b should be -1, got",
                wg_contact.body_b,
            )
            ok = False
        if wg_contact.dist >= 0.0:
            print("  FAIL: contact dist should be < 0 (penetrating)")
            ok = False

    if ok:
        print("  PASS (", wg_contacts, "worldbody contacts detected)")
    return ok


fn test_empty_worldbody_noop() -> Bool:
    """Test that EmptyWorldBody detection is a no-op."""
    print("Test: EmptyWorldBody is no-op")
    var ok = True

    comptime TestBodies = Bodies[SphereBody[mass=1.0, radius=0.1, pos_z=0.5]]
    comptime TestJoints = Joints[SlideJoint[
        body_idx=0, axis_z=1.0, range_min=-10.0, range_max=10.0,
    ]]
    comptime NBODY = TestBodies.N
    comptime NJOINT = TestJoints.N
    comptime NQ = TestJoints._sum_nq()
    comptime NV = TestJoints._sum_nv()
    comptime MAX_CONTACTS = 10

    var model = Model[DType.float64, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    TestBodies.setup_model(model)
    TestJoints.setup_model(model)

    var data = Data[DType.float64, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    TestJoints.reset_data(data)
    forward_kinematics(model, data)

    var before = data.num_contacts
    EmptyWorldBody.detect_contacts(model, data)
    var after = data.num_contacts

    if after != before:
        print("  FAIL: EmptyWorldBody should not add contacts")
        ok = False

    if ok:
        print("  PASS")
    return ok


fn test_copy_geoms_to_buffer() -> Bool:
    """Test WorldBody.copy_geoms_to_buffer writes correct data."""
    print("Test: copy_geoms_to_buffer")
    var ok = True

    comptime TestWB = WorldBody[
        PlaneGeom[z=0.0, friction=0.4],
        SphereGeom[pos_x=2.0, pos_z=0.5, radius=0.3, friction=0.7],
        BoxGeom[pos_y=1.0, half_x=0.1, half_y=0.2, half_z=0.3, friction=0.6],
    ]

    comptime NBODY = 1
    comptime NJOINT = 1
    comptime NWGEOM = TestWB.N  # 3

    # Allocate buffer large enough for bodies + joints + wgeoms + metadata
    comptime BUF_SIZE = model_size[NBODY, NJOINT, NWGEOM]()

    try:
        var ctx = DeviceContext()
        var buffer = ctx.enqueue_create_host_buffer[DType.float32](BUF_SIZE)
        for i in range(BUF_SIZE):
            buffer[i] = Float32(0)

        TestWB.copy_geoms_to_buffer[DType.float32, NBODY, NJOINT](buffer)

        # Check geom 0 (PlaneGeom)
        var off0 = model_wgeom_offset[NBODY, NJOINT](0)
        if Int(buffer[off0 + WGEOM_IDX_TYPE]) != GEOM_PLANE:
            print("  FAIL: geom 0 type should be GEOM_PLANE")
            ok = False

        # Check geom 1 (SphereGeom)
        var off1 = model_wgeom_offset[NBODY, NJOINT](1)
        if Int(buffer[off1 + WGEOM_IDX_TYPE]) != GEOM_SPHERE:
            print("  FAIL: geom 1 type should be GEOM_SPHERE")
            ok = False
        if buffer[off1 + WGEOM_IDX_POS_X] != Float32(2.0):
            print("  FAIL: geom 1 pos_x should be 2.0")
            ok = False
        if buffer[off1 + WGEOM_IDX_POS_Z] != Float32(0.5):
            print("  FAIL: geom 1 pos_z should be 0.5")
            ok = False
        if buffer[off1 + WGEOM_IDX_RADIUS] != Float32(0.3):
            print("  FAIL: geom 1 radius should be 0.3")
            ok = False
        if buffer[off1 + WGEOM_IDX_FRICTION] != Float32(0.7):
            print("  FAIL: geom 1 friction should be 0.7")
            ok = False

        # Check geom 2 (BoxGeom)
        var off2 = model_wgeom_offset[NBODY, NJOINT](2)
        if Int(buffer[off2 + WGEOM_IDX_TYPE]) != GEOM_BOX:
            print("  FAIL: geom 2 type should be GEOM_BOX")
            ok = False
        if buffer[off2 + WGEOM_IDX_POS_Y] != Float32(1.0):
            print("  FAIL: geom 2 pos_y should be 1.0")
            ok = False
        if buffer[off2 + WGEOM_IDX_SIZE_X] != Float32(0.1):
            print("  FAIL: geom 2 size_x should be 0.1")
            ok = False
        if buffer[off2 + WGEOM_IDX_SIZE_Y] != Float32(0.2):
            print("  FAIL: geom 2 size_y should be 0.2")
            ok = False
        if buffer[off2 + WGEOM_IDX_SIZE_Z] != Float32(0.3):
            print("  FAIL: geom 2 size_z should be 0.3")
            ok = False
    except e:
        print("  FAIL: GPU context error")
        ok = False

    if ok:
        print("  PASS")
    return ok


fn test_per_contact_friction_ground() -> Bool:
    """Test that detect_ground_contacts writes per-contact friction."""
    print("Test: Per-contact friction (ground)")
    var ok = True

    comptime TestBodies = Bodies[SphereBody[mass=1.0, radius=0.2, pos_z=0.15]]
    comptime TestJoints = Joints[SlideJoint[
        body_idx=0, axis_z=1.0, range_min=-10.0, range_max=10.0,
    ]]
    comptime NBODY = TestBodies.N
    comptime NJOINT = TestJoints.N
    comptime NQ = TestJoints._sum_nq()
    comptime NV = TestJoints._sum_nv()
    comptime MAX_CONTACTS = 10

    comptime TestWB = WorldBody[PlaneGeom[z=0.0, friction=0.55]]

    var model = Model[DType.float64, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    TestBodies.setup_model(model)
    TestJoints.setup_model(model)
    TestWB.setup_model(model)

    var data = Data[DType.float64, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    TestJoints.reset_data(data)
    forward_kinematics(model, data)

    detect_ground_contacts(model, data)

    if data.num_contacts > 0:
        var friction = data.contacts[0].friction
        if friction != 0.55:
            print(
                "  FAIL: ground contact friction should be 0.55, got", friction
            )
            ok = False
    else:
        print("  FAIL: expected ground contacts (sphere at z=0.15, r=0.2)")
        ok = False

    if ok:
        print("  PASS")
    return ok


fn main():
    print("=== WorldBody Geom Tests ===\n")

    var all_ok = True
    all_ok = test_geom_types() and all_ok
    all_ok = test_worldbody_n_count() and all_ok
    all_ok = test_cpu_worldbody_sphere_contact() and all_ok
    all_ok = test_empty_worldbody_noop() and all_ok
    all_ok = test_copy_geoms_to_buffer() and all_ok
    all_ok = test_per_contact_friction_ground() and all_ok

    print()
    if all_ok:
        print("All WorldBody geom tests PASSED!")
    else:
        print("Some tests FAILED!")
