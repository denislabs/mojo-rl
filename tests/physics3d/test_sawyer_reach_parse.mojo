"""Test: Sawyer Reach-v3 XML parses correctly and model dimensions are sane."""

from std.testing import assert_equal, assert_true, TestSuite
from std.gpu.host import DeviceContext
from mojo_rl.envs.metaworld.sawyer_reach_xml import SawyerReachModel, pm
from mojo_rl.physics3d.fields import Model
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MOCAP,
    MODEL_META_IDX_GRAVITY_Z,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_NEQUALITY,
)


def test_sawyer_reach_dimensions() raises:
    """Verify parsed dimensions match expected Sawyer + Reach structure."""
    print("=== Sawyer Reach-v3 Parse Test ===")
    print("NBODY:", pm.NBODY)
    print("NJOINT:", pm.NJOINT)
    print("NQ:", pm.NQ)
    print("NV:", pm.NV)
    print("NGEOM:", pm.NGEOM)
    print("NACT:", pm.NACT)
    print("NSITE:", pm.NSITE)
    print("NEQ:", pm.NEQ)
    print("TIMESTEP:", pm.TIMESTEP)

    # Sawyer arm: base + controller_box + pedestal_feet + torso + pedestal +
    #   right_arm_base_link + right_l0 + head + right_l1 + right_l2 +
    #   right_l3 + right_l4 + right_l5 + right_l6 + right_hand + hand +
    #   rightclaw + rightpad + leftclaw + leftpad +
    #   tablelink + RetainingWall + mocap + obj = 24 bodies
    # + worldbody = 25
    assert_true(pm.NBODY > 20, "Expected >20 bodies (Sawyer + scene)")

    # 7 arm joints + 2 gripper slides + 1 free (obj) = 10
    assert_true(pm.NJOINT >= 10, "Expected >=10 joints")

    # Free joint: 7 qpos + 7 hinge + 2 slide = 16
    assert_true(pm.NQ >= 16, "Expected NQ >= 16")

    # 1 weld constraint
    assert_equal(pm.NEQ, 1)

    # 2 gripper actuators
    assert_equal(pm.NACT, 2)

    # 3 sites: endEffector + rightEndEffector + leftEndEffector + goal
    assert_true(pm.NSITE >= 3, "Expected >= 3 sites")

    print("\n=== Model Setup Test (spec-direct fields build; G4) ===")
    comptime DTYPE = DType.float64
    comptime M = SawyerReachModel
    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 16 * 256,
    ]()
    M.init_fields[DTYPE, 16 * 256](ctx, mf)

    print("Model built successfully!")
    print("Gravity Z:", mf.meta.data[MODEL_META_IDX_GRAVITY_Z])
    print("Timestep:", mf.meta.data[MODEL_META_IDX_TIMESTEP])
    print("Num equality:", mf.meta.data[MODEL_META_IDX_NEQUALITY])

    # Verify gravity
    assert_true(
        Float64(mf.meta.data[MODEL_META_IDX_GRAVITY_Z]) < -9.0,
        "Expected negative gravity"
    )

    # Verify weld constraint was set up
    assert_equal(Int(mf.meta.data[MODEL_META_IDX_NEQUALITY]), 1)

    # Verify mocap body was detected
    var found_mocap = False
    for i in range(pm.NBODY):
        if mf.bodies.data[i * MODEL_BODY_SIZE + BODY_IDX_MOCAP] != 0:
            found_mocap = True
            print("Mocap body index:", i)
            break
    assert_true(found_mocap, "Expected at least one mocap body")

    print("\n=== All tests passed! ===")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
