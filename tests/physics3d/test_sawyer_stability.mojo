"""Test: Sawyer Reach physics stability with random actions.

Verifies the arm doesn't diverge to NaN over 500 steps of random actions.
State reads go through the fields facade (`env.d`); the mesh-collision
diagnostics read the ModelFields records (`env.mf`).
"""

from std.testing import assert_true, TestSuite
from std.math import isnan
from std.random import seed, random_float64
from mojo_rl.envs.metaworld import SawyerReach
from mojo_rl.envs.metaworld.sawyer_reach_xml import SawyerReachModel
from std.gpu.host import DeviceContext
from mojo_rl.physics3d.fields import ModelFields
from mojo_rl.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
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
    GEOM_IDX_CONTYPE,
    GEOM_IDX_MESH_ID,
    GEOM_IDX_RBOUND,
)
from mojo_rl.core import ContAction


def test_sawyer_no_nan() raises:
    """Run 500 steps of random actions and verify no NaN in arm qpos."""
    print("=== Sawyer Stability Test (500 steps) ===")
    seed(42)

    var env = SawyerReach()
    _ = env.reset()

    var max_qpos: Float64 = 0
    var nan_step = -1
    comptime ACTION_DIM = 4

    # Mesh collision diagnostics from the env's ModelFields records (the
    # legacy CPU Model build was deleted at G4).
    comptime M = SawyerReachModel
    for m in range(16):
        var vnum = Int(env.mf.mesh_meta.data[m * 2 + 1])
        if vnum > 0:
            print("  mesh", m, ": verts=", vnum)
    for g in range(M.NGEOM):
        var go = g * MODEL_GEOM_SIZE
        var mid = Int(env.mf.geoms.data[go + GEOM_IDX_MESH_ID])
        if mid >= 0:
            print("  geom", g, "body=",
                  Int(env.mf.geoms.data[go + GEOM_IDX_BODY]),
                  "mesh_id=", mid,
                  "contype=", Int(env.mf.geoms.data[go + GEOM_IDX_CONTYPE]),
                  "rbound=", Float64(env.mf.geoms.data[go + GEOM_IDX_RBOUND]))

    # Print eGripperBase mesh hull extent (mesh 11, geom 27). The facade's
    # mf is built with NMESH_VERTS=0 (mesh verts unused on the CPU step
    # path), so build a diagnostics-only ModelFields with full mesh capacity.
    var ctx = DeviceContext()
    var mfd = ModelFields[
        DType.float64, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 16 * 256,
    ]()
    M.init_fields[DType.float64, 16 * 256](ctx, mfd)
    if Int(mfd.mesh_meta.data[11 * 2 + 1]) > 0:
        var vadr = Int(mfd.mesh_meta.data[11 * 2 + 0])
        var vnum = Int(mfd.mesh_meta.data[11 * 2 + 1])
        var min_x = Float64(1e10)
        var max_x = Float64(-1e10)
        var min_y = Float64(1e10)
        var max_y = Float64(-1e10)
        var min_z = Float64(1e10)
        var max_z = Float64(-1e10)
        for v in range(vnum):
            var vx = Float64(mfd.mesh_verts.data[(vadr + v) * 3 + 0])
            var vy = Float64(mfd.mesh_verts.data[(vadr + v) * 3 + 1])
            var vz = Float64(mfd.mesh_verts.data[(vadr + v) * 3 + 2])
            if vx < min_x:
                min_x = vx
            if vx > max_x:
                max_x = vx
            if vy < min_y:
                min_y = vy
            if vy > max_y:
                max_y = vy
            if vz < min_z:
                min_z = vz
            if vz > max_z:
                max_z = vz
        print("eGripperBase hull (mesh 11):", vnum, "verts")
        print("  x:", min_x, "to", max_x)
        print("  y:", min_y, "to", max_y)
        print("  z:", min_z, "to", max_z)

    # Print geom 27 (eGripperBase) local pose from the records
    comptime g27 = 27 * MODEL_GEOM_SIZE
    print("geom 27 local pos:",
          Float64(env.mf.geoms.data[g27 + GEOM_IDX_POS_X]),
          Float64(env.mf.geoms.data[g27 + GEOM_IDX_POS_Y]),
          Float64(env.mf.geoms.data[g27 + GEOM_IDX_POS_Z]))
    print("geom 27 local quat:",
          Float64(env.mf.geoms.data[g27 + GEOM_IDX_QUAT_X]),
          Float64(env.mf.geoms.data[g27 + GEOM_IDX_QUAT_Y]),
          Float64(env.mf.geoms.data[g27 + GEOM_IDX_QUAT_Z]),
          Float64(env.mf.geoms.data[g27 + GEOM_IDX_QUAT_W]))
    print("body 23 xpos:", Float64(env.d.xpos.data[23*3+0]),
          Float64(env.d.xpos.data[23*3+1]), Float64(env.d.xpos.data[23*3+2]))
    print("body 23 xquat:", Float64(env.d.xquat.data[23*4+0]),
          Float64(env.d.xquat.data[23*4+1]), Float64(env.d.xquat.data[23*4+2]),
          Float64(env.d.xquat.data[23*4+3]))

    # Print object geom details
    for g in range(M.NGEOM):
        var go = g * MODEL_GEOM_SIZE
        if Int(env.mf.geoms.data[go + GEOM_IDX_BODY]) == 33:
            print("  obj geom", g,
                  "type=", Int(env.mf.geoms.data[go + GEOM_IDX_TYPE]),
                  "hl=", Float64(env.mf.geoms.data[go + GEOM_IDX_HALF_LENGTH]),
                  "r=", Float64(env.mf.geoms.data[go + GEOM_IDX_RADIUS]),
                  "rbound=", Float64(env.mf.geoms.data[go + GEOM_IDX_RBOUND]))

    for step in range(500):
        var action = ContAction[ACTION_DIM]()
        for i in range(ACTION_DIM):
            action.data[i] = random_float64(-1.0, 1.0)

        _ = env.step(action)

        # Check arm qpos (first 9) for NaN — ignore object free joint
        for i in range(9):
            var q = Float64(env.d.qpos.data[i])
            if isnan(q):
                nan_step = step
                print("NaN detected at step", step, "qpos[", i, "]")
                break
            if abs(q) > max_qpos:
                max_qpos = abs(q)

        if nan_step >= 0:
            break

        if step < 3 or step % 100 == 0:
            var hz = Float64(env.d.xpos.data[24 * 3 + 2])
            var max_vel: Float64 = 0
            for i in range(9):
                var v = abs(Float64(env.d.qvel.data[i]))
                if v > max_vel:
                    max_vel = v
            var obj_z = Float64(env.d.qpos.data[11])
            print(
                "Step", step,
                " hand_z=", hz,
                " obj_z=", obj_z,
                " max_vel=", max_vel,
            )

    assert_true(nan_step == -1, "Physics diverged to NaN!")
    print("PASS: No NaN after 500 steps, max |qpos| =", max_qpos)


def main() raises:
    test_sawyer_no_nan()
