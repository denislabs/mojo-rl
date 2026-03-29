"""Test: Sawyer Reach physics stability with random actions.

Verifies the arm doesn't diverge to NaN over 500 steps of random actions.
"""

from std.testing import assert_true, TestSuite
from std.math import isnan
from std.random import seed, random_float64
from mojo_rl.envs.metaworld import SawyerReach
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

    # Print mesh collision info
    print("num_meshes:", env.model.num_meshes)
    for m in range(env.model.num_meshes):
        print("  mesh", m, ": verts=", env.model.mesh_vertnum[m])
    for g in range(len(env.model.geom_mesh_id)):
        if env.model.geom_mesh_id[g] >= 0:
            print("  geom", g, "body=", env.model.geom_body[g],
                  "mesh_id=", env.model.geom_mesh_id[g],
                  "contype=", env.model.geom_contype[g],
                  "rbound=", Float64(env.model.geom_rbound[g]))

    # Print eGripperBase mesh hull extent (mesh 11, geom 27)
    if env.model.num_meshes > 11:
        var vadr = env.model.mesh_vertadr[11]
        var vnum = env.model.mesh_vertnum[11]
        var min_x = Float64(1e10)
        var max_x = Float64(-1e10)
        var min_y = Float64(1e10)
        var max_y = Float64(-1e10)
        var min_z = Float64(1e10)
        var max_z = Float64(-1e10)
        for v in range(vnum):
            var vx = Float64(env.model.mesh_vert[vadr + v * 3 + 0])
            var vy = Float64(env.model.mesh_vert[vadr + v * 3 + 1])
            var vz = Float64(env.model.mesh_vert[vadr + v * 3 + 2])
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

    # Print geom 27 (eGripperBase) world position
    print("geom 27 local pos:", Float64(env.model.geom_pos[27*3+0]),
          Float64(env.model.geom_pos[27*3+1]), Float64(env.model.geom_pos[27*3+2]))
    print("geom 27 local quat:", Float64(env.model.geom_quat[27*4+0]),
          Float64(env.model.geom_quat[27*4+1]), Float64(env.model.geom_quat[27*4+2]),
          Float64(env.model.geom_quat[27*4+3]))
    print("body 23 xpos:", Float64(env.data.xpos[23*3+0]),
          Float64(env.data.xpos[23*3+1]), Float64(env.data.xpos[23*3+2]))
    print("body 23 xquat:", Float64(env.data.xquat[23*4+0]),
          Float64(env.data.xquat[23*4+1]), Float64(env.data.xquat[23*4+2]),
          Float64(env.data.xquat[23*4+3]))

    # Print object geom details
    for g in range(len(env.model.geom_type)):
        var gb = env.model.geom_body[g]
        if gb == 33:
            print("  obj geom", g,
                  "type=", env.model.geom_type[g],
                  "hl=", Float64(env.model.geom_half_length[g]),
                  "r=", Float64(env.model.geom_radius[g]),
                  "rbound=", Float64(env.model.geom_rbound[g]))

    for step in range(500):
        var action = ContAction[ACTION_DIM]()
        for i in range(ACTION_DIM):
            action.data[i] = random_float64(-1.0, 1.0)

        _ = env.step(action)

        # Check arm qpos (first 9) for NaN — ignore object free joint
        for i in range(9):
            var q = Float64(env.data.qpos[i])
            if isnan(q):
                nan_step = step
                print("NaN detected at step", step, "qpos[", i, "]")
                break
            if abs(q) > max_qpos:
                max_qpos = abs(q)

        if nan_step >= 0:
            break

        if step == 0:
            print("Contacts after step", step, ":", env.data.num_contacts)
            for c in range(env.data.num_contacts):
                var ct = env.data.contacts[c]
                print("  c", c, "body_a=", ct.body_a, "body_b=", ct.body_b,
                      "dist=", Float64(ct.dist), "nz=", Float64(ct.normal_z))
        if step < 3 or step % 100 == 0:
            var hx = Float64(env.data.xpos[24 * 3 + 0])
            var hy = Float64(env.data.xpos[24 * 3 + 1])
            var hz = Float64(env.data.xpos[24 * 3 + 2])
            var mx = Float64(env.data.mocap_pos[32 * 3 + 0])
            var my = Float64(env.data.mocap_pos[32 * 3 + 1])
            var mz = Float64(env.data.mocap_pos[32 * 3 + 2])
            var max_vel: Float64 = 0
            for i in range(9):
                var v = abs(Float64(env.data.qvel[i]))
                if v > max_vel:
                    max_vel = v
            var obj_z = Float64(env.data.qpos[11])
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
