"""Sawyer settling — the first ROLLOUT gate on a convex manifold contact.

Every other gate on the Phase 6 convex-collider work is SINGLE-STEP at ONE
hand-placed pose: `test_mesh_detection_fields` and `test_sap_fields` teleport
the obj, run FK plus detection, and compare one frame. Nothing checked that a
manifold contact behaves over TIME — that it holds an object at rest instead of
letting it rotate about a point and sink, which is the entire reason MuJoCo
builds a manifold at all. `test_sawyer_stability` runs 500 steps but only
asserts no NaN, so it would pass with the manifold completely wrong.

WHAT THIS GATES: the obj cylinder settling onto the table box under gravity.
MuJoCo resolves that with a 5-row CYLINDER x BOX manifold, which is exactly the
pair Phase 6 re-routed from `cylinder_box`'s capsule reduction to `mjc_Convex`'s
GJK+EPA (`d93b0a29`) and then gave perturbed extra points. The old reduction was
wrong by exactly -r = -0.02 m, so it would rest the obj a full 2 cm off or eject
it; this gate measures the resting height to 1 mm.

⚠ THIS DOES NOT GATE THE MESH MANIFOLD, and the distinction matters because the
mesh work is what motivated the rollout. Measured on the reference: settling
from four different obj heights (0.02, 0.15, 0.20, 0.28) the obj ALWAYS falls to
the table and comes to rest on a cylinder/box contact — no mesh contact is
load-bearing at equilibrium anywhere in this scene, because sawyer's only meshes
are the arm links and the gripper, all held above by the mocap weld. A mesh
settling fixture needs a purpose-built model (a static mesh with a free body
resting ON it), which the comptime `parse_xml` path cannot express since mesh
assets need file I/O. Scoped, not built.

⚠ THE SINGLE-FRAME FIXTURE POSE IS NOT USABLE HERE. `test_mesh_detection_fields`
teleports the obj to z=0.28, 2.77 cm INSIDE the gripper hull. Stepping from
there the reference ejects it to [11.46, -4.47, 17.65] with |qvel| = 25.8 — a
deep-penetration blowout, not a settle. A pose chosen to exercise detection is
not automatically a pose you can integrate from.

WHY EQUILIBRIUM AND NOT TRAJECTORY: contact-rich trajectories separate
exponentially, which is why `test_hopper_vs_dm_control` compares state only
while `ncon == 0` and drops to aggregates once contact starts. A resting
configuration is not chaotic — it is determined by geometry, and the reference
converges to the same rest pose from all four starts above. So this gate asserts
the equilibrium, not the path to it.

Run: pixi run mojo run -I . tests/physics3d/test_sawyer_settle_vs_mujoco.mojo
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.core.cont_action import ContAction
from mojo_rl.envs.metaworld import SawyerReach
from mojo_rl.physics3d.gpu.constants import META_IDX_NUM_CONTACTS

comptime REF_XML = (
    "references/Metaworld-master/metaworld/assets/sawyer_xyz/"
    "sawyer_reach_v3.xml"
)

comptime ACTION_DIM = 4
comptime FRAME_SKIP = 5  # SawyerReachConfig.FRAME_SKIP
comptime N_ENV_STEPS = 160  # x5 model steps x 0.0025 s = 2.0 s

# Free-joint qpos slots for the obj: [x y z qw qx qy qz] at qposadr 9.
comptime OBJ_QPOS = 9
# ...and its 6 dofs. nv = 15 = 7 arm hinges + 2 claw slides + 6 free, so the
# obj's velocity block starts at 9. Written out rather than derived from a
# tensor length, because a wrong offset here would read ARM velocities and
# quietly report "settled".
comptime OBJ_DOF = 9

# --- Measured on MuJoCo 3.10.0 at this protocol, 2026-08-10 ---------------
# Rest pose is start-independent: obj released at z = 0.02, 0.15 and 0.20 all
# converge here, which is what makes it safe to assert.
comptime MJ_REST_X = 0.0
comptime MJ_REST_Y = 0.6
comptime MJ_REST_Z = 0.019973
# 1 mm. The defect this is built to catch moves the resting height by r = 20 mm
# (the capsule reduction's -r error), so 1 mm is 20x inside the failure it
# guards while leaving room for solver-iteration differences at equilibrium.
comptime REST_TOL = 1e-3
# "Settled" — the reference's own residual is ~5e-3 after 2 s (the manifold is
# still micro-adjusting), so this bounds motion rather than demanding zero.
comptime SETTLED_VEL = 5e-2
# The manifold must actually be a manifold. MuJoCo emits 5 rows here; requiring
# >= 2 catches a collapse to a single point without pinning a count that
# `isDistinctContact` can legitimately shave by one (see the 4-vs-5 note in
# `test_mesh_detection_fields`).
comptime MIN_MANIFOLD_ROWS = 2


def _setup() raises -> PythonObject:
    var mujoco = Python.import_module("mujoco")
    var model = mujoco.MjModel.from_xml_path(String(REF_XML))
    var data = mujoco.MjData(model)
    return Python.tuple(mujoco, model, data)


def _reset_reference(mujoco: PythonObject, model: PythonObject,
                     data: PythonObject) raises:
    """Mirror `SawyerReachConfig.custom_reset_cpu` exactly.

    ⚠ THE MOCAP MUST BE PLACED BEFORE THE FIRST STEP. Setting it after an
    `mj_forward`, or leaving it at the origin, makes the weld yank the arm
    across the workspace — measured, the hand ends at x = 0.99 regardless of
    the target and the scene is meaningless. The port's reset sets mocap first,
    so this does too.
    """
    mujoco.mj_resetData(model, data)
    # Arm pose: MetaWorld's post-`_reset_hand` warmup values, copied from the
    # port's reset so both sides start from the same configuration.
    data.qpos[0] = 1.889288
    data.qpos[1] = -0.575769
    data.qpos[2] = -0.976659
    data.qpos[3] = 1.641991
    data.qpos[4] = 0.942860
    data.qpos[5] = 1.043696
    data.qpos[6] = 2.292833
    data.qpos[7] = 0.0
    data.qpos[8] = 0.0
    # Obj free joint: on the table, upright.
    data.qpos[OBJ_QPOS + 0] = 0.0
    data.qpos[OBJ_QPOS + 1] = 0.6
    data.qpos[OBJ_QPOS + 2] = 0.02
    data.qpos[OBJ_QPOS + 3] = 1.0
    data.qpos[OBJ_QPOS + 4] = 0.0
    data.qpos[OBJ_QPOS + 5] = 0.0
    data.qpos[OBJ_QPOS + 6] = 0.0
    # hand_init_pos, and MetaWorld's fixed hand orientation.
    data.mocap_pos[0][0] = 0.0
    data.mocap_pos[0][1] = 0.6
    data.mocap_pos[0][2] = 0.2
    data.mocap_quat[0][0] = 0.0
    data.mocap_quat[0][1] = 1.0
    data.mocap_quat[0][2] = 0.0
    data.mocap_quat[0][3] = 1.0
    mujoco.mj_forward(model, data)


def test_sawyer_obj_settles_where_mujoco_does() raises:
    """The obj must come to rest where the reference rests it, and STAY."""
    var handle = _setup()
    var mujoco = handle[0]
    var model = handle[1]
    var data = handle[2]
    _reset_reference(mujoco, model, data)

    var env = SawyerReach()
    _ = env.reset()

    # Zero action: the mocap delta is zero, so the hand holds `hand_init_pos`
    # on both sides and the obj settles under gravity alone.
    var action = ContAction[ACTION_DIM]()
    for i in range(ACTION_DIM):
        action.data[i] = 0.0

    for _ in range(N_ENV_STEPS):
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(model, data)
        _ = env.step(action)
    mujoco.mj_forward(model, data)

    var ox = Float64(env.d.qpos.data[OBJ_QPOS + 0])
    var oy = Float64(env.d.qpos.data[OBJ_QPOS + 1])
    var oz = Float64(env.d.qpos.data[OBJ_QPOS + 2])
    var rx = Float64(py=data.qpos[OBJ_QPOS + 0])
    var ry = Float64(py=data.qpos[OBJ_QPOS + 1])
    var rz = Float64(py=data.qpos[OBJ_QPOS + 2])

    print("sawyer settle,", N_ENV_STEPS, "env steps x", FRAME_SKIP, "=",
          N_ENV_STEPS * FRAME_SKIP, "model steps:")
    print("  obj rest ours  [", ox, oy, oz, "]")
    print("  obj rest MuJoCo[", rx, ry, rz, "]")
    print("  anchor         [", MJ_REST_X, MJ_REST_Y, MJ_REST_Z, "]")

    # Guard the ANCHOR against reference drift first: if MuJoCo itself no longer
    # rests here, the hardcoded numbers are stale and every comparison below is
    # measuring the wrong thing.
    var ref_drift = sqrt(
        (rx - MJ_REST_X) * (rx - MJ_REST_X)
        + (ry - MJ_REST_Y) * (ry - MJ_REST_Y)
        + (rz - MJ_REST_Z) * (rz - MJ_REST_Z)
    )
    print("  reference vs recorded anchor:", ref_drift)
    assert_true(
        ref_drift <= REST_TOL,
        String("the REFERENCE no longer rests at the recorded anchor (drift ")
        + String(ref_drift) + "). The MuJoCo build or the model changed —"
        " re-measure the anchor before trusting anything else in this file.",
    )

    var derr = sqrt(
        (ox - rx) * (ox - rx) + (oy - ry) * (oy - ry) + (oz - rz) * (oz - rz)
    )
    print("  ours vs MuJoCo rest-pose error:", derr, " tol", REST_TOL)
    assert_true(
        derr <= REST_TOL,
        String("obj rests ") + String(derr) + " m from MuJoCo's rest pose."
        " A ~0.02 m error in z is the signature of the cylinder being reduced"
        " to a CAPSULE (`cylinder_box`, wrong by exactly -r); an obj that sank"
        " through the table means the manifold is not carrying the load.",
    )

    # Settled, not still moving or drifting.
    var vmax = Float64(0)
    for k in range(6):
        var v = abs(Float64(env.d.qvel.data[OBJ_DOF + k]))
        if v > vmax:
            vmax = v
    print("  max |obj qvel| ours:", vmax, " tol", SETTLED_VEL)
    assert_true(
        vmax <= SETTLED_VEL,
        String("obj has not settled (max |qvel| = ") + String(vmax)
        + "). A resting object that keeps moving is the single-point failure"
        " this gate exists to catch: it rotates about the contact and creeps.",
    )


def test_sawyer_rest_contact_is_a_manifold() raises:
    """At rest the load must be carried by several points, not one."""
    var env = SawyerReach()
    _ = env.reset()
    var action = ContAction[ACTION_DIM]()
    for i in range(ACTION_DIM):
        action.data[i] = 0.0
    for _ in range(N_ENV_STEPS):
        _ = env.step(action)

    var ncon = Int(env.d.meta.data[META_IDX_NUM_CONTACTS])
    print("  contacts at rest:", ncon, " (MuJoCo: 5, all objGeom x table)")
    assert_true(
        ncon >= MIN_MANIFOLD_ROWS,
        String("only ") + String(ncon) + " contact row(s) at rest; MuJoCo"
        " carries this on a 5-row manifold. One row cannot hold a resting"
        " body — it rotates about the point and sinks, which is what"
        " `multi_ccd` exists to prevent.",
    )


def main() raises:
    var suite = TestSuite()
    suite.test[test_sawyer_obj_settles_where_mujoco_does]()
    suite.test[test_sawyer_rest_contact_is_a_manifold]()
    suite^.run()
