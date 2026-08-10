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
from mojo_rl.physics3d.constants import GEOM_MESH, GEOM_CYLINDER
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS,
    MODEL_GEOM_SIZE,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    CONTACT_SIZE,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
)

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

# ⚠⚠ DEFECT 28 RATCHET — THIS IS NOT A TOLERANCE, IT IS A MEASURED BUG.
#
# The mocap WELD does not hold the hand. MuJoCo tracks the welded body to the
# mocap target within 0.4 mm; ours sags to 77.6 mm away — 57 mm in y and 56 mm
# in z, i.e. downward under gravity. Measured 2026-08-10:
#
#     mocap target   [ 0.0      0.6      0.2     ]
#     MuJoCo         [-0.000026 0.600395 0.196231]   0.4 mm from target
#     ours           [-0.001391 0.542980 0.144079]   77.6 mm from MuJoCo
#
# Both sides use an identity weld relpose (MetaWorld zeroes it on the reference
# via `reset_mocap_welds`; ours never applies the XML's, or the hand would be a
# METRE out), so this is not the relpose trap — it is the equality solve not
# holding the constraint.
#
# ⚠ THIS IS THE PATH EVERY METAWORLD ACTION DRIVES. SawyerReach's action space
# is a mocap position delta; if the weld does not track, the commanded hand
# pose is not the achieved one and every Phase 7 manipulation policy is
# learning against a mis-actuated arm. It is invisible to every other gate
# because the obj rests on the table, decoupled from the arm — which is how the
# settling comparison agreed to 0.39 um while this was 77.6 mm wrong.
#
# Pinned at the measurement + 1% so it cannot silently worsen. It must come
# DOWN to ~1e-3 (the reference's own tracking error), never be raised to pass.
comptime HAND_TOL = 7.84e-2


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

    # ⚠⚠ ZERO THE WELD RELPOSE — MetaWorld's `reset_mocap_welds`, and WITHOUT
    # IT THIS REFERENCE IS NOT THE ENVIRONMENT WE PORT. sawyer_reach_v3's weld
    # ships an `eq_data` relpose of pos (1.1355, 0.1603, 0.317) and quat
    # (0.64279, -0.76604, 0, 0), so the hand tracks `mocap (x) relpose` — over a
    # metre away — not the mocap itself. Left as-is, the arm is flung across the
    # workspace (hand ends at x ~ 0.99-1.19 against a target of 0.0) and drags
    # the obj 12 mm off its rest pose.
    #
    # That cost real time: the 12 mm gap and a 2.24 rad arm divergence looked
    # exactly like a defect in our weld/mocap path, and I was one step from
    # filing it as one. With the relpose zeroed the reference's hand tracks the
    # mocap to 0.4 mm and its obj rest pose matches ours EXACTLY. The engine was
    # right; the reference protocol was incomplete.
    for i in range(Int(py=model.neq)):
        if Int(py=model.eq_type[i]) == 1:  # mjEQ_WELD
            for k in range(11):
                model.eq_data[i][k] = 0.0
            model.eq_data[i][6] = 1.0  # relpose quat w
            model.eq_data[i][10] = 1.0  # torquescale
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
    # ⚠ WXYZ HERE, XYZW IN THE PORT — THE SAME ORIENTATION, SPELLED DIFFERENTLY.
    # MuJoCo's `mocap_quat` is (w,x,y,z); `custom_reset_cpu` stores (x,y,z,w)
    # and writes (0,1,0,1), i.e. MetaWorld's wxyz [1,0,1,0]. Copying the port's
    # literal into this field verbatim gives w=0,x=1,y=0,z=1 — a 180-degree
    # rotation instead of 90, so the two sides held the hand in DIFFERENT
    # orientations. This gate passed anyway because the obj it measures is
    # decoupled from the arm, which is exactly why the arm is now compared too.
    data.mocap_quat[0][0] = 1.0
    data.mocap_quat[0][1] = 0.0
    data.mocap_quat[0][2] = 1.0
    data.mocap_quat[0][3] = 0.0
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

    # --- ARM AGREEMENT — printed BEFORE any assert, deliberately -------------
    # qpos[0..8] are the 7 arm hinges and 2 claw slides, driven by the mocap
    # WELD equality. Nothing in this repo compares that constraint against
    # MuJoCo over a rollout, and the obj comparison is blind to it: the obj
    # falls to the table and rests there whatever the arm does. That blindness
    # is exactly how a mismatched hand orientation survived here.
    #
    # This prints above the asserts so a failing gate still reports the
    # diagnostic that explains it — a measurement that only runs when
    # everything already passes cannot tell you why anything failed.
    var arm_err = Float64(0)
    var arm_worst = 0
    for i in range(9):
        var e = abs(Float64(py=data.qpos[i]) - Float64(env.d.qpos.data[i]))
        if e > arm_err:
            arm_err = e
            arm_worst = i
    print(
        "  max |arm qpos| ours vs MuJoCo:", arm_err, " at joint", arm_worst,
        " (DIAGNOSTIC ONLY — see below)",
    )

    # ⚠ JOINT ANGLES ARE THE WRONG QUANTITY TO ASSERT, and the measurement is
    # what showed it: 0.271 rad at joint 4 while the hand poses agree to
    # sub-millimetre. Sawyer's arm is 7-DOF under a 6-DOF weld, so the
    # constraint leaves a ONE-DIMENSIONAL NULL SPACE — two solvers can land on
    # different joint angles for the SAME hand pose, and neither is wrong.
    # Gating qpos here would pin solver bookkeeping and fail on any legitimate
    # change to the equality solve.
    #
    # The hand POSE is the physical quantity, and it is what a mesh contact
    # fixture depends on: where the gripper actually is decides whether it
    # touches anything.
    #
    # ⚠ COMPARE THE BODY THE WELD ACTUALLY CONSTRAINS — `eq_obj2id`, not the
    # body carrying the eGripperBase mesh. Those are different (23 vs 24): the
    # mesh body is a NEIGHBOUR of the welded one, so the arm's null-space
    # freedom can move it while the weld is perfectly satisfied. Measured on
    # geom 27's body the gap reads 62 mm, which invites a defect report about a
    # constraint that is doing its job.
    #
    # Body ids DO line up between the two here (geom 27 -> body 23 on both
    # sides), but they are still resolved rather than assumed, because geom
    # order is XML order for us and MuJoCo sorts by body id — an alignment that
    # happens to hold for sawyer is not one to rely on.
    var mesh_body = Int(
        env.mf.geoms.data[27 * MODEL_GEOM_SIZE + GEOM_IDX_BODY]
    )
    var hand_gtype = Int(
        env.mf.geoms.data[27 * MODEL_GEOM_SIZE + GEOM_IDX_TYPE]
    )
    assert_true(
        hand_gtype == GEOM_MESH,
        "geom 27 is not a MESH — the eGripperBase index moved and this hand"
        " comparison is measuring some other body",
    )
    var mj_mesh_body = Int(py=model.geom_bodyid[27])
    print(
        "  eGripperBase body ours", mesh_body, " MuJoCo", mj_mesh_body,
        " ours [", Float64(env.d.xpos.data[mesh_body * 3 + 0]),
        Float64(env.d.xpos.data[mesh_body * 3 + 1]),
        Float64(env.d.xpos.data[mesh_body * 3 + 2]), "]",
    )
    var hand_body = Int(py=model.eq_obj2id[0])
    var mj_hand_body = hand_body
    var hx = Float64(env.d.xpos.data[hand_body * 3 + 0])
    var hy = Float64(env.d.xpos.data[hand_body * 3 + 1])
    var hz = Float64(env.d.xpos.data[hand_body * 3 + 2])
    var mhx = Float64(py=data.xpos[mj_hand_body][0])
    var mhy = Float64(py=data.xpos[mj_hand_body][1])
    var mhz = Float64(py=data.xpos[mj_hand_body][2])
    var hand_err = sqrt(
        (hx - mhx) * (hx - mhx)
        + (hy - mhy) * (hy - mhy)
        + (hz - mhz) * (hz - mhz)
    )
    print("  WELDED body (eq_obj2id)", hand_body)
    print("    ours  [", hx, hy, hz, "]")
    print("    MuJoCo[", mhx, mhy, mhz, "]")
    print("    mocap target [ 0.0 0.6 0.2 ]")
    print("  welded-body pose err:", hand_err, " tol", HAND_TOL)
    assert_true(
        hand_err <= HAND_TOL,
        String("the WELD-held gripper sits ") + String(hand_err)
        + " m from MuJoCo's. This is the mocap-weld path every MetaWorld"
        " action drives, and nothing else in the repo compares it against the"
        " reference. ⚠ Check the weld RELPOSE is zeroed on the reference side"
        " (MetaWorld's `reset_mocap_welds`) before suspecting the engine — a"
        " non-identity relpose flings the arm a metre and looks exactly like"
        " an engine defect.",
    )

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


def test_env_facade_collides_meshes() raises:
    """POSITIVE CONTROL: a mesh contact must be reachable through the env.

    ⚠ THIS EXISTS BECAUSE THE TWO GATES ABOVE CANNOT SEE THE BUG THEY WERE
    WRITTEN AFTER. `Phyics3dEnv` hardcoded `NMESH_VERTS = 0`, so every mesh
    geom in every environment emitted no contact — both narrow phases guard
    their mesh branch with `comptime if NMESH_VERTS > 0`. Enabling meshes left
    the settling numbers BIT-IDENTICAL (rest error 3.859900638120598e-07 either
    way), because no mesh row is load-bearing at that equilibrium. Which is the
    correct physics, and also indistinguishable from the fix doing nothing.

    So assert the mesh path directly: put the obj inside the gripper hull and
    require a contact between the obj body and a mesh-carrying body. If this
    fails while the settle tests pass, mesh collision is off again.

    Single detection step only — this pose is 2.77 cm deep and INTEGRATING from
    it is a blowout (the reference ejects the obj at |qvel| 25.8), which is why
    the settle tests start the obj on the table instead.
    """
    var env = SawyerReach()
    _ = env.reset()

    # Bodies that own a MESH geom, and the obj's body.
    var mesh_bodies = List[Int]()
    var obj_body = -1
    for g in range(env.NGEOM):
        var go = g * MODEL_GEOM_SIZE
        var gt = Int(env.mf.geoms.data[go + GEOM_IDX_TYPE])
        var gb = Int(env.mf.geoms.data[go + GEOM_IDX_BODY])
        if gt == GEOM_MESH:
            mesh_bodies.append(gb)
        elif gt == GEOM_CYLINDER:
            obj_body = gb
    print("  mesh-geom bodies:", len(mesh_bodies), " obj body:", obj_body)
    assert_true(
        len(mesh_bodies) > 0 and obj_body >= 0,
        "no mesh geom or no obj cylinder in the compiled model — this control"
        " cannot mean anything",
    )

    # Teleport the obj into the eGripperBase hull (the fixture pose the
    # single-frame mesh gates use), then take ONE step to run detection.
    var qs = List[Float64]()
    var vs = List[Float64]()
    for i in range(env.NQ):
        qs.append(Float64(env.d.qpos.data[i]))
    for _ in range(env.NV):
        vs.append(0.0)
    qs[OBJ_QPOS + 0] = 0.005
    qs[OBJ_QPOS + 1] = 0.601
    qs[OBJ_QPOS + 2] = 0.28
    qs[OBJ_QPOS + 3] = 1.0
    qs[OBJ_QPOS + 4] = 0.0
    qs[OBJ_QPOS + 5] = 0.0
    qs[OBJ_QPOS + 6] = 0.0
    env.set_state(qs, vs)

    var action = ContAction[ACTION_DIM]()
    for i in range(ACTION_DIM):
        action.data[i] = 0.0
    _ = env.step(action)

    var ncon = Int(env.d.meta.data[META_IDX_NUM_CONTACTS])
    var mesh_rows = 0
    for c in range(ncon):
        var b = c * CONTACT_SIZE
        var ba = Int(env.d.contacts.data[b + CONTACT_IDX_BODY_A])
        var bb = Int(env.d.contacts.data[b + CONTACT_IDX_BODY_B])
        for mb in mesh_bodies:
            if (ba == mb and bb == obj_body) or (bb == mb and ba == obj_body):
                mesh_rows += 1
    print("  contacts:", ncon, " obj-vs-MESH rows:", mesh_rows)
    assert_true(
        mesh_rows > 0,
        "NO mesh contact through the env facade. Mesh geoms are not colliding"
        " — check `CONFIG.NMESH_VERTS` is nonzero for this env and that"
        " `Phyics3dEnv` forwards it instead of a literal 0. Every environment"
        " shipped this way until 2026-08-10.",
    )


def main() raises:
    var suite = TestSuite()
    suite.test[test_sawyer_obj_settles_where_mujoco_does]()
    suite.test[test_sawyer_rest_contact_is_a_manifold]()
    suite.test[test_env_facade_collides_meshes]()
    suite^.run()
