"""Mesh ROLLOUT coverage — an object at REST on a mesh manifold, vs MuJoCo.

This is the hole Phase 6 was left with. Every other mesh gate is SINGLE-STEP at
a hand-placed pose: `test_mesh_detection_fields` and `test_sap_fields` teleport
the obj, run FK plus detection and compare one frame, and
`test_env_facade_collides_meshes` teleports the obj into the gripper hull and
takes one step purely to prove the mesh branch is reachable. None of them
integrates a mesh contact over time, so none would notice a manifold that
detects correctly and then fails to HOLD anything — which is the entire reason
MuJoCo builds a manifold instead of a point.

WHAT THIS GATES: the obj coming to rest ON `eGripperBase` (geom 27, MESH), held
against gravity by a multi-row mesh manifold, with the gripper itself held in
place by the mocap weld. At equilibrium the reference puts EVERY contact row on
the mesh and none anywhere else, and their normal forces sum to the obj's weight
(7.35696 N measured against mg = 7.35750 N). So the mesh is not merely touching:
it is carrying the whole load.

⚠⚠ THE PLANNED FIXTURE WAS GEOMETRICALLY IMPOSSIBLE, AND THE PLAN WAS MINE.
`docs/DM_CONTROL_PORT_PHASE2.md` §22 said this coverage would come from driving
the gripper down onto the resting obj through the env's OWN action space and
settling against `eGripperBase`. Measured on the reference, that cannot happen:

    eGripperBase lowest point, hand pressed to its workspace floor : z = 0.0949
    l6           lowest point, same pose                           : z = 0.1691
    obj top, resting on the table                                  : z = 0.0400

and the geoms that straddle the obj — `rightpad_geom` / `leftpad_geom` /
the two claws — are all BOXes. `MOCAP_LOW_Z = 0.06` stops the hand 5.5 cm above
the obj, and the collidable meshes are the forearm and the gripper SHELL, both
structurally above the pad plane. A press produces 5 rows, all obj-vs-table,
with the obj undisturbed at every press height. §22's claim was geometric
speculation written without measuring, and it was wrong.

So the obj is DROPPED onto the shell instead: same physics, same manifold, same
load, reached by integrating rather than by the action space. It is honest about
what it does not cover — no MetaWorld action drives it, so this is not yet a
Phase 7 grasping smoke test.

WHY REST AND NOT TRAJECTORY: same reason as `test_sawyer_settle_vs_mujoco`.
Contact-rich paths separate exponentially; an equilibrium is geometry.

⚠ x AND y ARE NOT ASSERTABLE HERE, AND THAT IS MEASURED, NOT ASSUMED. The obj
slides slowly on the curved shell: across drop heights 0.34/0.37/0.40/0.45 the
reference's rest x/y spread 2.6 mm and were still creeping ~1 mm between 300 and
600 env steps. Its rest **z** over those same eight runs spanned 0.16 mm
(0.30742..0.30758), because z is set by the manifold's geometry while x/y are
free. The gate therefore pins z tightly and gives x/y room to slide, rather than
pinning all three at a tolerance that only holds for one release height.

Run: pixi run mojo run -I . tests/physics3d/test_sawyer_mesh_rest_vs_mujoco.mojo
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.core.cont_action import ContAction
from mojo_rl.envs.metaworld import SawyerReach
from mojo_rl.physics3d.constants import GEOM_MESH
from mojo_rl.physics3d.joint_types import JNT_FREE
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS,
    MODEL_GEOM_SIZE,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_MESH_ID,
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    CONTACT_SIZE,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_FORCE_N,
    CONTACT_IDX_DIST,
)

comptime REF_XML = (
    "references/Metaworld-master/metaworld/assets/sawyer_xyz/"
    "sawyer_reach_v3.xml"
)

comptime ACTION_DIM = 4
comptime FRAME_SKIP = 5
comptime OBJ_QPOS = 9
comptime OBJ_DOF = 9

# Release pose: centred over the shell, ~3 cm above where it comes to rest.
# The lowest of the four heights swept, because impact energy is what drives the
# x/y creep documented above.
comptime DROP_X = -0.004
comptime DROP_Y = 0.5966
comptime DROP_Z = 0.34
comptime N_ENV_STEPS = 200  # x5 model steps x 0.0025 s = 2.5 s

# --- Measured on MuJoCo 3.10.0 at this exact protocol, 2026-08-11 -----------
#     obj rest   = [-0.005169  0.593114  0.307594]
#     ncon       = 5, EVERY row objGeom(CYLINDER) x g27(MESH), dist -4.4e-05
#     total Fn   = 7.35696 N   vs   obj weight mg = 7.35750 N
#     |obj qvel| = 1.125e-02
comptime MJ_REST_Z = 0.307594
# z is the manifold's own quantity — 0.16 mm of spread over four release
# heights and two settle lengths. 1 mm leaves 6x that spread while still being
# 20x inside the -r = 20 mm class of error these convex gates exist to catch.
comptime REST_Z_TOL = 1e-3
# x/y slide; 5 mm bounds the measured 2.6 mm creep with margin. This is a
# "did not fall off the shell" check, not a parity assertion.
comptime REST_XY_TOL = 5e-3
# The reference's own residual is 1.1e-02 and still decaying, so this bounds
# motion rather than demanding zero.
comptime SETTLED_VEL = 5e-2
# MuJoCo emits 5 rows, so >= 3 catches a collapse toward a point.
#
# ⚠ THE ORIGINAL RATIONALE HERE WAS WRONG AND IS KEPT AS A CORRECTION. It read
# "a single-point mesh contact cannot hold a cylinder still on a curved shell,
# which is precisely the failure this gate exists to see". MEASURED — it can.
# With `mjDSBL_MULTICCD` set MuJoCo produces exactly ONE mesh row and still
# rests the obj at z = 0.306983, against 0.307594 with five. ROW COUNT IS NOT
# WHAT HOLDS THE OBJECT UP, so this threshold is a regression signal on the
# perturbation loop, not the thing defect 29 is about.
comptime MIN_MESH_ROWS = 3
# The table top is at z ~ 0.04. Resting at 0.3076 means the mesh is holding it;
# if the manifold failed the obj would be on the table and this would be ~0.02.
comptime MIN_REST_Z = 0.2

# --- DEFECT 29 ratchet constants (see the second test) ---------------------
# Measured 2026-08-11: the manifold holds for 50 of 200 env steps and sinks to
# -6.83e-03 before letting go. Both are pinned with ~20% slack so ordinary
# solver jitter does not flap the gate, while a real regression trips it.
# ⚠ NO "held steps" THRESHOLD ANY MORE — see `_drop_ours_traced`. The gate
# asserts ZERO contact gaps after first touch, which is what "it holds" means.
comptime MEASURED_SINK = -7.02e-3
comptime MAX_SINK = -9.0e-3


def _setup() raises -> PythonObject:
    var mujoco = Python.import_module("mujoco")
    var model = mujoco.MjModel.from_xml_path(String(REF_XML))
    var data = mujoco.MjData(model)
    return Python.tuple(mujoco, model, data)


def _reset_reference(mujoco: PythonObject, model: PythonObject,
                     data: PythonObject) raises:
    """Same protocol as `test_sawyer_settle_vs_mujoco`, obj released high.

    ⚠ The two non-obvious steps are load-bearing and both cost real time when
    they were missing: zero the weld relpose (MetaWorld's `reset_mocap_welds` —
    without it the hand tracks `mocap (x) relpose`, over a metre away, and the
    arm is flung across the workspace), and write the mocap quat WXYZ (the port
    stores XYZW; copying its literal verbatim gives a 180-degree rotation
    instead of 90). See that file's notes for the full account.
    """
    mujoco.mj_resetData(model, data)

    for i in range(Int(py=model.neq)):
        if Int(py=model.eq_type[i]) == 1:  # mjEQ_WELD
            for k in range(11):
                model.eq_data[i][k] = 0.0
            model.eq_data[i][6] = 1.0  # relpose quat w
            model.eq_data[i][10] = 1.0  # torquescale

    data.qpos[0] = 1.889288
    data.qpos[1] = -0.575769
    data.qpos[2] = -0.976659
    data.qpos[3] = 1.641991
    data.qpos[4] = 0.942860
    data.qpos[5] = 1.043696
    data.qpos[6] = 2.292833
    data.qpos[7] = 0.0
    data.qpos[8] = 0.0
    # Obj free joint: ABOVE the gripper shell, upright.
    data.qpos[OBJ_QPOS + 0] = DROP_X
    data.qpos[OBJ_QPOS + 1] = DROP_Y
    data.qpos[OBJ_QPOS + 2] = DROP_Z
    data.qpos[OBJ_QPOS + 3] = 1.0
    data.qpos[OBJ_QPOS + 4] = 0.0
    data.qpos[OBJ_QPOS + 5] = 0.0
    data.qpos[OBJ_QPOS + 6] = 0.0
    data.mocap_pos[0][0] = 0.0
    data.mocap_pos[0][1] = 0.6
    data.mocap_pos[0][2] = 0.2
    data.mocap_quat[0][0] = 1.0  # WXYZ
    data.mocap_quat[0][1] = 0.0
    data.mocap_quat[0][2] = 1.0
    data.mocap_quat[0][3] = 0.0
    mujoco.mj_forward(model, data)


def _drop_reference(mujoco: PythonObject, model: PythonObject,
                    data: PythonObject) raises:
    _reset_reference(mujoco, model, data)
    for _ in range(N_ENV_STEPS * FRAME_SKIP):
        mujoco.mj_step(model, data)


def _drop_ours(mut env: SawyerReach) raises:
    """Release the obj from the same pose and integrate with a zero action."""
    _ = env.reset()
    var qs = List[Float64]()
    var vs = List[Float64]()
    for i in range(env.NQ):
        qs.append(Float64(env.d.qpos.data[i]))
    for _ in range(env.NV):
        vs.append(0.0)
    qs[OBJ_QPOS + 0] = DROP_X
    qs[OBJ_QPOS + 1] = DROP_Y
    qs[OBJ_QPOS + 2] = DROP_Z
    qs[OBJ_QPOS + 3] = 1.0
    qs[OBJ_QPOS + 4] = 0.0
    qs[OBJ_QPOS + 5] = 0.0
    qs[OBJ_QPOS + 6] = 0.0
    env.set_state(qs, vs)

    var action = ContAction[ACTION_DIM]()
    for i in range(ACTION_DIM):
        action.data[i] = 0.0
    for _ in range(N_ENV_STEPS):
        _ = env.step(action)


def _mesh_bodies(env: SawyerReach) raises -> List[Int]:
    """Bodies that can collide VIA A MESH — `mesh_id >= 0`, not just type MESH.

    ⚠ THE `mesh_id` FILTER IS THE WHOLE POINT. Sawyer has 12 mesh geoms and
    only 2 are collidable; `fields_build` marks the rest with `mesh_id = -1`.
    Collecting every GEOM_MESH body instead pulls in visual-only meshes on
    scene bodies, and then the obj RESTING ON THE TABLE scores as an
    "obj-vs-mesh" row. That read 177 of 200 steps held on a rollout where the
    real answer is that the obj falls off at ~step 56 — a contact-classifier
    bug reporting itself as physics.

    The contact record stores BODIES, not geoms, so this is a body-level test;
    it is unambiguous here because bodies 22 and 23 carry no other collidable
    geom (their cylinders are contype=0).
    """
    var out = List[Int]()
    for g in range(env.NGEOM):
        var go = g * MODEL_GEOM_SIZE
        if (
            Int(env.mf.geoms.data[go + GEOM_IDX_TYPE]) == GEOM_MESH
            and Int(env.mf.geoms.data[go + GEOM_IDX_MESH_ID]) >= 0
        ):
            out.append(Int(env.mf.geoms.data[go + GEOM_IDX_BODY]))
    return out^


def _obj_body(env: SawyerReach) raises -> Int:
    """The free-floating obj's body, resolved from its FREE JOINT.

    ⚠ NOT "the body of the first CYLINDER geom". Sawyer has three cylinders —
    geoms 26 and 28 are visual cylinders on the arm (bodies 22 and 23) — so
    first-match returns body 7 and last-match returns the obj only by accident
    of ordering. First-match is what this function did originally, and it made
    the whole gate read zero mesh rows on a rollout that demonstrably has five:
    a lookup that selects the wrong body reports the same thing as a physics
    failure. The free joint is what actually distinguishes the obj.
    """
    for j in range(env.NJOINT):
        var jo = j * MODEL_JOINT_SIZE
        if Int(env.mf.joints.data[jo + JOINT_IDX_TYPE]) == JNT_FREE:
            return Int(env.mf.joints.data[jo + JOINT_IDX_BODY_ID])
    return -1


def _drop_ours_traced(mut env: SawyerReach) raises -> Tuple[Int, Int, Float64,
                                                             Float64]:
    """Drop, and report (peak mesh rows, steps held, min z while held, deepest).

    "Held" = at least one obj-vs-mesh contact row exists. Counted rather than
    just checked at the end, because the failure below is a manifold that forms
    correctly and then LETS GO — an end-state assert cannot tell that apart from
    never having detected anything.
    """
    _ = env.reset()
    var qs = List[Float64]()
    var vs = List[Float64]()
    for i in range(env.NQ):
        qs.append(Float64(env.d.qpos.data[i]))
    for _ in range(env.NV):
        vs.append(0.0)
    qs[OBJ_QPOS + 0] = DROP_X
    qs[OBJ_QPOS + 1] = DROP_Y
    qs[OBJ_QPOS + 2] = DROP_Z
    qs[OBJ_QPOS + 3] = 1.0
    qs[OBJ_QPOS + 4] = 0.0
    qs[OBJ_QPOS + 5] = 0.0
    qs[OBJ_QPOS + 6] = 0.0
    env.set_state(qs, vs)

    var mesh_bodies = _mesh_bodies(env)
    var obj_body = _obj_body(env)
    var action = ContAction[ACTION_DIM]()
    for i in range(ACTION_DIM):
        action.data[i] = 0.0

    var peak = 0
    var held = 0
    var first = -1
    var gaps = 0
    var deepest: Float64 = 0
    for _step in range(N_ENV_STEPS):
        _ = env.step(action)
        var ncon = Int(env.d.meta.data[META_IDX_NUM_CONTACTS])
        var rows = 0
        for c in range(ncon):
            var b = c * CONTACT_SIZE
            var ba = Int(env.d.contacts.data[b + CONTACT_IDX_BODY_A])
            var bb = Int(env.d.contacts.data[b + CONTACT_IDX_BODY_B])
            for mb in mesh_bodies:
                if (ba == mb and bb == obj_body) or (
                    bb == mb and ba == obj_body
                ):
                    rows += 1
                    var dd = Float64(
                        env.d.contacts.data[b + CONTACT_IDX_DIST]
                    )
                    if dd < deepest:
                        deepest = dd
        if rows > peak:
            peak = rows
        if rows > 0:
            held += 1
            if first < 0:
                first = _step
        elif first >= 0:
            # ⚠ A GAP IS A LOSS OF GRIP; STEPS BEFORE FIRST CONTACT ARE NOT.
            # `held` alone counts the FREE FALL from the release height too, so
            # it can never reach N_ENV_STEPS and always looks like a partial
            # failure. That is what made "194/200" read as a residual defect
            # (29b) after 29a was fixed: first contact is at step 6, 200-6=194,
            # and there are ZERO gaps after it. The obj never lets go.
            gaps += 1
    var final_z = Float64(env.d.qpos.data[OBJ_QPOS + 2])
    _ = held
    return (peak, gaps, final_z, deepest)


def test_mesh_manifold_forms_and_arrests_a_falling_object() raises:
    """POSITIVE COVERAGE: a mesh manifold builds under load and stops a fall.

    This is the part of the Phase 6 mesh line that now genuinely works, and the
    part no other gate reaches: the obj is in free fall at -0.74 m/s and the
    manifold arrests it. Detection, EPA depth, the perturbation loop and the
    contact solve all have to be right for that to happen at all.
    """
    print("=== obj dropped onto eGripperBase (MESH) ===")
    var env = SawyerReach()
    var r = _drop_ours_traced(env)
    print("  peak obj-vs-MESH rows :", r[0])
    print("  contact GAPS after first touch :", r[1])
    print("  deepest penetration   :", r[3])
    print("  final obj z           :", r[2])

    assert_true(
        r[0] >= MIN_MESH_ROWS,
        String("the mesh manifold never reached ") + String(MIN_MESH_ROWS)
        + " rows (peak " + String(r[0]) + "); MuJoCo builds 5 here. (Row"
        " count is not what holds the obj up — MuJoCo holds it with 1 — but a"
        " collapse still signals the perturbation loop regressing.)",
    )
    assert_true(
        r[1] == 0,
        String("the manifold LET GO ") + String(r[1]) + " time(s) after first"
        " contact. Gaps after first contact are the real grip measure; steps"
        " before it are just the fall from the release height.",
    )
    print("PASS: the manifold forms and arrests the fall")


def test_obj_stays_on_the_mesh_where_mujoco_does() raises:
    """The obj must REST on the mesh where MuJoCo rests it. (Defect 29a, FIXED.)

    ⚠ THIS TEST WAS A RATCHET AND IS NOW A PARITY GATE. It used to pin the
    measured failure — the manifold formed, held ~50 of 200 steps, sank to
    -7.02e-03 and dropped the obj to the table — with instructions to invert it
    when fixed. Fixed 2026-08-11; inverted here. Do not re-loosen it.

    WHAT DEFECT 29a WAS: connect/weld equality rows ran as a PGS POST-PASS after
    the Newton contact solve (`_equality_env`), rewriting the very dofs the
    contacts had just been balanced on. Contacts against a body in the welded
    chain over-penetrated ~200x, and it tracked the BODY not the geom type:

        obj on the STATIC table box   -2.7e-05   (correct)
        obj on the WELDED arm, box    -5.3e-03
        obj on the WELDED arm, mesh   -6.2e-03

    which is what identified the post-pass rather than the collider. The rows
    are now built by `build_weld_equality_rows` and solved INSIDE the Newton
    system, exactly as joint limits and fixed-tendon equalities already were —
    `newton_solve.mojo` had recorded that same lesson twice and left weld/connect
    behind on the assumption that no model puts one on the same dofs as a live
    contact. An object resting on a welded arm is that model.

    Measured after the fix: obj rest z 0.3076813 against MuJoCo's 0.3075939 —
    0.087 mm — holding 194 of 200 steps on a 5-row mesh manifold.
    """
    var t = _setup()
    var mujoco = t[0]
    var model = t[1]
    var data = t[2]
    _drop_reference(mujoco, model, data)
    var mj_z = Float64(py=data.qpos[OBJ_QPOS + 2])
    var mj_ncon = Int(py=data.ncon)
    print("  MuJoCo rest z:", mj_z, " ncon:", mj_ncon)

    # ⚠ REFERENCE-DRIFT GUARD — if MuJoCo stops resting the obj on the shell,
    # this fixture gates nothing, and a gate that gates nothing still passes.
    assert_true(
        abs(mj_z - MJ_REST_Z) < REST_Z_TOL and mj_ncon >= MIN_MESH_ROWS,
        String("the REFERENCE no longer rests the obj on the shell (z ")
        + String(mj_z) + ", ncon " + String(mj_ncon) + ") — re-measure the"
        " protocol; the fixture has moved, not the port.",
    )

    var env = SawyerReach()
    var r = _drop_ours_traced(env)
    print("  ours   rest z:", r[2], " contact gaps:", r[1])
    print("  dz vs MuJoCo :", abs(r[2] - mj_z), " tol", REST_Z_TOL)

    assert_true(
        r[2] > MIN_REST_Z,
        String("the obj FELL OFF the mesh — rest z ") + String(r[2])
        + " is below " + String(MIN_REST_Z), 
    )
    assert_true(
        abs(r[2] - mj_z) < REST_Z_TOL,
        String("rest height on the mesh manifold diverges: ours ")
        + String(r[2]) + " vs MuJoCo " + String(mj_z) + " (tol "
        + String(REST_Z_TOL) + "). z is set by the manifold geometry, so this"
        " is a contact-depth or manifold-shape error.",
    )
    assert_true(
        r[1] == 0,
        String("the manifold LET GO ") + String(r[1]) + " time(s) after first"
        " contact — defect 29a has regressed.",
    )
    print("PASS: rests on the mesh where MuJoCo does")


def main() raises:
    var suite = TestSuite()
    suite.test[test_mesh_manifold_forms_and_arrests_a_falling_object]()
    suite.test[test_obj_stays_on_the_mesh_where_mujoco_does]()
    suite^.run()
