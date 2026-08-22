"""`<joint type="ball">` — a joint that was free in the model and frozen in
the simulation.

    pixi run mojo run -I . tests/physics3d/test_ball_joint_vs_mujoco.mojo

FOUR SEPARATE DEFECTS, all in ball-joint support, all found from one symptom
(agility_cassie settling 11% too high). Each is independently sufficient to
make the joint wrong, which is why three of them hid behind the others.

  1. `qpos0` LEFT IT AT (0,0,0,0). The pose vector is zero-filled and only the
     FREE joint's `w` was ever set to 1 — from `free_joint_qpos_adr`, which
     records the FIRST free joint, so a second floating body was wrong too.
     A zero quaternion is NOT a rotation: forward kinematics multiplies by it
     and the body AND EVERYTHING BELOW collapses. cassie's achilles rods came
     out with `xquat` exactly (0,0,0,0).
  2. FORWARD KINEMATICS READ THE QUATERNION w-LAST. MuJoCo stores a ball
     joint's qpos as (w, x, y, z) — the same order as a free joint's rotation
     half, which the FREE branch of the same function already read correctly.
  3. NO INTEGRATOR ADVANCED IT. euler / implicit / rk4 each roll their own
     qpos loop and only FREE and HINGE/SLIDE were transcribed, so the three
     ball DOFs accumulated velocity that was never applied to `qpos`.
     `kinematics/integrate_pos.mojo` has held the correct body since it was
     written and has no callers.
  4. ⚠⚠ AND THE ONE UNDERNEATH: `dynamics/cdof.mojo` DID NOT IMPORT
     `JNT_BALL` AT ALL. With no motion subspace the joint's three `cdof` rows
     stayed zero, so its mass-matrix rows and every Jacobian column built from
     them were zero, no force could reach it, and `qvel` stayed EXACTLY 0.0
     forever. `mass_matrix.mojo` and `rne.mojo` both already had their
     `JNT_BALL` cases — they were consuming a `cdof` nobody wrote.

⚠ ONLY ONE MODEL IN THE TREE HAS A BALL JOINT, which is why none of this
surfaced: agility_cassie. `integrate_pos.mojo`'s docstring says so outright —
"THE BALL BRANCH IS UNEXERCISED ... written for faithfulness, not because it
is known to be right. Anything relying on it must gate it first." This is that
gate.

MEASURED, cassie dropped from 1.1 m (MuJoCo 3.10.0, dt 5e-4):

    step 1, max |qpos - MuJoCo|  : 2.2e-16   (was 1.99e-04)
    settles at z                 : 1.0112    (was 1.1276; MuJoCo 1.0135)
    zmax                         : 1.0999991 (MuJoCo 1.0999991)

⚠ THE STEP-1 ROW IS THE ONE THAT MATTERS. Agreement to machine precision on
the first step means FK, cdof, the mass matrix, RNE, the contact set and the
four closed-loop equality rows all agree at the initial state. The settled
height is a 300-step integration of a stiff spring-loaded closed-loop
mechanism and still differs by 0.2%; that residual is a solver-convergence
question, NOT a structural one, and it is deliberately not gated tightly here.
"""

from std.math import abs, acos
from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.studio.stepping import StudioIntegPyr
from mojo_rl.physics3d.joint_types import JNT_BALL

comptime DT = DType.float64
comptime CASSIE = String(
    "references/mujoco_menagerie-main/agility_cassie/scene.xml"
)
# MuJoCo 3.10.0: `m.jnt_qposadr[4]` / `m.jnt_dofadr[4]` for left-achilles-rod.
comptime BALL_QADR = 10
comptime BALL_DADR = 9


struct Built(Movable):
    var fmd_nq: Int
    var key_qpos: List[Float64]
    var m: Model[DT, DynDims]
    var d: Data[DT, DynDims, 1]
    var dims: DynDims

    def __init__(out self) raises:
        var src = read_model_source(CASSIE)
        var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
        var verts = 262144
        var dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
        var m = Model[DT, DynDims](dims)
        var tries = 0
        while True:
            try:
                build_model_runtime[DT](fmd, dims, m)
                break
            except e:
                if String(e).find("mesh vertex capacity") == -1 or tries > 24:
                    raise e
                tries += 1
                verts = verts * 2
                dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
                m = Model[DT, DynDims](dims)
        var sf = spec_fields_runtime[DT](fmd, dims, m)
        var d = Data[DT, DynDims, 1](dims)
        for i in range(dims.get_nq()):
            d.qpos.data[i] = sf.qpos0.data[i]
        for i in range(dims.get_nv()):
            d.qvel.data[i] = Scalar[DT](0)
        # ⚠ THE GATE READS `sf.qpos0`, NOT the model's own reset — that is the
        # slot the bug was in, and it is what every caller starts from.
        assert_true(
            abs(Float64(sf.qpos0.data[BALL_QADR]) - 1.0) < 1e-15
            and Float64(sf.qpos0.data[BALL_QADR + 1]) == 0.0
            and Float64(sf.qpos0.data[BALL_QADR + 2]) == 0.0
            and Float64(sf.qpos0.data[BALL_QADR + 3]) == 0.0,
            "a ball joint's qpos0 must be the IDENTITY quaternion (1,0,0,0),"
            " w FIRST — MuJoCo reports exactly that. Ours is ("
            + String(Float64(sf.qpos0.data[BALL_QADR])) + ", "
            + String(Float64(sf.qpos0.data[BALL_QADR + 1])) + ", "
            + String(Float64(sf.qpos0.data[BALL_QADR + 2])) + ", "
            + String(Float64(sf.qpos0.data[BALL_QADR + 3]))
            + "). All zeros is not a rotation at all.",
        )
        # ⚠ AND THE KEYFRAME, which is the ONLY pose in this file where the
        # ball quaternion is not the identity — see
        # `test_forward_kinematics_matches_mujoco_at_the_keyframe`.
        self.key_qpos = List[Float64]()
        if dims.get_nkey() > 0:
            for i in range(dims.get_nq()):
                self.key_qpos.append(Float64(sf.key_qpos.data[i]))
        self.fmd_nq = dims.get_nq()
        self.dims = dims
        self.m = m^
        self.d = d^


def test_ball_joint_is_present_and_starts_at_identity() raises:
    """The fixture is what it claims to be, and qpos0 is MuJoCo's."""
    print("=== cassie has ball joints, and qpos0 is the identity ===")
    var src = read_model_source(CASSIE)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var nball = 0
    for j in range(len(fmd.joints)):
        if fmd.joints[j].jnt_type == JNT_BALL:
            nball += 1
    print("  ball joints:", nball, " (MuJoCo: 2)")
    assert_true(
        nball == 2,
        "cassie has two ball joints (the achilles rods); parsed "
        + String(nball) + " — the rest of this file would be vacuous",
    )
    var b = Built()  # its constructor asserts the qpos0 identity
    _ = b^
    print("  PASS")


def test_forward_kinematics_matches_mujoco_at_qpos0() raises:
    """The pose the whole model is built from.

    ⚠ COMPARED UP TO SIGN, because q and -q are the same rotation and nothing
    downstream can tell them apart. Ours happens to match MuJoCo's sign too,
    but asserting that would be gating a convention rather than the physics.
    """
    print("=== FK at qpos0: the achilles rod's world orientation ===")
    var b = Built()
    forward_kinematics["cpu", DT, DynDims, 1](b.d, b.m)
    # MuJoCo `d.xquat[5]` = (w, x, y, z); ours is stored (x, y, z, w).
    var mw = -0.299291030002
    var mx = -0.327765565886
    var my = -0.661729639866
    var mz = 0.604242084683
    var ox = Float64(b.d.xquat.data[5 * 4 + 0])
    var oy = Float64(b.d.xquat.data[5 * 4 + 1])
    var oz = Float64(b.d.xquat.data[5 * 4 + 2])
    var ow = Float64(b.d.xquat.data[5 * 4 + 3])
    print("  ours   (x,y,z,w)", ox, oy, oz, ow)
    print("  MuJoCo (x,y,z,w)", mx, my, mz, mw)
    var dot = abs(ox * mx + oy * my + oz * mz + ow * mw)
    print("  |dot| =", dot, " (1.0 means the same rotation)")
    assert_true(
        abs(dot - 1.0) < 1e-9,
        "body 5's world quaternion disagrees with MuJoCo (|dot| " + String(dot)
        + "). Before the fix this was EXACTLY (0,0,0,0): a zero quaternion,"
        " which FK propagates to every body below it.",
    )
    _ = b^
    print("  PASS")


def test_ball_joint_moves_and_the_first_step_matches_mujoco() raises:
    """The defect stated as a property: the joint must actually rotate.

    ⚠ THE `qvel == 0` ASSERTION IS THE HEART OF THIS FILE. With no `cdof` the
    three ball DOFs were not merely inaccurate, they were EXACTLY zero on
    every step — a joint the model declares free and the simulation never
    moves. A tolerance-based check on the trajectory would have called that
    "close"; this cannot.
    """
    print("=== the ball joint rotates, and step 1 matches MuJoCo ===")
    var b = Built()
    var integ = StudioIntegPyr(b.dims)
    integ.step["cpu"](b.d, b.m)

    var vx = Float64(b.d.qvel.data[BALL_DADR + 0])
    var vy = Float64(b.d.qvel.data[BALL_DADR + 1])
    var vz = Float64(b.d.qvel.data[BALL_DADR + 2])
    print("  after 1 step, ball qvel", vx, vy, vz)
    print("    MuJoCo              0.004889322 0.004982316 -0.184994536")
    assert_true(
        abs(vx) + abs(vy) + abs(vz) > 1e-9,
        "the ball joint's three DOFs are still at zero velocity. That is the"
        " `cdof` defect: with no motion subspace no force can reach the joint,"
        " so it never moves no matter how long you integrate.",
    )
    assert_true(
        abs(vx - 0.004889322) < 1e-6
        and abs(vy - 0.004982316) < 1e-6
        and abs(vz - (-0.184994536)) < 1e-6,
        "the ball joint's velocity after one step disagrees with MuJoCo",
    )
    # The quaternion the integrator wrote back, w first.
    var qw = Float64(b.d.qpos.data[BALL_QADR + 0])
    var qz = Float64(b.d.qpos.data[BALL_QADR + 3])
    print("  ball qpos w,z", qw, qz, " (MuJoCo 0.999999998929, -4.624863e-05)")
    assert_true(
        abs(qz - (-4.624863403496e-05)) < 1e-11,
        "the ball quaternion did not advance to MuJoCo's value; got z = "
        + String(qz)
        + ". Exactly 0 means no integrator has a JNT_BALL branch.",
    )
    assert_true(
        abs(qw - 0.9999999989290) < 1e-12,
        "ball quaternion w is " + String(qw) + ", MuJoCo 0.9999999989290",
    )
    # ⚠ AND THE WHOLE-BODY CONSEQUENCE, so this cannot pass on the joint alone.
    var z = Float64(b.d.qpos.data[2])
    print("  root z after 1 step", z, " (MuJoCo 1.099999093606)")
    assert_true(
        abs(z - 1.099999093606) < 1e-11,
        "the root height after one step is " + String(z)
        + " against MuJoCo's 1.099999093606",
    )
    _ = b^
    print("  PASS")


def test_forward_kinematics_matches_mujoco_at_the_keyframe() raises:
    """⚠⚠ EVERY OTHER ROW IN THIS FILE IS AT OR NEXT TO THE IDENTITY.

    A ball joint's `qpos0` is (1,0,0,0), and the identity COMMUTES — so
    `q_parent * q_ball` and `q_ball * q_parent` are the same quaternion and no
    gate started from `qpos0` can tell them apart. One step from `qpos0` moves
    the ball by 4.6e-05 rad, which is not enough either: the row above
    measured 2.2e-16 while `_fk_body` was composing the ball quat in the WRONG
    ORDER, as a WORLD rotation where `mj_kinematics` composes it as a LOCAL
    one (`mju_mulQuat(xquat, xquat, qloc)`, engine_core_smooth.c:141).

    cassie's keyframe puts the left rod at (0.97861, -0.01641, 0.01778,
    -0.20430) — **23.6 deg** off the identity — and that is the only pose in
    the tree where the two orders differ.

    ⚠ AND IT IS INVISIBLE IN `xpos`. Both ball joints have `jnt_pos == 0`, so
    the child's origin is the anchor and no rotation reaches the position:
    `xpos` was exact to 1.7e-16 throughout. `xipos` (= `xpos + R ipos`, with
    `ipos = (0.247, 0, 0)`) is where it shows, and it is also the field the
    rest of the pipeline reads — `subtree_com`, `cdof`, RNE. Measured:

        |d qfrc_bias| on cassie   8.999e-04 -> 1.208e-17
        board, one step           4.460e-04 -> 1.522e-04
    """
    print("=== FK at the KEYFRAME, where the ball quat is NOT the identity ===")
    var b = Built()
    assert_true(
        len(b.key_qpos) == b.fmd_nq,
        "cassie's keyframe did not load; this row would be a second copy of"
        " the qpos0 one",
    )
    for i in range(b.fmd_nq):
        b.d.qpos.data[i] = Scalar[DT](b.key_qpos[i])

    # ⚠ NON-VACUITY: the pose must actually rotate the ball joint.
    var kw = b.key_qpos[BALL_QADR]
    var kwc = kw if kw <= 1.0 else 1.0
    var ang = 2.0 * acos(abs(kwc)) * 57.29577951308232
    print("  keyframe ball quat w =", kw, " (~", ang, "deg off identity)")
    assert_true(
        abs(kw - 1.0) > 1e-3,
        "the keyframe leaves the ball joint at the identity (w = " + String(kw)
        + "), so this row cannot see a composition-ORDER defect at all",
    )

    forward_kinematics["cpu", DT, DynDims, 1](b.d, b.m)

    # MuJoCo 3.10.0 at keyframe 0. `xipos` is three plain numbers and carries
    # no layout convention, unlike `xquat`.
    var mjxi = List[Float64]()
    mjxi.append(-0.18201613); mjxi.append(0.10913965); mjxi.append(0.70822279)
    mjxi.append(-0.18201611); mjxi.append(-0.10913967); mjxi.append(0.70822278)
    var bodies = List[Int]()
    bodies.append(5); bodies.append(17)
    for k in range(2):
        var bi = bodies[k]
        var rx = mjxi[k * 3 + 0]
        var ry = mjxi[k * 3 + 1]
        var rz = mjxi[k * 3 + 2]
        var ox = Float64(b.d.xipos.data[bi * 3 + 0])
        var oy = Float64(b.d.xipos.data[bi * 3 + 1])
        var oz = Float64(b.d.xipos.data[bi * 3 + 2])
        var e = max(abs(ox - rx), max(abs(oy - ry), abs(oz - rz)))
        print("  body", bi, " xipos ours", ox, oy, oz)
        print("            MuJoCo", rx, ry, rz, "  |d| =", e)
        assert_true(
            e < 1e-7,
            String("achilles rod body ") + String(bi) + " xipos is "
            + String(e) + " from MuJoCo's. With the ball quat composed as a"
            " WORLD rotation this was 9.8e-02 — a 22 deg error in the rod's"
            " orientation, with `xpos` still exact.",
        )
    _ = b^
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
