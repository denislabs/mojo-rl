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

from std.math import abs
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
