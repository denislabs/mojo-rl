"""A free joint's `qpos0` is its BODY's pose, and a keyframe quaternion is
REPAIRED before it is used.

    pixi run mojo run -I . tests/physics3d/test_free_joint_and_keyframe_quat_vs_mujoco.mojo

TWO DEFECTS, both of which put a robot in the wrong RESET POSE — so every
trajectory from that model started somewhere MuJoCo never was.

── 1. `qpos0` TOOK THE BODY'S POSITION AND THEN HARDCODED `w = 1` ──────────

A free joint's `qpos0` is its body's pose in the parent frame: MuJoCo takes
BOTH `body_pos` and `body_quat`. We took the position and wrote the identity
quaternion. Measured on anybotics_anymal_b, whose base is

    <body name="base" pos="0 0 0.58" quat="0 0 0 1" ...>

so MuJoCo's `qpos0[:7]` is `[0, 0, 0.58, 0, 0, 0, 1]` — a 180-degree yaw —
against our `[0, 0, 0.58, 1, 0, 0, 0]`. anymal ships NO keyframe, so that is
the pose everything starts from, and both anymal scenes diverged by exactly
1.000e+00 at step ONE.

⚠⚠ AND FIXING IT REVEALED A SECOND WRITER. A `qpos0[qw] = 1.0` stamp sat
further down whose own comment read "NOW REDUNDANT with the loop above ...
re-writing an identical value is cheaper than proving the two can never
disagree." It stopped being identical the moment the loop started writing the
body's quaternion: the loop wrote (0,0,0,1), the stamp put `w = 1` back over
it, and the result was (1,0,0,1) — norm sqrt(2). The residual after the first
half of the fix was exactly 1/sqrt(2) = 0.7071, which is what a
half-corrected quaternion looks like. A REDUNDANT WRITE IS A SECOND WRITER; it
is only redundant while the other one agrees, and nothing was checking.

⚠ THE RECORD IS (x, y, z, w) AND `qpos` IS (w, x, y, z).

── 2. A DEGENERATE KEYFRAME QUATERNION WAS TAKEN VERBATIM ──────────────────

`mjCModel::Compile` runs `mj_normalizeQuat` over every key
(`user_model.cc:5353`), and `mju_normalize4` sets a vector whose norm is below
`mjMINVAL` to the IDENTITY, normalizing otherwise. We used the file's numbers
as written.

⚠ IT IS NOT A HYPOTHETICAL. `pal_tiago/tiago_position.xml` writes
`qpos="0 0 -0.985 0 0 0 0 ..."` — a ZERO quaternion on its free joint. MuJoCo
compiles that to `key_qpos[3] = 1`; we reset with a zero quat and forward
kinematics multiplied by it. `scene_position` and `scene_velocity` both
diverged by exactly 1.000e+00 at step ONE while the same robot's
`scene_motor` — which ships no keyframe — sat at 6.7e-16. That contrast is
what named it.

⚠⚠ READING ONLY THE FIRST NORMALIZER GIVES THE WRONG RULE. `mjuu_normvec`
(`user_model.cc:4057`) runs EARLIER over the same array and RETURNS ON A ZERO
VECTOR WITHOUT TOUCHING IT. It is the later `mju_normalize4` that does the
repair. Two functions, same array, opposite behaviour on the degenerate case.

MEASURED, worst |d(qpos)| against MuJoCo 3.10.0 at step ONE, random ctrl:

    anybotics_anymal_b     1.000e+00 -> 6.072e-18
    anybotics_anymal_c     1.000e+00 -> 3.786e-12
    pal_tiago/position     1.000e+00 -> 3.990e-17
    pal_tiago/velocity     1.000e+00 -> 9.689e-18
    pal_talos/position     1.000e+00 -> 1.843e-17

Across all 85 loadable Menagerie scenes at step one, this took the count at or
below 1e-9 from 50 to 55 and the count above 1e-3 from 16 to 11, with nothing
else moving.
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from mojo_rl.physics3d.fields import Model, DynDims

comptime DT = DType.float64

comptime ANYMAL = String(
    "references/mujoco_menagerie-main/anybotics_anymal_b/scene.xml"
)
comptime TIAGO = String(
    "references/mujoco_menagerie-main/pal_tiago/scene_position.xml"
)

# Three free bodies, each exercising a different branch, and one keyframe
# exercising the other three.
#
#   body a  quat="0 0 0 1"     a NON-identity body quat        -> qpos0
#   body b  no quat at all     the identity, still             -> qpos0 control
#   body c  a 90-degree quat   an ordinary unit quat           -> qpos0
#   key  a  "0 0 0 0"          DEGENERATE -> identity          -> keyframe
#   key  b  "0.6 0 0 0.6"      UNNORMALISED -> normalised      -> keyframe
#   key  c  "1 0 0 0"          already unit -> UNTOUCHED       -> keyframe control
comptime XML = String(
    """<mujoco><compiler angle="radian"/>
<worldbody>
  <body name="a" pos="0 0 1" quat="0 0 0 1"><freejoint/>
    <geom type="box" size=".1 .1 .1" mass="1"/></body>
  <body name="b" pos="1 0 1"><freejoint/>
    <geom type="box" size=".1 .1 .1" mass="1"/></body>
  <body name="c" pos="2 0 1" quat="0.7071067811865476 0 0 0.7071067811865476">
    <freejoint/>
    <geom type="box" size=".1 .1 .1" mass="1"/></body>
</worldbody>
<keyframe>
  <key name="k" qpos="0 0 1  0 0 0 0   1 0 1  0.6 0 0 0.6   2 0 1  1 0 0 0"/>
</keyframe></mujoco>"""
)

comptime R2 = 0.7071067811865476
comptime R2K = 0.7071067811865475


def _load(
    xml: String, base: String
) raises -> Tuple[List[Float64], List[Float64]]:
    """(`qpos0`, keyframe-0 `qpos`) through the runtime path."""
    var fmd = parse_xml_full(expand_mjcf(xml, base), base)
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=32, nmesh_verts=verts)
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
            dims = dims_from_flat(fmd, max_contacts=32, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var nq = dims.get_nq()
    var q0 = List[Float64]()
    var k0 = List[Float64]()
    for i in range(nq):
        q0.append(Float64(sf.qpos0.data[i]))
        k0.append(
            Float64(sf.key_qpos.data[i]) if dims.get_nkey() > 0 else 0.0
        )
    return (q0^, k0^)


def test_free_joint_qpos0_is_the_body_pose() raises:
    """`qpos0` = body pos AND body quat, with the identity only when earned.

    ⚠ EXPECTED VALUES ARE MUJOCO'S `m.qpos0` for this exact fixture.
    """
    print("=== free-joint qpos0 ===")
    var r = _load(XML, String(""))
    var q0 = r[0].copy()
    assert_true(
        len(q0) == 21,
        "three free joints is 21 qpos slots — got " + String(len(q0)),
    )
    var want: List[Float64] = [
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        2.0, 0.0, 1.0, R2, 0.0, 0.0, R2,
    ]
    var worst = 0.0
    for i in range(21):
        var e = abs(q0[i] - want[i])
        if e > worst:
            worst = e
    print("  a (quat 0 0 0 1):", q0[3], q0[4], q0[5], q0[6],
          "  mj 0 0 0 1")
    print("  b (no quat)     :", q0[10], q0[11], q0[12], q0[13],
          "  mj 1 0 0 0")
    print("  c (90 deg)      :", q0[17], q0[18], q0[19], q0[20],
          "  mj", R2, "0 0", R2)
    print("  worst |d| =", worst)
    assert_true(
        abs(q0[3] - 0.0) < 1e-15 and abs(q0[6] - 1.0) < 1e-15,
        "body a declares `quat=\"0 0 0 1\"`, so its free joint's qpos0"
        " quaternion is (0, 0, 0, 1) — a 180-degree yaw. Got ("
        + String(q0[3]) + ", " + String(q0[4]) + ", " + String(q0[5]) + ", "
        + String(q0[6]) + "). A (1, 0, 0, 0) here means the body quat was"
        " ignored; a (1, 0, 0, 1) means it was written and then stamped over.",
    )
    # ⚠ THE NEGATIVE CONTROL. A body with NO quat must still get the identity
    # — without this row, "always copy the body quat" would pass while a
    # zero-filled record silently produced a zero quaternion.
    assert_true(
        abs(q0[10] - 1.0) < 1e-15 and abs(q0[11]) < 1e-15
        and abs(q0[12]) < 1e-15 and abs(q0[13]) < 1e-15,
        "body b declares no quat, so its qpos0 quaternion is the IDENTITY;"
        " got (" + String(q0[10]) + ", " + String(q0[11]) + ", "
        + String(q0[12]) + ", " + String(q0[13]) + ")",
    )
    assert_true(
        worst < 1e-15,
        "qpos0 must match MuJoCo's; worst |d| = " + String(worst),
    )
    print("  PASS")


def test_keyframe_quaternion_is_repaired() raises:
    """Degenerate -> identity, unnormalised -> normalised, unit -> untouched.

    ⚠ EXPECTED VALUES ARE MUJOCO'S `m.key_qpos[0]` for this exact fixture.
    """
    print("=== keyframe quaternion repair ===")
    var r = _load(XML, String(""))
    var k0 = r[1].copy()
    var want: List[Float64] = [
        0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 1.0, R2K, 0.0, 0.0, R2K,
        2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
    ]
    var worst = 0.0
    for i in range(21):
        var e = abs(k0[i] - want[i])
        if e > worst:
            worst = e
    print("  a (key 0 0 0 0)    :", k0[3], k0[4], k0[5], k0[6],
          "  mj 1 0 0 0")
    print("  b (key .6 0 0 .6)  :", k0[10], k0[11], k0[12], k0[13],
          "  mj", R2K, "0 0", R2K)
    print("  c (key 1 0 0 0)    :", k0[17], k0[18], k0[19], k0[20],
          "  mj 1 0 0 0")
    print("  worst |d| =", worst)
    # THE DEFECT: a zero quaternion must become the identity.
    assert_true(
        abs(k0[3] - 1.0) < 1e-15 and abs(k0[4]) < 1e-15
        and abs(k0[5]) < 1e-15 and abs(k0[6]) < 1e-15,
        "a keyframe quaternion of (0,0,0,0) is DEGENERATE and MuJoCo's"
        " `mju_normalize4` replaces it with the identity; got ("
        + String(k0[3]) + ", " + String(k0[4]) + ", " + String(k0[5]) + ", "
        + String(k0[6]) + "). Taken verbatim, forward kinematics multiplies"
        " by it and the body and everything under it collapses.",
    )
    # The other half of the same rule.
    assert_true(
        abs(k0[10] - R2K) < 1e-15 and abs(k0[13] - R2K) < 1e-15,
        "a keyframe quaternion of (0.6,0,0,0.6) has norm 0.8485 and must be"
        " NORMALISED, not merely accepted; got (" + String(k0[10]) + ", "
        + String(k0[11]) + ", " + String(k0[12]) + ", " + String(k0[13]) + ")",
    )
    # ⚠ THE NEGATIVE CONTROL. An already-unit quaternion must come through
    # untouched — MuJoCo skips the divide when `|norm - 1| <= mjMINVAL`, and a
    # gate without this row would pass an implementation that renormalised
    # every quaternion on every load.
    assert_true(
        k0[17] == 1.0 and k0[18] == 0.0 and k0[19] == 0.0 and k0[20] == 0.0,
        "an already-unit keyframe quaternion must be BIT-IDENTICAL, not"
        " re-divided; got (" + String(k0[17]) + ", " + String(k0[18]) + ", "
        + String(k0[19]) + ", " + String(k0[20]) + ")",
    )
    assert_true(
        worst < 1e-15,
        "key_qpos must match MuJoCo's; worst |d| = " + String(worst),
    )
    print("  PASS")


def test_the_two_real_models() raises:
    """anymal_b (no keyframe, body quat) and pal_tiago (zero keyframe quat).

    ⚠ EXPECTED VALUES ARE MUJOCO'S OWN, read off the 3.10.0 runtime:
    anymal_b `qpos0[:7] = [0, 0, 0.58, 0, 0, 0, 1]`, and pal_tiago's
    `key_qpos[0][3] = 1` where the FILE says 0.
    """
    print("=== the two models that named these ===")
    var src = read_model_source(ANYMAL)
    var ra = _load(src[0], src[1])
    var qa = ra[0].copy()
    print("  anymal_b qpos0[:7]:", qa[0], qa[1], qa[2], qa[3], qa[4], qa[5],
          qa[6])
    assert_true(
        abs(qa[2] - 0.58) < 1e-12
        and abs(qa[3]) < 1e-12 and abs(qa[6] - 1.0) < 1e-12,
        "anymal_b's base is `pos=\"0 0 0.58\" quat=\"0 0 0 1\"`, so MuJoCo's"
        " qpos0[:7] is [0, 0, 0.58, 0, 0, 0, 1]; got [" + String(qa[0]) + ", "
        + String(qa[1]) + ", " + String(qa[2]) + ", " + String(qa[3]) + ", "
        + String(qa[4]) + ", " + String(qa[5]) + ", " + String(qa[6]) + "]",
    )

    var srt = read_model_source(TIAGO)
    var rt = _load(srt[0], srt[1])
    var kt = rt[1].copy()
    print("  pal_tiago key0[:7]:", kt[0], kt[1], kt[2], kt[3], kt[4], kt[5],
          kt[6])
    assert_true(
        abs(kt[2] - (-0.985)) < 1e-12 and abs(kt[3] - 1.0) < 1e-12,
        "pal_tiago's keyframe writes a ZERO quaternion and MuJoCo repairs it"
        " to (1,0,0,0); got qpos[3] = " + String(kt[3])
        + ". A 0 here is the reset pose the whole model starts from.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
