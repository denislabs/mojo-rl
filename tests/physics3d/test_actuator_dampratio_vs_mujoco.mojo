"""`<position dampratio>` must produce MuJoCo's `kv`.

    pixi run mojo run -I . tests/physics3d/test_actuator_dampratio_vs_mujoco.mojo

WHAT WAS MISSING. `dampratio` was parsed NOWHERE, so every actuator declaring
it kept `kv = 0` — an UNDAMPED position servo. It is not a rare attribute:
`<position kp="500" dampratio="1"/>` is how unitree_g1 states all 29 of its
actuators, and how much of Menagerie states theirs.

⚠⚠ THE SYMPTOM ONLY APPEARS UNDER DRIVE, WHICH IS WHY IT SURVIVED. An
undamped spring sitting at its equilibrium is indistinguishable from a damped
one, so g1 opened in the studio and left alone looked perfect — measured, it
holds z = 0.7915 for 1200 steps either way. Give it a random policy and there
is nothing to remove the energy the servos put in: g1 left the ground and
reached 2.9 m. "Fine at rest, flies under a policy" is the fingerprint.

WHAT IT IS (`engine_setconst.c:998-1035`). MuJoCo carries `dampratio` in the
SAME SLOT as `kv` under a sign convention — `user_api.cc:1211`, negative is a
literal kv, positive is a pending dampratio — and resolves it at the end of
`mj_setConst`, once the mass matrix at qpos0 exists:

    mass = sum over the transmission dofs of  dof_M0[dof] / trn^2
    kv   = dampratio * 2 * sqrt(kp * mass)

⚠ SO IT CANNOT BE DONE IN THE PARSER, and that is the whole reason this took
a new model field. `dof_M0` is the diagonal of M at qpos0 — NOT
`1/dof_invweight0`, which is the diagonal of M INVERSE and only its reciprocal
for a diagonal M. `compute_invweight0` already forms M there, so it banks the
diagonal on the way past, before `ldl_factor` overwrites it in place.

⚠⚠ AND THE RULE IS NOT ATTACHED TO THE `<position>` TAG. It lives in the
ENGINE, downstream of every spelling, so `<general biastype="affine"
biasprm="0 -kp Z">` reaches `mj_setConst` as the same record and gets the same
conversion. Reading `dampratio` off the tag alone left that spelling with
`kv = -Z` — not a weaker damper but an ANTI-damper, which the implicit
integrators SUBTRACT from the mass matrix. sharpa_wave is the pair that shows
it: the two hands are the same robot, the left spells it `<position
dampratio="0.9">` and the right spells it `<general biasprm="0 -6.95 0.9">`,
and one file apart the left stepped to 4.3e-18 while the right went to
6.2e-03.

⚠ THE EXPECTED VALUES ARE MUJOCO'S OWN, read off `-actuator_biasprm[:,2]` on
the 3.10.0 runtime. The synthetic fixture is built so they are exact round
numbers and a wrong formula cannot coincide with the right one: a box of mass
2 and half-extent 0.1 on a z hinge has `dof_M0 = 0.0133333...`, so
`kp = 300` gives `sqrt(kp*mass) = 2` exactly and `dampratio = 1` gives kv 4.
"""

from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from mojo_rl.physics3d.fields import Model, DynDims
from mojo_rl.physics3d.gpu.constants import ACT_IDX_KV, MODEL_ACTUATOR_SIZE

comptime DT = DType.float64

comptime G1 = String(
    "references/mujoco_menagerie-main/unitree_g1/scene.xml"
)

comptime SHARPA_R = String(
    "references/mujoco_menagerie-main/sharpa_wave/scene_right.xml"
)

# ⚠ FOUR ACTUATORS ON ONE JOINT, ON PURPOSE. They share a transmission, so
# every difference below is the dampratio arithmetic and nothing else.
# ⚠ `<compiler angle="radian"/>` IS LOAD-BEARING — MJCF defaults to DEGREES
# and a fixture without it has bitten this tree before.
comptime XML = String(
    """<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body>
      <joint name='j' type='hinge' axis='0 0 1'/>
      <geom type='box' size='.1 .1 .1' mass='2'/>
    </body>
  </worldbody>
  <actuator>
    <position name='a' joint='j' kp='300' dampratio='1'/>
    <position name='b' joint='j' kp='300' dampratio='0.5'/>
    <position name='c' joint='j' kp='300' kv='7'/>
    <position name='d' joint='j' kp='300'/>
  </actuator>
</mujoco>"""
)

# The SAME body and the SAME kp, stated the other way. `gaintype`/`biastype`
# are written out because a `<general>` defaults to `fixed`/`none` and a
# bias-free actuator would not reach the rule at all.
comptime XML_GENERAL = String(
    """<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body>
      <joint name='j' type='hinge' axis='0 0 1'/>
      <geom type='box' size='.1 .1 .1' mass='2'/>
    </body>
  </worldbody>
  <actuator>
    <general name='a' joint='j' gaintype='fixed' gainprm='300'
             biastype='affine' biasprm='0 -300 1'/>
    <general name='b' joint='j' gaintype='fixed' gainprm='300'
             biastype='affine' biasprm='0 -300 0.5'/>
    <general name='c' joint='j' gaintype='fixed' gainprm='300'
             biastype='affine' biasprm='0 -300 -7'/>
    <general name='d' joint='j' gaintype='fixed' gainprm='300'
             biastype='affine' biasprm='0 0 1'/>
    <general name='e' joint='j' gaintype='fixed' gainprm='300'
             biastype='affine' biasprm='0 -300 0'/>
  </actuator>
</mujoco>"""
)


def _kv_of(xml: String, base: String) raises -> List[Float64]:
    """Every actuator's FINAL kv, through the runtime path the studio uses."""
    var fmd = parse_xml_full(expand_mjcf(xml, base), base)
    # ⚠ THE MESH BUDGET IS DISCOVERED, NOT GUESSED — the same retry-on-raise
    # loop the studio's loader runs. g1's collidable hulls need 27925
    # vertices and the builder raises WITH that number, so doubling converges;
    # a hardcoded budget here would make this gate fail for a reason that has
    # nothing to do with dampratio the next time a mesh changes.
    # ⚠ THE FIRST GUESS IS 32768, NOT 0, PURELY FOR THE CLOCK. Each failed
    # attempt re-loads every STL, and starting from 0 made g1 build its meshes
    # five times — 416 s against 180 s. The loop still adapts if the model
    # grows; only the starting rung moved.
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=verts)
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
            dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var out = List[Float64]()
    for i in range(dims.get_nact()):
        out.append(
            Float64(sf.actuators.data[i * MODEL_ACTUATOR_SIZE + ACT_IDX_KV])
        )
    return out^


def test_dampratio_formula_matches_mujoco() raises:
    """The formula, the scaling, the precedence, and the negative control."""
    print("=== dampratio -> kv, synthetic ===")
    var kv = _kv_of(XML, String(""))
    assert_true(
        len(kv) == 4,
        "fixture did not parse four actuators — the gate would be vacuous",
    )
    for i in range(4):
        print("  actuator", i, " kv", kv[i])

    # MuJoCo: 2 * 1 * sqrt(300 * 0.0133333...) = 4 exactly.
    assert_true(
        abs(kv[0] - 4.0) < 1e-9,
        "dampratio=1 with kp=300 on a dof_M0 of 0.013333 must give kv 4"
        " (MuJoCo reports 4.0); got " + String(kv[0]),
    )
    # ⚠ THE ROW THAT PINS THE FACTOR. kv is LINEAR in dampratio, so half the
    # ratio is half the kv — a formula that squared it, or dropped the 2,
    # matches row 0 by luck and fails here.
    assert_true(
        abs(kv[1] - 2.0) < 1e-9,
        "dampratio=0.5 must scale kv linearly to 2.0; got " + String(kv[1]),
    )
    # ⚠ EXCLUSIVE WITH kv. MuJoCo REFUSES the file outright ("kv and
    # dampratio cannot both be defined"); a parser that must keep loading
    # cannot, so the explicit kv wins — the same precedence `inheritrange`
    # takes against an explicit `ctrlrange`, and the one a saved file has.
    assert_true(
        abs(kv[2] - 7.0) < 1e-12,
        "an explicit kv must win over dampratio; got " + String(kv[2]),
    )
    # ⚠ THE NEGATIVE CONTROL. Without it this file passes against an
    # implementation that damps EVERY position actuator.
    assert_true(
        kv[3] == 0.0,
        "an actuator declaring neither kv nor dampratio must stay UNDAMPED,"
        " got " + String(kv[3]),
    )
    print("  PASS")


def test_dampratio_on_g1_matches_mujoco() raises:
    """The real model it was found on, against MuJoCo's own 29 numbers.

    ⚠ A REAL MODEL IS NOT REDUNDANT WITH THE FIXTURE. The synthetic joint is
    a single box at the origin; g1's `dof_M0` comes from a 35-dof tree, so
    this is what catches a `dof_M0` read at the wrong pose, from the wrong
    matrix, or after `ldl_factor` has overwritten it — none of which the
    one-body fixture can distinguish.
    """
    print("=== dampratio -> kv, unitree_g1 ===")
    var src = read_model_source(G1)
    var kv = _kv_of(src[0], src[1])
    print("  nact", len(kv))
    assert_true(
        len(kv) == 29,
        "g1 has 29 actuators; parsed " + String(len(kv))
        + " — the gate would be comparing the wrong rows",
    )
    # MuJoCo 3.10.0, `-actuator_biasprm[:,2]`, printed to 10 digits.
    var want: List[Float64] = [
        43.0106827599, 39.5852058375, 9.8315037375, 15.8470106826,
        5.0551682530, 4.5578780790,
    ]
    var names: List[String] = [
        String("left_hip_pitch"), String("left_hip_roll"),
        String("left_hip_yaw"), String("left_knee"),
        String("left_ankle_pitch"), String("left_ankle_roll"),
    ]
    var worst = 0.0
    for i in range(len(want)):
        var err = abs(kv[i] - want[i])
        if err > worst:
            worst = err
        print("   ", names[i], " ours", kv[i], " MuJoCo", want[i])
        assert_true(
            err < 1e-6,
            "g1 actuator " + names[i] + ": kv is " + String(kv[i])
            + " but MuJoCo derives " + String(want[i])
            + " from dampratio=1. A kv of 0 here means dampratio was not"
            " parsed at all, and the robot flies under any policy.",
        )
    print("  worst |diff| over the six", worst)
    # ⚠ AND NONE OF THEM MAY BE ZERO. The assertions above are per-row; this
    # states the property the bug violated, for all 29.
    for i in range(len(kv)):
        assert_true(
            kv[i] > 1.0,
            "g1 actuator #" + String(i) + " has kv " + String(kv[i])
            + " — every one of its 29 servos declares dampratio='1' and none"
            " should be undamped.",
        )
    print("  PASS")


def test_general_biasprm_carries_the_same_dampratio() raises:
    """The SECOND spelling of the same rule, with its own negative controls.

    Same body, same kp, same numbers as the `<position>` fixture above — the
    only thing that changes is which tag states the ratio. Rows a and b must
    land on 4.0 and 2.0 exactly as rows 0 and 1 there do.

    ⚠ ROW `d` IS THE ONE THAT KEEPS THIS HONEST. `biasprm[1] = 0` makes it a
    VELOCITY servo, not a position one, so MuJoCo's gate
    (`gainprm[0] != -biasprm[1]` -> skip) leaves its positive `biasprm[2]`
    ALONE and the actuator really does carry `kv = -1`. Measured on 3.10.0:
    `-biasprm[:,2]` reads `[4, 2, 7, -1, 0]`. Converting every positive
    `biasprm[2]` would pass rows a-c and fail here.
    """
    print("=== <general biasprm> -> kv, synthetic ===")
    var kv = _kv_of(XML_GENERAL, String(""))
    assert_true(
        len(kv) == 5,
        "fixture did not parse five actuators — the gate would be vacuous",
    )
    for i in range(5):
        print("  actuator", i, " kv", kv[i])
    assert_true(
        abs(kv[0] - 4.0) < 1e-9,
        "biasprm='0 -300 1' is dampratio 1 on a position-like actuator and"
        " MuJoCo derives kv 4.0; got " + String(kv[0])
        + ". A raw `kv = -biasprm[2]` gives -1 here — an ANTI-damper.",
    )
    assert_true(
        abs(kv[1] - 2.0) < 1e-9,
        "kv is LINEAR in the ratio: 0.5 must halve it to 2.0; got "
        + String(kv[1]),
    )
    # ⚠ THE SIGN IS THE WHOLE DISCRIMINATOR. A NEGATIVE biasprm[2] is a
    # literal kv and must pass through untouched — no sqrt, no mass.
    assert_true(
        abs(kv[2] - 7.0) < 1e-12,
        "a NEGATIVE biasprm[2] is a literal kv and must survive verbatim;"
        " got " + String(kv[2]),
    )
    # ⚠ NEGATIVE CONTROL — see the docstring. MuJoCo keeps this one at -1.
    assert_true(
        abs(kv[3] + 1.0) < 1e-12,
        "biasprm='0 0 1' is NOT position-like (gainprm[0] != -biasprm[1]), so"
        " MuJoCo does not convert it and it keeps kv -1; got " + String(kv[3])
        + ". Converting it would invent damping the reference does not have.",
    )
    # ⚠ NEGATIVE CONTROL — a zero must stay a zero, not become a ratio of 0
    # that some later sqrt turns into something else.
    assert_true(
        kv[4] == 0.0,
        "biasprm[2] = 0 must leave the servo UNDAMPED; got " + String(kv[4]),
    )
    print("  PASS")


def test_general_dampratio_on_sharpa_wave_matches_mujoco() raises:
    """The model it was found on: 22 servos, all spelled `<general biasprm>`.

    ⚠ THE FIXTURE CANNOT CATCH WHAT THIS CATCHES. Its single box makes
    `kp * mass` a round number by construction; sharpa's reflected inertias
    come from a 22-dof hand and every one of these is irrational, so a
    formula that agrees with MuJoCo here is using MuJoCo's `dof_M0`.
    """
    print("=== <general biasprm> -> kv, sharpa_wave right hand ===")
    var src = read_model_source(SHARPA_R)
    var kv = _kv_of(src[0], src[1])
    print("  nact", len(kv))
    assert_true(
        len(kv) == 22,
        "sharpa_wave's right hand has 22 actuators; parsed "
        + String(len(kv)) + " — the gate would be comparing the wrong rows",
    )
    # MuJoCo 3.10.0, `-actuator_biasprm[:,2]`, printed at full precision.
    var want: List[Float64] = [
        0.2844001777075252, 0.403408719431069, 0.20380858182545714,
        0.2403441941312949, 0.04189004080632966, 0.2078418814338731,
    ]
    var names: List[String] = [
        String("thumb_CMC_FE"), String("thumb_CMC_AA"),
        String("thumb_MCP_FE"), String("thumb_MCP_AA"),
        String("thumb_IP"), String("index_MCP_FE"),
    ]
    var worst = 0.0
    for i in range(len(want)):
        var err = abs(kv[i] - want[i])
        if err > worst:
            worst = err
        print("   ", names[i], " ours", kv[i], " MuJoCo", want[i])
        assert_true(
            err < 1e-9,
            "sharpa_wave actuator " + names[i] + ": kv is " + String(kv[i])
            + " but MuJoCo derives " + String(want[i]) + " from the"
            " dampratio 0.9 in its `biasprm`.",
        )
    print("  worst |diff| over the six", worst)
    # ⚠ THE PROPERTY THE BUG VIOLATED, stated for all 22: every one of them
    # was NEGATIVE — an anti-damper — and the smallest true value is 0.0287.
    for i in range(len(kv)):
        assert_true(
            kv[i] > 0.02,
            "sharpa_wave actuator #" + String(i) + " has kv " + String(kv[i])
            + " — a negative or zero value here is the anti-damper this gate"
            " exists for.",
        )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
