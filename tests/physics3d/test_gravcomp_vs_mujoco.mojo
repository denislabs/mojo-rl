"""`<body gravcomp>` — the whole of `qfrc_passive` on eight Menagerie models.

    pixi run mojo run -I . tests/physics3d/test_gravcomp_vs_mujoco.mojo

WHAT WAS MISSING. `mj_gravcomp` (`engine_passive.c:817`) holds a body up
against a fraction of its own weight:

    force = gravity * -(body_mass[i] * body_gravcomp[i])
    mj_applyFT(m, d, force, /*torque=*/0, d->xipos + 3*i, i, d->qfrc_gravcomp)

and `mj_passive` folds the result into `qfrc_passive`. The attribute was read
by NOBODY in this engine — `grep -r gravcomp mojo_rl/` returned nothing — so
every compensated link fell under its full weight from the first step.

⚠⚠ EIGHT MODELS HERE DECLARE IT, AND ON ALL EIGHT IT IS THE ENTIRE PASSIVE
FORCE. Measured against MuJoCo 3.10.0 at each model's keyframe,
`|qfrc_gravcomp|max` and `|qfrc_passive|max` agree to the last digit:

    hello_robot_stretch_3   14 bodies   46.897        (was #1 on the board)
    hello_robot_stretch     10 bodies   41.129
    flexiv_rizon4            7 bodies   25.395
    i2rt_yam                11 bodies    6.484
    agilex_piper             9 bodies    6.157
    arx_l5                   2 bodies    0.541
    google_robot            10 bodies    0.118
    shadow_dexee            16 bodies    0.0338      (was #2 on the board)

Those eight were 8 of the 16 scenes that had not reached 1e-9. Landing this
took the 1-step sweep from **69/85 to 73/85**.

⚠ WHY IT HID FOR SO LONG. It does not look like a missing force, it looks like
a slightly wrong one. `|d qfrc_actuator|` was **0.000e+00** on every one of
them — the actuators were already right — `nefc` matched, the mass matrix
matched, and the arm still drifted. An arm that holds its own pose is what a
well-tuned position servo looks like, so "the servo gains are a bit off" is
the story the numbers tell if you are looking at the actuator column. **The
tell was `|qfrc_passive|max` in `tri.py`'s triage row**, a number that had been
printed next to these models for weeks: 25.4 N·m of passive force on a model
whose XML has no spring and no damping is not a rounding error, it is a term.

⚠ THE JACOBIAN IS MuJoCo'S FORM, NOT AN EQUIVALENT ONE. `mj_jacSparse`
(`engine_core_util.c:359`) builds `jacp[k][i] = cdof[3+k] + (cdof[0:3] x
offset)[k]` and dots that with the force; `fluid_forces.mojo` next door
transports the wrench to the com reference FIRST and then dots with
`(cdof_lin, cdof_ang)`. Those are algebraically equal and not equal in
floating point, and this term lands on models we want at 1e-16.

WHAT THIS GATE CHECKS

1. A hand-built chain whose answer depends on every part of the arithmetic:
   a free root at `gravcomp="0.6"`, a hinge child at `1`, a hinge grandchild
   at the default `0`, all three frames rotated, and gravity pointing
   `(1.2, -2.5, -9.81)` so no force component can be dropped unnoticed.
   ⚠ ITS CONTROL IS THE SAME FILE WITH THE ATTRIBUTE ZEROED — MuJoCo's two
   answers differ by 1e-4 in z and put joint `j1` at 3.218e-04 instead of
   -9.1e-19, so a build that ignored the attribute cannot pass both rows.

2. `mjModel.ngravcomp`'s counting rule, which is `> 0` and not `!= 0`
   (`engine_setconst.c:102`). It gates the whole pass, so getting it wrong
   turns gravity compensation OFF for a model that asked for it.

3. flexiv_rizon4 itself — seven compensated links, one step from its
   keyframe under a fixed control.
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
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.fields.dynamics_scratch import DynamicsScratch
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.studio.stepping import (
    StudioImpFastPyr, StudioImpFastEll, StudioIntegPyr, StudioIntegEll,
    studio_cone_of, studio_uses_implicit,
)
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.gpu.constants import (
    KEY_IDX_NQPOS, MODEL_META_IDX_NGRAVCOMP, MODEL_BODY_SIZE,
    BODY_IDX_GRAVCOMP,
)

comptime DT = DType.float64

comptime RIZON4 = String(
    "references/mujoco_menagerie-main/flexiv_rizon4/scene.xml"
)

# ── the fixture, and its single-variable control ─────────────────────────
# Everything here is deliberate: the root is FREE so all six of its dofs go
# through `cross(cdof_ang, offset)`; the frames are rotated so a dropped
# rotation shows; gravity has three non-zero components so a routine that
# only handled -z would pass a level model and fail this one; and the middle
# link is compensated while its child is not, which is what makes `j1` move
# at all.
comptime XML_GC = String(
    """<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.004" gravity="1.2 -2.5 -9.81"/>
  <worldbody>
    <body name="root" pos="0.1 -0.2 1.3" euler="0.3 -0.4 0.5" gravcomp="0.6">
      <freejoint/>
      <geom type="box" size="0.12 0.08 0.05" mass="2.5"/>
      <body name="link" pos="0.2 0.05 -0.03" euler="0 0.7 0" gravcomp="1">
        <joint name="j1" type="hinge" axis="0 1 0" damping="0.02"/>
        <geom type="capsule" fromto="0 0 0 0.3 0.02 0.01" size="0.03" mass="1.1"/>
        <body name="tip" pos="0.3 0.02 0.01" euler="0.2 0 0">
          <joint name="j2" type="hinge" axis="1 0 0" damping="0.01"/>
          <geom type="sphere" size="0.04" mass="0.4"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""
)

comptime XML_GC_OFF = String(
    """<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.004" gravity="1.2 -2.5 -9.81"/>
  <worldbody>
    <body name="root" pos="0.1 -0.2 1.3" euler="0.3 -0.4 0.5" gravcomp="0">
      <freejoint/>
      <geom type="box" size="0.12 0.08 0.05" mass="2.5"/>
      <body name="link" pos="0.2 0.05 -0.03" euler="0 0.7 0" gravcomp="0">
        <joint name="j1" type="hinge" axis="0 1 0" damping="0.02"/>
        <geom type="capsule" fromto="0 0 0 0.3 0.02 0.01" size="0.03" mass="1.1"/>
        <body name="tip" pos="0.3 0.02 0.01" euler="0.2 0 0">
          <joint name="j2" type="hinge" axis="1 0 0" damping="0.01"/>
          <geom type="sphere" size="0.04" mass="0.4"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""
)

# ⚠ `gravcomp="0"` IS NOT COUNTED AND NEITHER IS A NEGATIVE ONE. MuJoCo's
# `ngravcomp` is `sum(body_gravcomp[i] > 0)`, and `mj_gravcomp` returns early
# on `!ngravcomp` — so this model's answer for `ngravcomp` is 1, not 2 and
# not 3.
comptime XML_COUNT = String(
    """<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="a" pos="0 0 1" gravcomp="0"><joint type="hinge" axis="0 1 0"/>
      <geom type="sphere" size="0.05" mass="1"/></body>
    <body name="b" pos="1 0 1" gravcomp="0.5"><joint type="hinge" axis="0 1 0"/>
      <geom type="sphere" size="0.05" mass="1"/></body>
    <body name="c" pos="2 0 1"><joint type="hinge" axis="0 1 0"/>
      <geom type="sphere" size="0.05" mass="1"/></body>
  </worldbody>
</mujoco>"""
)


def _mj_gc() -> List[Float64]:
    """MuJoCo 3.10.0 `qpos` after one step of `XML_GC`."""
    return [
        0.10001181053706082, -0.20001069588487091, 1.2999444782704086,
        0.9462629218731604, 0.09332687602364108, -0.22664241503618576,
        0.21097344052345104, 0.0003218139668192049, -7.193953236299093e-06,
    ]


def _mj_gc_off() -> List[Float64]:
    """The same file with both `gravcomp` values zeroed — the control row."""
    return [
        0.1000192, -0.20004000000000002, 1.29984304,
        0.9462808319656861, 0.0933065937729005, -0.2265663068902134,
        0.2109838268563661, -9.094947017729282e-19, -1.8280904931291798e-18,
    ]


def _mj_rizon4() -> List[Float64]:
    """MuJoCo `qpos` after one step from keyframe 0 at `_rizon4_ctrl()`."""
    return [
        8.011814209504154e-05, -0.00012054608592164993,
        2.024722647740475e-05, 1.5699073840824556,
        -0.00042127365517612256, 0.0008118112627615488,
        -0.0010345164819123619,
    ]


def _rizon4_ctrl() -> List[Float64]:
    return [0.35, -0.20, 0.15, 0.60, -0.10, 0.25, -0.30]


def _step_once(
    xml: String, base: String, ctrl: List[Float64], from_key: Bool
) raises -> List[Float64]:
    """One step through the studio's own dispatch, `qpos` out."""
    var fmd = parse_xml_full(expand_mjcf(xml, base), base)
    var dims = dims_from_flat(fmd, max_contacts=128, nmesh_verts=65536)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var d = Data[DT, DynDims, 1](dims)
    for i in range(nq):
        d.qpos.data[i] = sf.qpos0.data[i]
    if from_key:
        assert_true(
            dims.get_nkey() > 0,
            "this scene must carry a keyframe — the gate measures from it",
        )
        var nqp = Int(Float64(sf.key_meta.data[KEY_IDX_NQPOS]))
        for i in range(min(nqp, nq)):
            d.qpos.data[i] = sf.key_qpos.data[i]
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)

    var nact = dims.get_nact()
    var act = List[Scalar[DT]](
        length=nact if nact > 0 else 1, fill=Scalar[DT](0)
    )
    var use_imp = studio_uses_implicit(fmd)
    var cone = studio_cone_of(fmd)
    var imp_e = StudioImpFastEll(dims)
    var imp_p = StudioImpFastPyr(dims)
    var ell = StudioIntegEll(dims)
    var pyr = StudioIntegPyr(dims)
    for i in range(nv):
        d.qfrc.data[i] = Scalar[DT](0)
    if nact > 0:
        apply_actions_fields[DT](sf, d, ctrl, act, fmd.timestep)
    if use_imp:
        if cone == ConeType.ELLIPTIC:
            imp_e.step["cpu"](d, m)
        else:
            imp_p.step["cpu"](d, m)
    else:
        if cone == ConeType.ELLIPTIC:
            ell.step["cpu"](d, m)
        else:
            pyr.step["cpu"](d, m)
    var out = List[Float64]()
    for i in range(nq):
        out.append(Float64(d.qpos.data[i]))
    return out^


def _worst(got: List[Float64], want: List[Float64]) -> Float64:
    var w = 0.0
    for i in range(len(want)):
        var e = abs(got[i] - want[i])
        if e > w:
            w = e
    return w


def test_gravcomp_chain_and_its_ablation() raises:
    """The same file twice, one attribute apart, MuJoCo's two answers."""
    print("=== <body gravcomp> on a rotated free-root chain ===")
    var want = _mj_gc()
    var want_off = _mj_gc_off()
    var empty: List[Float64] = []

    # ⚠ THE CONTROL FIRST. It shares every line of the integrator, the
    # kinematics and the parser with the row below; if it fails, the gap is
    # not gravity compensation.
    var got_off = _step_once(XML_GC_OFF, String(""), empty, False)
    assert_true(
        len(got_off) == 9,
        "the fixture must build 9 qpos slots; got " + String(len(got_off)),
    )
    var worst_off = _worst(got_off, want_off)
    print("  gravcomp=0  worst |d(qpos)| =", worst_off)
    assert_true(
        worst_off < 1e-13,
        "the ABLATED body moved away from MuJoCo by " + String(worst_off)
        + " — the gap is in the integrator or the kinematics, not in"
        " gravity compensation. Fix this row before reading the next one.",
    )

    var got = _step_once(XML_GC, String(""), empty, False)
    var worst = _worst(got, want)
    print("  gravcomp on worst |d(qpos)| =", worst)
    for i in range(9):
        print("    qpos", i, " ours", got[i], " mj", want[i])
    assert_true(
        worst < 1e-13,
        "the COMPENSATED chain is " + String(worst) + " away from MuJoCo."
        " Matching the ablated row instead (z = 1.29984304, j1 = -9.1e-19)"
        " means `<body gravcomp>` was parsed by nobody, which is where this"
        " engine stood until 2026-08-21.",
    )

    # ⚠ VACUITY. The two reference answers must be far enough apart that
    # passing both is impossible for one behaviour. They differ by 1.14e-04
    # in z and put `j1` at 3.218e-04 against -9.1e-19.
    var spread = _worst(want, want_off)
    print("  the two reference answers differ by", spread)
    assert_true(
        spread > 1e-5,
        "the fixture and its control answer the same thing to "
        + String(spread) + "; it cannot discriminate and needs a heavier"
        " compensated link or a longer step.",
    )
    print("  PASS")


def test_ngravcomp_counts_strictly_positive() raises:
    """`engine_setconst.c:102` counts `> 0`, and that count gates the pass."""
    print("=== mjModel.ngravcomp counting rule ===")
    var fmd = parse_xml_full(
        expand_mjcf(XML_COUNT, String("")), String("")
    )
    var dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=0)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var n = Int(Float64(m.meta.data[MODEL_META_IDX_NGRAVCOMP]))
    print("  ngravcomp ours", n, " MuJoCo 1")
    assert_true(
        n == 1,
        "ngravcomp is " + String(n) + " against MuJoCo's 1. A 3 means the"
        " count is `!= 0` or unconditional; a 0 means `gravcomp` did not"
        " reach the record at all and the whole pass is switched off.",
    )
    # And the per-body values, so a right count over wrong values still fails.
    var gc_b = Float64(
        m.bodies.data[2 * MODEL_BODY_SIZE + BODY_IDX_GRAVCOMP]
    )
    var gc_c = Float64(
        m.bodies.data[3 * MODEL_BODY_SIZE + BODY_IDX_GRAVCOMP]
    )
    print("  body b", gc_b, " (0.5)   body c", gc_c, " (0.0, absent)")
    assert_true(
        abs(gc_b - 0.5) < 1e-15 and gc_c == 0.0,
        "body b reads " + String(gc_b) + " (want 0.5) and body c reads "
        + String(gc_c) + " (want 0.0 — the attribute is absent and MuJoCo's"
        " default is 0).",
    )
    print("  PASS")


def test_rizon4_seven_compensated_links() raises:
    """The real model: 25.4 N·m of passive force and nothing else."""
    print("=== flexiv_rizon4, one step from keyframe 0 ===")
    var src = read_model_source(RIZON4)
    var want = _mj_rizon4()
    var got = _step_once(src[0], src[1], _rizon4_ctrl(), True)
    assert_true(
        len(got) == 7,
        "rizon4 has 7 qpos slots; got " + String(len(got)),
    )
    var worst = _worst(got, want)
    var wi = 0
    for i in range(7):
        if abs(got[i] - want[i]) == worst:
            wi = i
    print("  worst |d(qpos)| =", worst, " at dof", wi)
    print("  ours", got[wi], " mj", want[wi])
    # ⚠ VACUITY. The arm must have MOVED off the keyframe (which is
    # [0,0,0,1.57,0,0,0]); a gate on a model that did not integrate compares
    # two copies of the start pose.
    var moved = abs(got[3] - 1.57)
    print("  dof 3 moved", moved, "rad off the keyframe")
    assert_true(
        moved > 1e-5,
        "the arm did not move — the gate would be comparing the keyframe to"
        " itself. One step at this control takes dof 3 from 1.57 to"
        " 1.5699074, i.e. 9.26e-05 rad; it moved " + String(moved) + ".",
    )
    assert_true(
        worst < 1e-12,
        "flexiv_rizon4 is " + String(worst) + " from MuJoCo. The 1-step"
        " sweep figure for this scene was 3.133e-05 before gravity"
        " compensation existed.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
