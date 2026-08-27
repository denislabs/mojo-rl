"""`inheritrange` must produce MuJoCo's `ctrlrange`, and the condim a model
needs must be visible to whoever sizes `MAX_CONDIM`.

    pixi run mojo run -I . tests/physics3d/test_inheritrange_and_condim_vs_mujoco.mojo

── 1. `inheritrange` ────────────────────────────────────────────────────────
`<position inheritrange="X">` sets the actuator's `ctrlrange` from the
TRANSMISSION TARGET's range, about its midpoint (`user_objects.cc:7138`):

    mean   = 0.5*(hi + lo)
    radius = 0.5*(hi - lo) * X
    ctrlrange = [mean - radius, mean + radius]

We did not implement it, so an actuator declaring it kept the default
`ctrlrange` and `ctrllimited=false`.

⚠⚠ THE CONSEQUENCE IS A SERVO THAT TARGETS A POSE THE JOINT FORBIDS. MuJoCo
clamps `ctrl` to `ctrlrange`. On spot, whose knees have range
[-2.793, -0.254] while `qpos0` puts them at 0, a commanded 0 is clamped to
-0.254, so the actuator pulls each knee INTO its limit — measured, -127.2 N.m
on all four at reset. Unclamped, ours held the knee at 0, a configuration the
joint-limit constraint is simultaneously pushing out of. Implementing it moved
spot's free drop from diverging at step 273 to step 617, and its lowest body
height from 0.56 to 0.277 against MuJoCo's own 0.2713.

⚠ EXCLUSIVE WITH `ctrlrange`: MuJoCo raises when both are given. A parser that
must keep loading cannot raise, so an explicit `ctrlrange` WINS — which is also
the precedence a saved XML has, since MuJoCo always converts `inheritrange` to
an explicit `ctrlrange` on save.

── 2. the condim a model needs ──────────────────────────────────────────────
`contact_solve` clamps `condim > MAX_CONDIM` down to it SILENTLY, in both cone
branches. The comptime path has `ParsedModel.MAX_CONDIM` and every def passes
it; the RUNTIME path had no equivalent, so the studio's hardcoded 3 could not
even be compared against what the file wanted — spot's `condim="6"` feet were
solved as condim 3, torsional and rolling friction dropped, with no indication.

⚠ THE TWO SCANNERS MUST AGREE, and comparing them is what found a bug:
`_scan_max_condim` matched only `condim="` and missed `condim='6'`, returning
the floor of 3 — the SILENT under-estimate its own docstring warns about
("the model spins and rolls without resistance"). The runtime side reads the
PARSED attribute and is quote-agnostic, so the two disagreed 3 vs 6 on a
single-quoted fixture. No asset in the tree used single quotes (audited: zero
files), so that was latent rather than live — which is exactly why a gate is
worth more here than the absence of a bug report.

⚠ NOTHING RESIZES ITSELF FROM THIS. `MAX_CONDIM` is a comptime parameter of the
integrator; recording the requirement is what lets a caller check the one it
built, not a mechanism for changing it.
"""

from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser import parse_xml
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, read_model_source,
)
from mojo_rl.physics3d.fields import Model, DynDims
from mojo_rl.physics3d.gpu.constants import MODEL_META_IDX_MAX_CONDIM

comptime DT = DType.float64

comptime SPOT = String(
    "references/mujoco_menagerie-main/boston_dynamics_spot/scene.xml"
)

# A hinge with a range that is NOT symmetric about zero, so a wrong formula
# (taking the range verbatim, or forgetting the midpoint) cannot coincide with
# the right one. inheritrange=1 -> exactly the joint range.
#
# ⚠ `<compiler angle="radian"/>` IS LOAD-BEARING AND WAS MISSING AT FIRST.
# MJCF's default is DEGREES, so `range='-2.0 0.5'` compiled to
# [-0.0349, 0.0087] rad and the assertion failed against a CORRECT
# implementation — the fixture was wrong, not the code. Spot's own XML says
# `angle="radian"`, which is why the real-model test passed while this one did
# not.
comptime XML_IR = String(
    """<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body>
      <joint name='j' type='hinge' axis='0 0 1' range='-2.0 0.5'/>
      <geom type='sphere' size='0.1' mass='1'/>
    </body>
  </worldbody>
  <actuator>
    <position name='a' joint='j' kp='10' inheritrange='1'/>
    <position name='b' joint='j' kp='10' inheritrange='0.8'/>
    <position name='c' joint='j' kp='10' inheritrange='1' ctrlrange='-9 9'/>
    <position name='d' joint='j' kp='10'/>
  </actuator>
</mujoco>"""
)

# Single-quoted `condim`, the style `_scan_max_condim` used to miss entirely.
comptime XML_C6 = String(
    "<mujoco><worldbody><geom type='plane' size='0 0 1'/>"
    "<body><freejoint/><geom type='sphere' size='.1' condim='6'/></body>"
    "</worldbody></mujoco>"
)
comptime PM_C6 = parse_xml(XML_C6)


def test_inheritrange_matches_mujoco() raises:
    """The midpoint formula, and the precedence when `ctrlrange` is also set.

    ⚠ EXPECTED VALUES ARE DERIVED FROM MUJOCO'S FORMULA, not from our output:
    for range [-2.0, 0.5], mean = -0.75 and half-width = 1.25, so
    inheritrange=1 gives [-2.0, 0.5] and 0.8 gives [-1.75, 0.25].
    """
    print("=== inheritrange ===")
    var fmd = parse_xml_full(XML_IR, String(""))
    assert_true(
        len(fmd.actuators) == 4,
        "fixture did not parse four actuators — the gate would be vacuous",
    )
    for i in range(4):
        ref a = fmd.actuators[i]
        print(
            "  actuator", i, " ctrl_limited", a.is_ctrl_limited,
            " range [", a.ctrl_min, ",", a.ctrl_max, "]",
        )

    ref a0 = fmd.actuators[0]
    assert_true(
        a0.is_ctrl_limited
        and abs(a0.ctrl_min - (-2.0)) < 1e-12
        and abs(a0.ctrl_max - 0.5) < 1e-12,
        "inheritrange=1 must give exactly the joint range [-2.0, 0.5], got ["
        + String(a0.ctrl_min) + ", " + String(a0.ctrl_max) + "]",
    )
    ref a1 = fmd.actuators[1]
    assert_true(
        a1.is_ctrl_limited
        and abs(a1.ctrl_min - (-1.75)) < 1e-12
        and abs(a1.ctrl_max - 0.25) < 1e-12,
        "inheritrange=0.8 must scale about the MIDPOINT -0.75, giving"
        " [-1.75, 0.25], got ["
        + String(a1.ctrl_min) + ", " + String(a1.ctrl_max) + "]",
    )
    # ⚠ EXPLICIT `ctrlrange` WINS — MuJoCo rejects the combination outright and
    # a parser that must keep loading cannot, so the explicit value stands.
    ref a2 = fmd.actuators[2]
    assert_true(
        abs(a2.ctrl_min - (-9.0)) < 1e-12
        and abs(a2.ctrl_max - 9.0) < 1e-12,
        "an explicit ctrlrange must win over inheritrange, got ["
        + String(a2.ctrl_min) + ", " + String(a2.ctrl_max) + "]",
    )
    # ⚠ THE NEGATIVE CONTROL. Without it this file would pass against an
    # implementation that set a range on EVERY position actuator.
    ref a3 = fmd.actuators[3]
    assert_true(
        not a3.is_ctrl_limited,
        "an actuator declaring NO inheritrange must stay unlimited — it got ["
        + String(a3.ctrl_min) + ", " + String(a3.ctrl_max) + "]",
    )
    print("  PASS")


def test_inheritrange_on_spot_matches_mujoco() raises:
    """The real model it was found on, against MuJoCo's own numbers.

    ⚠ MEASURED ON THE 3.10.0 RUNTIME: spot's knee actuators report
    `ctrlrange = [-2.7929, -0.2544]` and `actuator_ctrllimited` true on all
    twelve. `qpos0` puts the knees at 0, which is OUTSIDE that range — which is
    the whole point, and why leaving it unset let the servo fight the limit.
    """
    print("=== inheritrange on spot ===")
    var src = read_model_source(SPOT)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var n_lim = 0
    var knee_lo = 0.0
    var knee_hi = 0.0
    for i in range(len(fmd.actuators)):
        ref a = fmd.actuators[i]
        if a.is_ctrl_limited:
            n_lim += 1
        if i == 2:
            knee_lo = a.ctrl_min
            knee_hi = a.ctrl_max
    print("  ctrllimited on", n_lim, "of", len(fmd.actuators))
    print("  knee ctrlrange [", knee_lo, ",", knee_hi, "]  (MuJoCo:"
          " [-2.7929, -0.2544])")
    assert_true(
        n_lim == len(fmd.actuators) and len(fmd.actuators) == 12,
        "MuJoCo reports actuator_ctrllimited true on all 12 of spot's"
        " actuators; we have " + String(n_lim) + " of "
        + String(len(fmd.actuators)),
    )
    assert_true(
        abs(knee_lo - (-2.7929)) < 1e-3 and abs(knee_hi - (-0.2544)) < 1e-3,
        "spot's knee ctrlrange must be MuJoCo's [-2.7929, -0.2544], got ["
        + String(knee_lo) + ", " + String(knee_hi) + "]",
    )
    print("  PASS")


def test_model_condim_requirement_is_recorded() raises:
    """Both parsers report the condim the model needs, and agree.

    ⚠ THE SINGLE-QUOTED FIXTURE IS THE POINT. `_scan_max_condim` matched only
    `condim="` and returned the floor of 3 for it, while the runtime side read
    the parsed attribute and returned 6.
    """
    print("=== the condim a model needs ===")
    print("  comptime scanner, condim='6' model ->", PM_C6.MAX_CONDIM)
    assert_true(
        PM_C6.MAX_CONDIM == 6,
        "`condim='6'` (SINGLE quotes, which MJCF admits) scanned as "
        + String(PM_C6.MAX_CONDIM)
        + " — under-estimating is silent: the extra friction rows are built"
        " into a workspace nothing reads and the model rolls without"
        " resistance.",
    )

    var src = read_model_source(SPOT)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=0)
    var m = Model[DT, DynDims](dims)
    try:
        build_model_runtime[DT](fmd, dims, m)
    except e:
        # spot needs a mesh budget this fixture does not give it; the meta
        # write happens before the hulls load, so the value is still valid.
        pass
    var mc = Int(Float64(m.meta.data[MODEL_META_IDX_MAX_CONDIM]))
    print("  spot -> fmd.max_condim", fmd.max_condim, " META", mc)
    assert_true(
        fmd.max_condim == 6 and mc == 6,
        "spot's feet declare condim=6, so the runtime parser must report 6 and"
        " write it to META; got " + String(fmd.max_condim) + " / "
        + String(mc),
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
