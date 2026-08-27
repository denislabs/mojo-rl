"""`<motor site="...">` — the transmission both Menagerie quadrotors fly on.

    pixi run mojo run -I . tests/physics3d/test_site_transmission_vs_mujoco.mojo

WHAT WAS MISSING. `_fill_actuator_transmission` resolved `joint=` and
`tendon=` and nothing else, so an actuator driven through a SITE kept
`trn_n = 0`: it took a slot in `nact`, consumed its control, and applied ZERO
FORCE. `skydio_x2` and `bitcraze_crazyflie_2` drive every rotor that way —
`<motor site="thrust1" gear="0 0 1 0 0 -.0201"/>` — so **neither aircraft had
any thrust in this engine.** MuJoCo answers skydio's first step with
`qfrc_actuator = [0, 0, 12.6, 2.6e-17, -0.028, 0.00804]` at ctrl
(3.0, 3.2, 3.1, 3.3); we answered six zeros.

`mjTRN_SITE` (`engine_core_smooth.c`) is short:

    mj_jacSite(m, d, jac, jacS, id);
    length[i] = 0;
    wrench[0:3] = site_xmat * gear[0:3];   wrench[3:6] = site_xmat * gear[3:6];
    moment      = jac^T wrench[0:3] + jacS^T wrench[3:6];

⚠⚠ `gear` IS A SIX-VECTOR HERE, AND WE STORED ONE FLOAT. `ActuatorData.gear`
was a scalar parsed with `_parse_float("0 0 1 0 0 -.0201")`, i.e. **0** — so
even the magnitude was wrong before the transmission was. A site actuator's
gear is a WRENCH in the site frame, not a ratio, and the whole of it is baked
into the moment; it must NOT be multiplied in again the way a joint or
fixed-tendon transmission's scalar is.

⚠ `length` IS 0, NOT THE SITE'S ANYTHING. A `<position site=>` therefore
servos toward 0. That is MuJoCo's definition, not an omission.

WHY IT LANDS NOW. The obstacle was never the arithmetic — `jac_site` and
`jac_point` have been in `dynamics/jac_point.mojo` since IK — it was ORDERING:
`apply_actions_fields` runs BEFORE the integrator's forward kinematics, so
`xpos`/`xquat`/`cdof` describe the pose one substep back. `12f77342` added
`dynamics/pose_transmission.mojo`, which refreshes all three at the current
`qpos` before reading them and is already called from the CPU env, the studio
and the drive harness. This is an addition to one loop there.

⚠ THE SITE'S WORLD ROTATION IS COMPOSED, NOT READ. `Data` has no `site_xmat`;
it has body `xquat` and the site's LOCAL quat on the site record. Rotating a
vector by `R_body · R_site` is two `quat_rotate` calls and needs no matrix —
and it avoids binding `site_xpos`, which is EMPTY on every site-less model and
crashed three solver tests when `tendon.mojo` tried it (see `_site_world`).

⚠ CPU ONLY. `apply_actions_kernel_gpu` still walks the `(qadr, dadr, coef)`
triples, so a site actuator produces no force on the batched GPU path. The
parser prints a count at load. No batched env in this tree drives one.
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
from mojo_rl.physics3d.dynamics.pose_transmission import (
    apply_pose_transmission,
)
from mojo_rl.physics3d.studio.stepping import StudioIntegPyr
from mojo_rl.physics3d.gpu.constants import KEY_IDX_NQPOS

comptime DT = DType.float64

comptime SKYDIO = String(
    "references/mujoco_menagerie-main/skydio_x2/scene.xml"
)
comptime CRAZYFLIE = String(
    "references/mujoco_menagerie-main/bitcraze_crazyflie_2/scene.xml"
)

# ── the fixture that tests the ROTATION, which the quadrotors do not ──────
# Both Menagerie drones put their thrust sites on a level body with an
# IDENTITY local quat, so `site_xmat` is the identity there and a version
# that dropped the rotation entirely would still fly them. This body is
# turned 90 deg about y and its site a further 45 deg about z, and the gear
# has all six components non-zero — so every entry of the answer depends on
# the composed rotation. Gravity is off, so `qfrc_actuator` IS the whole of
# `d.qfrc`.
comptime XML_SITE = String(
    """<mujoco>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="b" pos="0 0 1" quat="0.70710678118654752 0 0.70710678118654752 0">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.02" mass="1"/>
      <site name="s" pos="0.05 0.03 0.01"
            quat="0.92387953251128674 0 0 0.38268343236508977" size="0.005"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="m" site="s" gear="0.3 -0.2 1 0.05 -0.07 0.02" ctrlrange="-10 10"/>
  </actuator>
</mujoco>"""
)


def _mj_site() -> List[Float64]:
    """MuJoCo 3.10.0 `qfrc_actuator` on `XML_SITE` at ctrl = 2.5."""
    return [
        2.5,
        0.17677669529663684,
        -0.88388347648318444,
        0.28536426740299797,
        -0.1515165042944957,
        0.032322330470336308,
    ]


def _mj_skydio() -> List[Float64]:
    """MuJoCo `qfrc_actuator` from keyframe 0 at ctrl (3.0, 3.2, 3.1, 3.3)."""
    return [
        0.0, 0.0, 12.600000000000001, 2.6423307986078724e-17,
        -0.027999999999999966, 0.008040000000000002,
    ]


def _mj_crazyflie() -> List[Float64]:
    """MuJoCo `qfrc_actuator` from keyframe 0 at ctrl (0.3, .02, -.01, .005)."""
    return [
        0.0, 0.0, 0.29999999999999999, -2.0000000000000002e-07,
        1.0000000000000001e-07, -5.0000000000000004e-08,
    ]


def _qfrc(xml: String, ctrl: List[Float64]) raises -> List[Float64]:
    """`d.qfrc` after both actuation passes, at qpos0."""
    var fmd = parse_xml_full(expand_mjcf(xml, String("")), String(""))
    var dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=0)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
    var act = List[Scalar[DT]](
        length=dims.get_nact() if dims.get_nact() > 0 else 1,
        fill=Scalar[DT](0),
    )
    var sc = DynamicsScratch[DT, DynDims, 1](dims)
    apply_actions_fields[DT](sf, d, ctrl, act, fmd.timestep)
    apply_pose_transmission[DT](sf, m, d, sc, ctrl, act, fmd.timestep)
    var out = List[Float64]()
    for i in range(dims.get_nv()):
        out.append(Float64(d.qfrc.data[i]))
    return out^


def _qfrc_scene(path: String, ctrl: List[Float64]) raises -> List[Float64]:
    """The same, for a Menagerie scene started from keyframe 0."""
    var src = read_model_source(path)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var dims = dims_from_flat(fmd, max_contacts=128, nmesh_verts=65536)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var nq = dims.get_nq()
    var d = Data[DT, DynDims, 1](dims)
    assert_true(
        dims.get_nkey() > 0,
        "this scene must carry a keyframe — the gate measures from it",
    )
    var nqp = Int(Float64(sf.key_meta.data[KEY_IDX_NQPOS]))
    for i in range(nq):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(min(nqp, nq)):
        d.qpos.data[i] = sf.key_qpos.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
    var act = List[Scalar[DT]](length=dims.get_nact(), fill=Scalar[DT](0))
    var sc = DynamicsScratch[DT, DynDims, 1](dims)
    apply_actions_fields[DT](sf, d, ctrl, act, fmd.timestep)
    apply_pose_transmission[DT](sf, m, d, sc, ctrl, act, fmd.timestep)
    var out = List[Float64]()
    for i in range(dims.get_nv()):
        out.append(Float64(d.qfrc.data[i]))
    return out^


def _worst(got: List[Float64], want: List[Float64]) -> Float64:
    var w = 0.0
    for i in range(len(want)):
        var e = abs(got[i] - want[i])
        if e > w:
            w = e
    return w


def test_site_gear_is_a_wrench_in_the_site_frame() raises:
    """Every entry depends on `R_body · R_site` and on all six gear terms."""
    print("=== site transmission, rotated body and rotated site ===")
    var want = _mj_site()
    var ctrl: List[Float64] = [2.5]
    var got = _qfrc(XML_SITE, ctrl)
    assert_true(
        len(got) == 6, "one free body — got " + String(len(got)) + " dofs"
    )
    for i in range(6):
        print("  dof", i, " ours", got[i], " mj", want[i])
    var worst = _worst(got, want)
    print("  worst |d(qfrc)| =", worst)

    # ⚠ VACUITY, TWO WAYS. (1) Every reference entry must be far from zero, or
    # "no force at all" would pass. (2) ctrl = 0 must give EXACTLY zero, or the
    # comparison is measuring something that is not the actuator.
    var small = 0
    for i in range(6):
        if abs(want[i]) < 1e-3:
            small += 1
    assert_true(
        small == 0,
        "every reference entry must be well away from 0 or the gate cannot"
        " tell a working transmission from a dead one",
    )
    var zero_ctrl: List[Float64] = [0.0]
    var zero = _qfrc(XML_SITE, zero_ctrl)
    var zmax = 0.0
    for i in range(6):
        if abs(zero[i]) > zmax:
            zmax = abs(zero[i])
    print("  ctrl = 0 gives |qfrc|max", zmax)
    assert_true(
        zmax < 1e-15,
        "with gravity off and ctrl 0 there is no force in this model;"
        " |qfrc|max = " + String(zmax),
    )

    assert_true(
        worst < 1e-12,
        "ours vs MuJoCo, worst |d(qfrc)| = " + String(worst)
        + ". All zeros means the actuator resolved to `trn_n = 0`. A wrong"
        " dof 0 with the rest close means the gear is being read as a scalar"
        " (`_parse_float` takes the FIRST of six). A right dof 0 with wrong"
        " torques means the site's local quat is not composed with the"
        " body's.",
    )
    print("  PASS")


def test_skydio_has_thrust() raises:
    """`skydio_x2` — four rotors, all on `site=` transmissions."""
    print("=== skydio_x2, keyframe 0, ctrl (3.0, 3.2, 3.1, 3.3) ===")
    var want = _mj_skydio()
    var ctrl: List[Float64] = [3.0, 3.2, 3.1, 3.3]
    var got = _qfrc_scene(SKYDIO, ctrl)
    assert_true(len(got) == 6, "a free-flying body — got " + String(len(got)))
    for i in range(6):
        print("  dof", i, " ours", got[i], " mj", want[i])
    var worst = _worst(got, want)
    print("  worst |d(qfrc)| =", worst)
    # ⚠ VACUITY. 12.6 N of lift is the whole point; a dead drone reads 0.
    assert_true(
        want[2] > 10.0,
        "the reference must carry real thrust or this gate is vacuous",
    )
    assert_true(
        worst < 1e-12,
        "skydio must feel its rotors; worst |d(qfrc)| = " + String(worst)
        + ". Before this it was 12.6 — the whole of the lift.",
    )
    print("  PASS")


def test_crazyflie_has_thrust() raises:
    """`bitcraze_crazyflie_2` — thrust plus three moment actuators, one site.

    ⚠ FOUR ACTUATORS ON THE **SAME** SITE, differing only in `gear`. It is the
    fixture that catches a transmission keyed on the site instead of on the
    actuator: three of these four have a gear whose translation part is
    entirely zero and whose moment is 1e-05, so their contribution is 1e-07
    against a 0.3 thrust — visible only because the comparison is per-dof.

    ⚠⚠ THIS GATE COMPARES `qfrc_actuator`, NOT A TRAJECTORY, AND FOR A REASON.
    This model ships `integrator="RK4"`, which evaluates the derivative four
    times per step at four different poses and recomputes the transmission at
    each; `apply_pose_transmission` runs once per control substep and stays
    frozen at the stage-0 moment. So the FORCE is exact (0.0 here) while the
    one-step pose is 3.314e-13 out — a number that belongs to the integrator
    seam, not to this feature.

    ⚠⚠ AND THAT SENTENCE USED TO SAY 9.200e-06, WHICH WAS NOT THIS AT ALL.
    "Rewriting the model to Euler gives 5.294e-23" was read as proof that the
    frozen transmission owned crazyflie's whole board residual. It proved only
    that the residual was an INTEGRATOR difference — and the integrator
    difference was that the studio's dispatch never built RK4 and stepped the
    scene with Euler. `test_studio_honours_option_rk4` is the gate for that;
    it is fixed, and what is left here is the 3.314e-13. An ablation that
    moves a residual to zero names an AXIS, not a mechanism on that axis.
    """
    print("=== bitcraze_crazyflie_2, keyframe 0 ===")
    var want = _mj_crazyflie()
    var ctrl: List[Float64] = [0.3, 0.02, -0.01, 0.005]
    var got = _qfrc_scene(CRAZYFLIE, ctrl)
    assert_true(len(got) == 6, "a free-flying body — got " + String(len(got)))
    for i in range(6):
        print("  dof", i, " ours", got[i], " mj", want[i])
    var worst = _worst(got, want)
    print("  worst |d(qfrc)| =", worst)
    assert_true(
        want[2] > 0.1 and abs(want[3]) > 1e-8,
        "the reference must carry both a thrust and a moment or the gate"
        " cannot see the three moment actuators at all",
    )
    assert_true(
        worst < 1e-12,
        "crazyflie must feel its rotor and its three moments; worst"
        " |d(qfrc)| = " + String(worst),
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
