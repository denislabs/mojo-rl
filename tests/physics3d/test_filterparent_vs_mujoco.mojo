"""`<flag filterparent="disable">` (AUD-34), against 3.12.

    pixi run mojo run -I . tests/physics3d/test_filterparent_vs_mujoco.mojo

MuJoCo skips a contact between a body and the body it hangs from. That skip is
the DEFAULT, not the law: `filterBodyPair`'s weld-parent test reads

    if ((!dsbl_filterparent && weldbody1 != 0 && weldbody2 != 0) &&
        (weldbody1 == weldparent2 || weldbody2 == weldparent1)) return 1;

(`engine_collision_driver.c:311-314`), and `<flag filterparent="disable"/>`
clears `dsbl_filterparent`. A model that sets it WANTS the parent-child pair —
a gripper pad against the hand it hangs from is the usual reason.

We applied the skip unconditionally, so those models silently lost every
parent-child contact. Not a numeric drift: a contact that is simply absent.

⚠ THE FLAG GATES EXACTLY ONE CLAUSE. Same-weldbody, both-dof-less and the
`<contact><exclude>` scan are all unconditional in MuJoCo and stay
unconditional here. Reading the flag as "no body filtering at all" would let
two geoms on the SAME body collide, which MuJoCo never does whatever the flag
says — so the third test below plants exactly that case and requires it still
to be filtered.

The fixture is a slide-jointed box with a slide-jointed sphere hanging 2 cm
below its own surface, i.e. penetrating its parent by construction:

    default   ncon 0    (the pair is filtered)
    disabled  ncon 1    (cg vs pg, dist -0.11)

and 40 steps later the child sits at qpos 0.158 instead of 0.
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from noeira.physics3d.fields import Model, Data, DynDims
from noeira.physics3d.parser.full_parser import parse_xml_full
from noeira.physics3d.parser.runtime_load import (
    dims_from_flat,
    build_model_runtime,
    spec_fields_runtime,
)
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.collision.contact_detection import detect_contacts
from noeira.physics3d.gpu.constants import META_IDX_NUM_CONTACTS
from noeira.physics3d.studio.stepping import StudioIntegPyr

comptime DT = DType.float64
comptime STEPS = 40
# Set from the measurement: the two legs land at ~1e-15 on qpos after 40
# steps. 1e-10 is five orders of slack, and still nine orders below the 0.158
# the flag itself is worth.
comptime TOL = 1e-10

comptime PARENT_XML = String(
    """<mujoco model="filterparent">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81">FLAGLINE</option>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="parent" pos="0 0 0.5">
      <joint name="jp" type="slide" axis="0 0 1"/>
      <geom name="pg" type="box" size="0.3 0.3 0.05" density="800"/>
      <body name="child" pos="0 0 0.02">
        <joint name="jc" type="slide" axis="0 0 1"/>
        <geom name="cg" type="sphere" size="0.08" density="800"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
)

# Two geoms on ONE body, overlapping. MuJoCo filters this pair through the
# same-weldbody clause, which the flag does NOT gate.
comptime SAME_BODY_XML = String(
    """<mujoco model="same_body_pair">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81">
    <flag filterparent="disable"/>
  </option>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="a" pos="0 0 0.5">
      <joint name="ja" type="slide" axis="0 0 1"/>
      <geom name="g1" type="sphere" size="0.1" density="800"/>
      <geom name="g2" type="sphere" size="0.1" pos="0.05 0 0" density="800"/>
    </body>
  </worldbody>
</mujoco>
"""
)


def _xml(flag: String) -> String:
    return PARENT_XML.replace(String("FLAGLINE"), flag)


def _our_ncon(xml: String) raises -> Int:
    var fmd = parse_xml_full(xml, String("."))
    var dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=64)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]
    forward_kinematics["cpu"](d, m)
    detect_contacts["cpu"](d, m)
    return Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))


def _mj_ncon(xml: String) raises -> Int:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(xml)
    var d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    return Int(py=d.ncon)


def _our_roll(xml: String) raises -> List[Float64]:
    var fmd = parse_xml_full(xml, String("."))
    var dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=64)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]
    var integ = StudioIntegPyr(dims)
    for _ in range(STEPS):
        integ.step["cpu"](d, m)
    var out = List[Float64]()
    for i in range(dims.get_nq()):
        out.append(Float64(d.qpos.data[i]))
    return out^


def _mj_roll(xml: String) raises -> List[Float64]:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(xml)
    var d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    for _ in range(STEPS):
        mujoco.mj_step(m, d)
    var out = List[Float64]()
    for i in range(Int(py=m.nq)):
        out.append(Float64(py=d.qpos[i]))
    return out^


def test_the_default_still_filters_the_parent() raises:
    """The control: the path the fix must not have moved."""
    print("=== default: the parent-child pair is filtered ===")
    var xml = _xml(String(""))
    var ours = _our_ncon(xml)
    var refs = _mj_ncon(xml)
    print("  ncon  ours", ours, " MuJoCo", refs, " (want 0)")
    assert_true(
        ours == refs and refs == 0,
        "with the flag absent MuJoCo says ncon " + String(refs) + " and we"
        " say " + String(ours) + "; both should be 0",
    )
    var oq = _our_roll(xml)
    var mq = _mj_roll(xml)
    for i in range(2):
        print("  qpos", i, " ours", oq[i], " MuJoCo", mq[i])
        assert_true(
            abs(oq[i] - mq[i]) <= TOL,
            "default leg qpos" + String(i) + ": ours " + String(oq[i])
            + ", MuJoCo " + String(mq[i]),
        )


def test_disabling_the_flag_produces_the_contact() raises:
    """The defect: with the flag set, the contact must APPEAR."""
    print("=== filterparent='disable': the pair collides ===")
    var xml = _xml(String("<flag filterparent=\"disable\"/>"))
    var ours = _our_ncon(xml)
    var refs = _mj_ncon(xml)
    print("  ncon  ours", ours, " MuJoCo", refs, " (want 1)")
    assert_true(
        refs == 1,
        "MuJoCo no longer emits the parent-child contact on this fixture"
        " (ncon " + String(refs) + "): the premise has moved and the geometry"
        " needs re-checking before this file means anything",
    )
    assert_true(
        ours == refs,
        "ncon ours " + String(ours) + ", MuJoCo " + String(refs)
        + ": `<flag filterparent=\"disable\">` did not reach the body-pair"
        " filter",
    )
    var oq = _our_roll(xml)
    var mq = _mj_roll(xml)
    for i in range(2):
        print("  qpos", i, " ours", oq[i], " MuJoCo", mq[i])
        assert_true(
            abs(oq[i] - mq[i]) <= TOL,
            "disabled leg qpos" + String(i) + ": ours " + String(oq[i])
            + ", MuJoCo " + String(mq[i]),
        )
    # ⚠ NON-VACUITY: the two legs must actually differ, or "ours == MuJoCo"
    # on both is consistent with the flag being ignored on both sides.
    var dq = _mj_roll(_xml(String("")))
    var gap = abs(mq[1] - dq[1])
    print("  MuJoCo's own default-vs-disabled gap on qpos1:", gap)
    assert_true(
        gap > 1e-3,
        "MuJoCo gives the same rollout with and without the flag on this"
        " fixture, so neither test above can see it: gap " + String(gap),
    )


def test_the_flag_does_not_unfilter_a_same_body_pair() raises:
    """⚠ THE OVER-FIX CONTROL. `filterparent` gates ONE clause of
    `filterBodyPair`. Reading it as "skip the body filter" would let two
    geoms on the same body collide — which MuJoCo never does, whatever the
    flag says, because the same-weldbody test sits above it unconditionally.
    """
    print("=== filterparent='disable' does NOT unfilter one body's own geoms ===")
    var ours = _our_ncon(SAME_BODY_XML)
    var refs = _mj_ncon(SAME_BODY_XML)
    print("  two overlapping geoms on body 'a':  ncon ours", ours,
          " MuJoCo", refs, " (want 0)")
    assert_true(
        refs == 0,
        "MuJoCo now collides two geoms of the same body under"
        " filterparent=disable (ncon " + String(refs) + "); this control's"
        " premise has moved",
    )
    assert_true(
        ours == 0,
        "we emit " + String(ours) + " contact(s) between two geoms of the"
        " SAME body: the flag was read as disabling the whole body filter"
        " rather than its weld-parent clause",
    )


def main() raises:
    var suite = TestSuite()
    suite.test[test_the_default_still_filters_the_parent]()
    suite.test[test_disabling_the_flag_produces_the_contact]()
    suite.test[test_the_flag_does_not_unfilter_a_same_body_pair]()
    suite^.run()
