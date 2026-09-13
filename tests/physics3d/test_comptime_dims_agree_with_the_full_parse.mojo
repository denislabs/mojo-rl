"""The two MJCF paths must agree on NQ/NV and on the ACTUATOR COUNT.

    pixi run mojo run -I . tests/physics3d/test_comptime_dims_agree_with_the_full_parse.mojo

AUD-19. This engine has two MJCF readers. `parse_xml` runs at comptime and
supplies the DIMENSIONS a `ModelDefFromXML` is instantiated with; `full_parser`
runs at load and produces the actual records. `init_fields` already cross-checks
nbody, njoint, ngeom, na and nkey, and raises naming the element type that
disagreed.

⚠ THE ACTUATOR HALF (2026-09-13) IS A FIX, NOT A GUARD, AND IT CARRIES ONE.
`parse_xml` counted `motor`/`position`/`velocity`/`general` where
`full_parser` also builds an actuator for `<adhesion>` and for a `<plugin>`
actuator, so `NACT` came out SHORT: the control vector was shorter than
MuJoCo's `nu` and every actuator past the uncounted one read the previous
one's control. `nact` was also the ONE dimension `init_fields` did not
cross-check, so nothing said so. Both tags are counted now, and the count is
checked — the two parsers are separate code and will drift again.

NQ and NV were not checked, and they are the pair that can differ while every
count above agrees. `parse_xml` reads a joint's `type` off the ELEMENT ONLY
(`xml_parser.mojo` `_count_joints_with_type` scans each `<joint …>` tag for a
literal `type="ball"`), so a `type` stated in a `<default>` class is invisible
to it: the joint is counted as a 1-dof hinge there and built as a 4/3 ball or
a 7/6 free joint here.

⚠⚠ THAT IS THE QUIETEST WAY TO GET A WRONG MODEL. njoint matches. nbody
matches. ngeom matches. Nothing is out of range. Every `qpos` index past the
mis-typed joint is simply shifted, so the model loads, steps, and is a
different model — and the only visible symptom is a parity residual in a
50-dimensional vector.

⚠ THIS GATE PINS THE GUARD, NOT A FIX. The right repair is for the comptime
scan to resolve a joint's class chain, which means a second class resolver in
a parser that cannot call the first one. Until then the disagreement RAISES
with a message naming the cause. Both are worth having: the raise is what
stops a silently shifted model reaching a rollout.

SURVEYED: no model in `mojo_rl/envs`, Menagerie or dm_control states a ball or
free joint `type` in a `<default>` class, so nothing in the tree is affected —
which is also why the hole stayed open.
"""

from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.fields import Model, Dims

comptime DTYPE = DType.float64

# ⚠ THE BALL `type` IS IN THE CLASS, NOT ON THE ELEMENT. That one placement is
# the whole fixture: `parse_xml` sees `<joint name="j1" class="spin"/>` and
# counts a hinge (nq 1, nv 1); `full_parser` resolves the class and builds a
# ball (nq 4, nv 3). Both agree njoint == 2.
comptime CLASS_BALL_XML = String(
    """<mujoco model="class_ball">
  <compiler angle="radian"/>
  <default>
    <default class="spin">
      <joint type="ball"/>
    </default>
  </default>
  <worldbody>
    <body name="b0" pos="0 0 1">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
      <body name="b1" pos="0.2 0 0">
        <joint name="j1" class="spin"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
)

# The same model with the type stated on the ELEMENT — the control. Both
# readers see the ball, the dimensions agree, and the build must succeed.
comptime ELEMENT_BALL_XML = String(
    """<mujoco model="element_ball">
  <compiler angle="radian"/>
  <worldbody>
    <body name="b0" pos="0 0 1">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
      <body name="b1" pos="0.2 0 0">
        <joint name="j1" type="ball"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" density="1000"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
)

comptime CP_CLASS = parse_xml(CLASS_BALL_XML)
comptime CP_ELEM = parse_xml(ELEMENT_BALL_XML)

# ⚠ AN INSTANCE, NOT A TYPE ALIAS. `_build` takes the model def as a comptime
# VALUE parameter, which is how every other gate in this directory passes one;
# a bare `ModelDefFromXML[...]` alias cannot have its parameters inferred at
# the call site.
def _class_ball() -> ModelDefFromXML[
    xml=CLASS_BALL_XML,
    nbody=CP_CLASS.NBODY, njoint=CP_CLASS.NJOINT,
    nq=CP_CLASS.NQ, nv=CP_CLASS.NV,
    ngeom=CP_CLASS.NGEOM, nact=CP_CLASS.NACT, ntex=CP_CLASS.NTEX,
    nmat=CP_CLASS.NMAT, nlight=CP_CLASS.NLIGHT, ncam=CP_CLASS.NCAM,
    nsite=CP_CLASS.NSITE,
    cone_type=ConeType.PYRAMIDAL, max_contacts=4,
    obs_dim_override=4, obs_qpos_skip=0, timestep=CP_CLASS.TIMESTEP,
]:
    return {}


def _element_ball() -> ModelDefFromXML[
    xml=ELEMENT_BALL_XML,
    nbody=CP_ELEM.NBODY, njoint=CP_ELEM.NJOINT,
    nq=CP_ELEM.NQ, nv=CP_ELEM.NV,
    ngeom=CP_ELEM.NGEOM, nact=CP_ELEM.NACT, ntex=CP_ELEM.NTEX,
    nmat=CP_ELEM.NMAT, nlight=CP_ELEM.NLIGHT, ncam=CP_ELEM.NCAM,
    nsite=CP_ELEM.NSITE,
    cone_type=ConeType.PYRAMIDAL, max_contacts=4,
    obs_dim_override=4, obs_qpos_skip=0, timestep=CP_ELEM.TIMESTEP,
]:
    return {}


comptime ClassBallModel = _class_ball()
comptime ElementBallModel = _element_ball()


def _build[M: ModelDefFromXML]() raises:
    comptime MD = Dims[
        nq=M.NQ,
        nv=M.NV,
        nbody=M.NBODY,
        njoint=M.NJOINT,
        ngeom=M.NGEOM,
        nsite=M.NSITE,
        max_contacts=M.MAX_CONTACTS,
        nequality=M.MAX_EQUALITY,
        ntendon=M.MAX_TENDON,
        nexclude=M.NEXCLUDE,
        nmesh_verts=0,
        npair=M.NPAIR,
        nact=M.NACT,
        nten=M.NTEN_F,
        nkey=M.NKEY,
    ]
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    M.init_fields[DTYPE](ctx, mf)


# ⚠ `<adhesion>` AND A `<plugin>` ACTUATOR TOGETHER: the two elements the
# comptime scan used to miss. MuJoCo gives this model `nu == 3`.
comptime MIXED_ACT_XML = String(
    """<mujoco model="mixed_actuators">
  <compiler angle="radian"/>
  <extension>
    <plugin plugin="mujoco.pid">
      <instance name="pid0">
        <config key="kp" value="10"/>
        <config key="ki" value="1"/>
        <config key="kd" value="0.1"/>
      </instance>
    </plugin>
  </extension>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="b0" pos="0 0 1">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom name="g0" type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      <body name="b1" pos="0.2 0 0">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom name="g1" type="box" size="0.03 0.03 0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m" joint="j0" gear="2"/>
    <plugin name="p" joint="j1" plugin="mujoco.pid" instance="pid0"/>
    <adhesion name="a" body="b1" ctrlrange="0 1" gain="5"/>
  </actuator>
</mujoco>
"""
)

comptime CP_MIXED = parse_xml(MIXED_ACT_XML)


def test_the_comptime_scan_counts_every_actuator_element() raises:
    """AUD-19's actuator half. ⚠ THE NAMED WRONG ANSWER IS 1.

    Before the fix `parse_xml` counted `<motor>` alone on this model — the
    `<plugin>` and `<adhesion>` elements were invisible to it — so `NACT` was
    1 against `full_parser`'s 3 and MuJoCo's `nu` of 3. Nothing compared the
    two, because `nact` was the one dimension `init_fields` did not check.
    """
    print("=== parse_xml counts <adhesion> and <plugin> actuators ===")
    var fmd = parse_xml_full(MIXED_ACT_XML, String("."))
    print("  parse_xml NACT =", CP_MIXED.NACT,
          "  full_parser actuators =", len(fmd.actuators))
    assert_true(
        len(fmd.actuators) == 3,
        "the fixture no longer builds three actuators (full_parser found "
        + String(len(fmd.actuators)) + ") — it is not testing what it claims",
    )
    assert_true(
        CP_MIXED.NACT == len(fmd.actuators),
        "parse_xml counts " + String(CP_MIXED.NACT) + " actuators where"
        " full_parser builds " + String(len(fmd.actuators))
        + ". A short NACT sizes the control vector short and shifts every"
        " actuator past the uncounted one",
    )
    assert_true(
        CP_MIXED.NACT != 1,
        "parse_xml counts 1 — the <motor> only. That is exactly the value"
        " the AUD-19 defect produced",
    )


def test_the_two_readers_really_do_disagree() raises:
    """⚠⚠ THE PREMISE, BEFORE ANY GUARD. If `parse_xml` ever learns to
    resolve a joint's class, this fixture stops testing the guard and starts
    testing nothing — so the disagreement itself is asserted first.
    """
    print("=== the premise: parse_xml vs full_parser on the same xml ===")
    var fmd = parse_xml_full(CLASS_BALL_XML, String("."))
    var nq = 0
    var nv = 0
    for i in range(len(fmd.joints)):
        nq += fmd.joints[i].nq
        nv += fmd.joints[i].nv
    print("  parse_xml   njoint", CP_CLASS.NJOINT, " nq", CP_CLASS.NQ,
          " nv", CP_CLASS.NV)
    print("  full_parser njoint", len(fmd.joints), " nq", nq, " nv", nv)
    assert_true(
        len(fmd.joints) == CP_CLASS.NJOINT,
        "the two readers disagree on the JOINT COUNT too, so the existing"
        " njoint check would already have caught this and the NQ/NV guard is"
        " not what this file is testing",
    )
    assert_true(
        nq != CP_CLASS.NQ or nv != CP_CLASS.NV,
        "the two readers now AGREE on nq/nv for a class-level ball joint —"
        " `parse_xml` has learned to resolve the class chain, which is the"
        " real fix. Delete this fixture and gate the fix instead.",
    )
    # Name the numbers so a future reader does not have to re-derive them.
    assert_true(
        nq == 5 and nv == 4 and CP_CLASS.NQ == 2 and CP_CLASS.NV == 2,
        "expected full_parser (hinge + ball) = nq 5 / nv 4 and parse_xml"
        " (hinge + hinge) = nq 2 / nv 2, got " + String(nq) + "/" + String(nv)
        + " and " + String(CP_CLASS.NQ) + "/" + String(CP_CLASS.NV),
    )


def test_a_shifted_model_raises_instead_of_loading() raises:
    """The guard: `init_fields` must refuse, not build a shifted model."""
    print("=== a class-level ball joint refuses to build ===")
    var raised = False
    var msg = String("")
    try:
        _build[ClassBallModel]()
    except e:
        raised = True
        msg = String(e)
    print("  raised:", raised)
    if raised:
        print("  message:", msg)
    assert_true(
        raised,
        "`init_fields` built a model whose comptime nq/nv (2/2) disagree with"
        " its own records (5/4). Every qpos index past the mis-typed joint is"
        " shifted and nothing said so.",
    )
    # ⚠ THE MESSAGE HAS TO NAME THE CAUSE, or the next person to hit this
    # spends an afternoon on a dimension they did not choose.
    assert_true(
        msg.find("NQ/NV") >= 0,
        "the error does not mention NQ/NV: " + msg,
    )


def test_the_same_model_spelled_on_the_element_still_builds() raises:
    """⚠ THE OVER-FIX CONTROL. A guard that refused every ball joint would
    pass the test above. This is the identical model with `type="ball"` on the
    element, where both readers see 5/4 and the build must succeed.
    """
    print("=== the same ball joint, stated on the element, builds ===")
    print("  parse_xml nq", CP_ELEM.NQ, " nv", CP_ELEM.NV, " (want 5 / 4)")
    assert_true(
        CP_ELEM.NQ == 5 and CP_ELEM.NV == 4,
        "the control fixture's comptime dims are " + String(CP_ELEM.NQ) + "/"
        + String(CP_ELEM.NV) + ", not 5/4 — it is not the same model",
    )
    _build[ElementBallModel]()
    print("  built: ok")


def main() raises:
    var suite = TestSuite()
    suite.test[test_the_comptime_scan_counts_every_actuator_element]()
    suite.test[test_the_two_readers_really_do_disagree]()
    suite.test[test_a_shifted_model_raises_instead_of_loading]()
    suite.test[test_the_same_model_spelled_on_the_element_still_builds]()
    suite^.run()
