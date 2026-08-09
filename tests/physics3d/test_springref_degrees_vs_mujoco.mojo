"""`<joint springref>` in a DEGREE model, against MuJoCo.

MJCF's default angle unit is DEGREE, and MuJoCo converts `ref` and `springref`
to radians for HINGE joints only (`user_objects.cc:3276` — byte-identical in
3.3.6, 3.6.0 and main, so there is no version-drift question here). Our parser
converted `range` and `ref` and silently did not convert `springref`, which
sits on the line between them.

WHAT THAT COSTS. dm_control's dog spells `springref="-11.0"` with
`stiffness="2.0"` on the jaw. Unconverted, the mandible spring pulls towards
-11 RADIANS instead of -0.192 — a rest position 56 revolutions away — and the
passive torque that produces dominates the whole solve. Measured against
MuJoCo's `qpos_spring`, max|d| over dog's 74 joints was 10.808 rad, exactly
`|-11 - (-0.191986)|`.

⚠ THE SLIDE JOINT IS NOT DECORATION. `springref` on a SLIDE joint is a LENGTH
and must pass through unscaled; a fix that scaled every joint would pass a
hinge-only test and quietly shrink every prismatic spring by 57x. The model
below carries one of each and the test asserts both.

⚠ NON-VACUITY: the fixture asserts DEGREE MODE IS ACTUALLY IN PLAY, by
requiring the raw attribute and MuJoCo's compiled `qpos_spring` to differ. In
a radian model they would agree and the whole file would pass without
exercising the conversion at all.

Run with:
    pixi run mojo run -I . tests/physics3d/test_springref_degrees_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.fields import Model, Data
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_SPRINGREF,
)
from max.gpu.host import DeviceContext


comptime DTYPE = DType.float64

# No `<compiler angle=...>`: MJCF's default IS degree, which is the case that
# actually ships. The hinge's springref is dog's own number.
comptime SR_XML = """
<mujoco model="springref">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="swing" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0"
             stiffness="2.0" springref="-11.0" damping="0.02"
             range="-90 90"/>
      <geom name="rod" type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"
            contype="0" conaffinity="0"/>
      <body name="slider" pos="0.3 0 0">
        <joint name="slide" type="slide" axis="1 0 0"
               stiffness="5.0" springref="0.12" damping="0.02"
               range="-0.5 0.5"/>
        <geom name="block" type="box" size="0.03 0.03 0.03"
              contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

comptime pp = parse_xml(SR_XML)
comptime M = ModelDefFromXML[
    xml=SR_XML,
    nbody=pp.NBODY, njoint=pp.NJOINT, nq=pp.NQ, nv=pp.NV,
    ngeom=pp.NGEOM, nact=pp.NACT, ntex=pp.NTEX, nmat=pp.NMAT,
    nlight=pp.NLIGHT, ncam=pp.NCAM, nsite=pp.NSITE,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=4,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep=pp.TIMESTEP,
    max_condim=pp.MAX_CONDIM,
    noslip_iter=pp.NOSLIP_ITER,
]

comptime N_STEPS: Int = 400
# Gravity is off and both sides run identical float64 arithmetic from an
# identical state, so this budgets round-off over 400 steps of a spring
# oscillation. Measured, then budgeted three orders above.
comptime TOL: Float64 = 1e-12


def _mj() raises -> PythonObject:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[SR_XML]())
    return Python.tuple(mujoco, m, mujoco.MjData(m))


def test_springref_is_converted_for_hinge_only() raises:
    """Our `springref` against MuJoCo's compiled `qpos_spring`."""
    print("--- springref: model constants ---")
    var h = _mj()
    var m = h[1]

    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DTYPE, 0](ctx, mf)

    var worst = 0.0
    for j in range(M.NJOINT):
        var o = j * MODEL_JOINT_SIZE
        var adr = Int(Float64(mf.joints.data[o + JOINT_IDX_QPOS_ADR]))
        var ours = Float64(mf.joints.data[o + JOINT_IDX_SPRINGREF])
        var mj = Float64(py=m.qpos_spring[adr])
        print("  joint", j, " ours", ours, " mj", mj)
        var e = abs(ours - mj)
        if e > worst:
            worst = e
    print("  max|d| =", worst)

    # NON-VACUITY. If the model were in radians the raw attribute and the
    # compiled value would agree and this file would gate nothing.
    var hinge_mj = Float64(py=m.qpos_spring[0])
    assert_true(
        abs(hinge_mj - (-11.0)) > 1.0,
        "MuJoCo compiled the hinge springref as -11 — the model is in RADIAN"
        " mode, so the degree conversion is never exercised and every"
        " assertion below is vacuous",
    )
    # And the slide must NOT have been scaled, or a blanket conversion passes.
    var slide_mj = Float64(py=m.qpos_spring[1])
    assert_true(
        abs(slide_mj - 0.12) < 1e-15,
        "MuJoCo scaled the SLIDE springref — the hinge-only rule this test"
        " asserts is not MuJoCo's, so the expectation is wrong, not the code",
    )

    assert_true(
        worst < 1e-15,
        "springref does not match MuJoCo's qpos_spring — a degree model's"
        " hinge springref must be converted to radians and a slide's must not",
    )


def test_springref_rollout_matches_mujoco() raises:
    """The spring has to actually drive the joint, not just tabulate."""
    print("--- springref: rollout ---")
    var h = _mj()
    var mujoco = h[0]
    var m = h[1]
    var md = h[2]
    mujoco.mj_resetData(m, md)

    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DTYPE, 0](ctx, mf)
    var d = Data[DTYPE, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1]()
    M.reset_data[DTYPE](d)
    forward_kinematics["cpu"](d, mf)

    var integ = EulerIntegrator[
        DTYPE, M.NQ, M.NV, M.NBODY, M.NJOINT, M.MAX_CONTACTS, M.NGEOM,
        M.MAX_EQUALITY, M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
        M.CONE_TYPE, 1, SOLVER="newton",
        MAX_CONDIM=M.MAX_CONDIM, NOSLIP_ITER=M.NOSLIP_ITER,
    ]()

    var worst_q = 0.0
    var worst_v = 0.0
    var travel = 0.0
    for _s in range(N_STEPS):
        for i in range(M.NV):
            d.qfrc.data[i] = Scalar[DTYPE](0)
        integ.step["cpu"](d, mf)
        mujoco.mj_step(m, md)
        var mq = md.qpos.flatten().tolist()
        var mv = md.qvel.flatten().tolist()
        for i in range(M.NQ):
            var e = abs(Float64(d.qpos.data[i]) - Float64(py=mq[i]))
            if e > worst_q:
                worst_q = e
            if abs(Float64(py=mq[i])) > travel:
                travel = abs(Float64(py=mq[i]))
        for i in range(M.NV):
            var e = abs(Float64(d.qvel.data[i]) - Float64(py=mv[i]))
            if e > worst_v:
                worst_v = e

    print("  max |qpos| reached =", travel)
    print("  worst |d(qpos)| =", worst_q, "  worst |d(qvel)| =", worst_v)

    # NON-VACUITY: gravity is off, so the ONLY thing that can move either
    # joint is its spring. A motionless rollout would post a perfect number.
    assert_true(
        travel > 0.05,
        "nothing moved — with gravity off the springs are the only driver, so"
        " a static rollout means springref never entered the dynamics",
    )
    assert_true(worst_q <= TOL, "qpos diverged under a springref spring")
    assert_true(worst_v <= TOL, "qvel diverged under a springref spring")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
