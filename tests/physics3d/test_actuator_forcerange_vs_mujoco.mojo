"""Actuator `forcerange` / `forcelimited` against MuJoCo 3.10.0 — Phase 7 gap C.

Before this, `forcerange` and `forcelimited` appeared in ZERO files under
`mojo_rl/physics3d/`. Every ported model either omits them or never saturates,
so nothing had gone visibly wrong — but they are load-bearing on Jaco, whose
nine `<velocity kv="500">` actuators are paired with `forcerange="-30.5 30.5"`:
a 0.63 rad/s command against a stationary joint asks for ~315 N·m where MuJoCo
delivers 30.5.

⚠⚠ THE ONE THING TO GET RIGHT: **the clamp is on the SCALAR actuator force,
BEFORE the moment transform.** Measured on the runtime —

    <motor gear="3" forcerange="-1 1"/>  at ctrl = 5
        actuator_force  1        (gain*ctrl = 5, clamped)
        actuator_moment 3        (= gear)
        qfrc_actuator   3        (= moment * force)

Clamping the accumulated `qfrc` instead would cap this actuator at 1 N·m where
MuJoCo delivers 3. `motor_clamped_gear` below is exactly that discriminator:
`gear` is 3 and the limit binds, so the two placements differ by 3x. A fixture
with `gear="1"` would pass either way, which is why none of these use gear 1
where the clamp is active.

`forcelimited` semantics, all MEASURED (the default is "auto"):

    <motor/>                                    -> limited 0, range [0, 0]
    <motor forcerange="0 0"/>                   -> limited 0   ("0 0" IS the
                                                   undefined marker)
    <motor forcerange="-1 1"/>                  -> limited 1
    <motor forcerange="-1 1" forcelimited="false"/> -> limited 0
    <motor forcelimited="true"/>  (no range)    -> COMPILE ERROR in MuJoCo,
                                                   "invalid force range"

so a limited-but-zero-range actuator is unrepresentable and needs no handling.

Gated against `d.qfrc_actuator` after `mj_forward`, so the actuator law and its
clamp are the only things under test.

Run: pixi run mojo run -I . tests/physics3d/test_actuator_forcerange_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.fields import Data, Model, Dims

comptime DTYPE = DType.float64

comptime _BODY = """
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1" damping="0.1"/>
      <geom name="g1" type="capsule" fromto="0 0 0 0.4 0 0" size="0.04"/>
      <body name="link2" pos="0.4 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" damping="0.1"/>
        <geom name="g2" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
      </body>
    </body>
  </worldbody>
"""

# ⚠ gear != 1 on BOTH actuators: with gear 1 the clamp-before-moment and
# clamp-after-moment readings coincide and the fixture proves nothing.
comptime XML_CLAMP_GEAR = (
    "<mujoco>"
    + _BODY
    + '<actuator>'
    + '<motor ctrllimited="true" ctrlrange="-10 10" name="a1" joint="j1"'
    + ' gear="3" forcerange="-1 1"/>'
    + '<motor ctrllimited="true" ctrlrange="-10 10" name="a2" joint="j2"'
    + ' gear="-2" forcerange="-1 1"/>'
    + '</actuator></mujoco>'
)

# Same model, driven well inside the range: pins that we do not clamp when
# MuJoCo does not.
comptime XML_INSIDE = XML_CLAMP_GEAR

comptime XML_ASYM = (
    "<mujoco>"
    + _BODY
    + '<actuator>'
    + '<motor ctrllimited="true" ctrlrange="-10 10" name="a1" joint="j1"'
    + ' gear="2" forcerange="-0.5 4"/>'
    + '<motor ctrllimited="true" ctrlrange="-10 10" name="a2" joint="j2"'
    + ' gear="2" forcerange="-0.5 4"/>'
    + '</actuator></mujoco>'
)

comptime XML_CLASS_DEFAULT = (
    '<mujoco><default><motor forcerange="-2 2"/></default>'
    + _BODY
    + '<actuator>'
    + '<motor ctrllimited="true" ctrlrange="-10 10" name="a1" joint="j1"'
    + ' gear="3"/>'
    + '<motor ctrllimited="true" ctrlrange="-10 10" name="a2" joint="j2"'
    + ' gear="3" forcerange="-0.25 0.25"/>'
    + '</actuator></mujoco>'
)

# `forcerange` present but explicitly disabled — the limit must NOT apply.
comptime XML_LIMITED_FALSE = (
    "<mujoco>"
    + _BODY
    + '<actuator>'
    + '<motor ctrllimited="true" ctrlrange="-10 10" name="a1" joint="j1"'
    + ' gear="3" forcerange="-1 1" forcelimited="false"/>'
    + '<motor ctrllimited="true" ctrlrange="-10 10" name="a2" joint="j2"'
    + ' gear="3" forcerange="-1 1" forcelimited="false"/>'
    + '</actuator></mujoco>'
)

# No forcerange at all — the regression that matters most, because the stored
# default range is [0, 0] and treating THAT as a live limit would zero every
# actuator in every previously-ported model.
comptime XML_NO_RANGE = (
    "<mujoco>"
    + _BODY
    + '<actuator>'
    + '<motor ctrllimited="true" ctrlrange="-10 10" name="a1" joint="j1"'
    + ' gear="3"/>'
    + '<motor ctrllimited="true" ctrlrange="-10 10" name="a2" joint="j2"'
    + ' gear="3"/>'
    + '</actuator></mujoco>'
)

# ⚠ JACO'S ACTUAL SHAPE — a velocity servo whose gain alone would blow past the
# limit by an order of magnitude. This is the combination Phase 7 needs.
comptime XML_JACO_SHAPE = (
    "<mujoco>"
    + _BODY
    + '<actuator>'
    + '<velocity ctrllimited="true" ctrlrange="-0.62831853 0.62831853"'
    + ' forcelimited="true" forcerange="-30.5 30.5" name="a1" joint="j1"'
    + ' kv="500"/>'
    + '<velocity ctrllimited="true" ctrlrange="-5 5"'
    + ' forcelimited="true" forcerange="-1 1" name="a2" joint="j2" kv="10"/>'
    + '</actuator></mujoco>'
)

comptime p_cg = parse_xml(XML_CLAMP_GEAR)
comptime p_as = parse_xml(XML_ASYM)
comptime p_cd = parse_xml(XML_CLASS_DEFAULT)
comptime p_lf = parse_xml(XML_LIMITED_FALSE)
comptime p_nr = parse_xml(XML_NO_RANGE)
comptime p_jc = parse_xml(XML_JACO_SHAPE)


def _m_cg() -> ModelDefFromXML[
    xml=XML_CLAMP_GEAR, nbody=p_cg.NBODY, njoint=p_cg.NJOINT, nq=p_cg.NQ,
    nv=p_cg.NV, ngeom=p_cg.NGEOM, nact=p_cg.NACT, ntex=p_cg.NTEX,
    nmat=p_cg.NMAT, nlight=p_cg.NLIGHT, ncam=p_cg.NCAM, nsite=p_cg.NSITE,
    max_tendon=p_cg.NTENDON, cone_type=ConeType.PYRAMIDAL, max_contacts=8,
    max_condim=p_cg.MAX_CONDIM, nexclude=p_cg.NEXCLUDE, npair=p_cg.NPAIR,
    obs_dim_override=1, obs_qpos_skip=0, timestep=p_cg.TIMESTEP,
]:
    return {}


def _m_as() -> ModelDefFromXML[
    xml=XML_ASYM, nbody=p_as.NBODY, njoint=p_as.NJOINT, nq=p_as.NQ,
    nv=p_as.NV, ngeom=p_as.NGEOM, nact=p_as.NACT, ntex=p_as.NTEX,
    nmat=p_as.NMAT, nlight=p_as.NLIGHT, ncam=p_as.NCAM, nsite=p_as.NSITE,
    max_tendon=p_as.NTENDON, cone_type=ConeType.PYRAMIDAL, max_contacts=8,
    max_condim=p_as.MAX_CONDIM, nexclude=p_as.NEXCLUDE, npair=p_as.NPAIR,
    obs_dim_override=1, obs_qpos_skip=0, timestep=p_as.TIMESTEP,
]:
    return {}


def _m_cd() -> ModelDefFromXML[
    xml=XML_CLASS_DEFAULT, nbody=p_cd.NBODY, njoint=p_cd.NJOINT, nq=p_cd.NQ,
    nv=p_cd.NV, ngeom=p_cd.NGEOM, nact=p_cd.NACT, ntex=p_cd.NTEX,
    nmat=p_cd.NMAT, nlight=p_cd.NLIGHT, ncam=p_cd.NCAM, nsite=p_cd.NSITE,
    max_tendon=p_cd.NTENDON, cone_type=ConeType.PYRAMIDAL, max_contacts=8,
    max_condim=p_cd.MAX_CONDIM, nexclude=p_cd.NEXCLUDE, npair=p_cd.NPAIR,
    obs_dim_override=1, obs_qpos_skip=0, timestep=p_cd.TIMESTEP,
]:
    return {}


def _m_lf() -> ModelDefFromXML[
    xml=XML_LIMITED_FALSE, nbody=p_lf.NBODY, njoint=p_lf.NJOINT, nq=p_lf.NQ,
    nv=p_lf.NV, ngeom=p_lf.NGEOM, nact=p_lf.NACT, ntex=p_lf.NTEX,
    nmat=p_lf.NMAT, nlight=p_lf.NLIGHT, ncam=p_lf.NCAM, nsite=p_lf.NSITE,
    max_tendon=p_lf.NTENDON, cone_type=ConeType.PYRAMIDAL, max_contacts=8,
    max_condim=p_lf.MAX_CONDIM, nexclude=p_lf.NEXCLUDE, npair=p_lf.NPAIR,
    obs_dim_override=1, obs_qpos_skip=0, timestep=p_lf.TIMESTEP,
]:
    return {}


def _m_nr() -> ModelDefFromXML[
    xml=XML_NO_RANGE, nbody=p_nr.NBODY, njoint=p_nr.NJOINT, nq=p_nr.NQ,
    nv=p_nr.NV, ngeom=p_nr.NGEOM, nact=p_nr.NACT, ntex=p_nr.NTEX,
    nmat=p_nr.NMAT, nlight=p_nr.NLIGHT, ncam=p_nr.NCAM, nsite=p_nr.NSITE,
    max_tendon=p_nr.NTENDON, cone_type=ConeType.PYRAMIDAL, max_contacts=8,
    max_condim=p_nr.MAX_CONDIM, nexclude=p_nr.NEXCLUDE, npair=p_nr.NPAIR,
    obs_dim_override=1, obs_qpos_skip=0, timestep=p_nr.TIMESTEP,
]:
    return {}


def _m_jc() -> ModelDefFromXML[
    xml=XML_JACO_SHAPE, nbody=p_jc.NBODY, njoint=p_jc.NJOINT, nq=p_jc.NQ,
    nv=p_jc.NV, ngeom=p_jc.NGEOM, nact=p_jc.NACT, ntex=p_jc.NTEX,
    nmat=p_jc.NMAT, nlight=p_jc.NLIGHT, ncam=p_jc.NCAM, nsite=p_jc.NSITE,
    max_tendon=p_jc.NTENDON, cone_type=ConeType.PYRAMIDAL, max_contacts=8,
    max_condim=p_jc.MAX_CONDIM, nexclude=p_jc.NEXCLUDE, npair=p_jc.NPAIR,
    obs_dim_override=1, obs_qpos_skip=0, timestep=p_jc.TIMESTEP,
]:
    return {}


comptime M_CG = _m_cg()
comptime M_AS = _m_as()
comptime M_CD = _m_cd()
comptime M_LF = _m_lf()
comptime M_NR = _m_nr()
comptime M_JC = _m_jc()


def _gate[
    M: ModelDefFromXML
](
    label: String,
    xml: String,
    q0: Float64,
    q1: Float64,
    v0: Float64,
    v1: Float64,
    c0: Float64,
    c1: Float64,
    mut failures: Int,
) raises:
    """Diff our `qfrc` after `apply_actions` against MuJoCo `qfrc_actuator`."""
    var sf = M.make_spec_fields[DTYPE]()
    comptime Dat = Data[DTYPE, Dims[nq=M.NQ, nv=M.NV, nbody=M.NBODY, max_contacts=M.MAX_CONTACTS, nsite=M.NSITE], 1]
    comptime Mod = Model[DTYPE, Dims[nv=M.NV, nbody=M.NBODY, njoint=M.NJOINT, ngeom=M.NGEOM, nequality=M.MAX_EQUALITY, ntendon=M.MAX_TENDON, nsite=M.NSITE, nexclude=M.NEXCLUDE, nmesh_verts=0, npair=M.NPAIR]]

    var ctx = DeviceContext()
    var mf = Mod()
    M.init_fields[DTYPE, 0](ctx, mf)
    var d = Dat()
    M.reset_data(sf, d)
    d.qpos.data[0] = Scalar[DTYPE](q0)
    d.qpos.data[1] = Scalar[DTYPE](q1)
    d.qvel.data[0] = Scalar[DTYPE](v0)
    d.qvel.data[1] = Scalar[DTYPE](v1)

    var actions = List[Float64]()
    actions.append(c0)
    actions.append(c1)
    var act = List[Scalar[DTYPE]]()
    M.apply_actions[DTYPE](sf, d, actions, act)

    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(xml)
    var md = mujoco.MjData(m)
    md.qpos[0] = q0
    md.qpos[1] = q1
    md.qvel[0] = v0
    md.qvel[1] = v1
    md.ctrl[0] = c0
    md.ctrl[1] = c1
    _ = mujoco.mj_forward(m, md)

    var worst = Float64(0)
    for i in range(M.NV):
        var diff = abs(
            Float64(d.qfrc.data[i]) - Float64(py=md.qfrc_actuator[i])
        )
        if diff > worst:
            worst = diff

    var ok = worst < 1e-12
    print(
        "  ",
        label,
        " ours[0]=",
        Float64(d.qfrc.data[0]),
        " mj[0]=",
        Float64(py=md.qfrc_actuator[0]),
        " mj_force[0]=",
        Float64(py=md.actuator_force[0]),
        " worst|d|=",
        worst,
        " -> ",
        "PASS" if ok else "FAIL",
    )
    if not ok:
        failures += 1


def test_actuator_forcerange_vs_mujoco() raises:
    print("=== actuator forcerange vs MuJoCo 3.10.0 (qfrc_actuator) ===")
    var failures = 0

    # ⚠ THE DISCRIMINATOR: ctrl 5 with gear 3 and range [-1, 1]. Clamping the
    # force gives qfrc 3; clamping qfrc would give 1.
    _gate[M_CG](
        "clamp_gear_hi ", XML_CLAMP_GEAR, 0.7, -0.4, 0.3, -0.9, 5.0, 5.0,
        failures,
    )
    # Same fixture, negative — pins the lower bound too.
    _gate[M_CG](
        "clamp_gear_lo ", XML_CLAMP_GEAR, 0.7, -0.4, 0.3, -0.9, -5.0, -5.0,
        failures,
    )
    # Inside the range: must NOT clamp.
    _gate[M_CG](
        "inside_range  ", XML_INSIDE, 0.7, -0.4, 0.3, -0.9, 0.4, -0.3,
        failures,
    )
    # Asymmetric [-0.5, 4]: one actuator saturates high, the other low.
    _gate[M_AS](
        "asymmetric    ", XML_ASYM, 0.7, -0.4, 0.3, -0.9, 9.0, -9.0, failures
    )
    # Class default reaches a1; a2 overrides it with a tighter range.
    _gate[M_CD](
        "class_default ", XML_CLASS_DEFAULT, 0.7, -0.4, 0.3, -0.9, 6.0, 6.0,
        failures,
    )
    # forcerange present, forcelimited="false": no clamp.
    _gate[M_LF](
        "limited_false ", XML_LIMITED_FALSE, 0.7, -0.4, 0.3, -0.9, 5.0, 5.0,
        failures,
    )
    # ⚠ No forcerange at all. The stored default is [0, 0]; treating that as a
    # live limit would zero every actuator in every model ported so far.
    _gate[M_NR](
        "no_forcerange ", XML_NO_RANGE, 0.7, -0.4, 0.3, -0.9, 5.0, -5.0,
        failures,
    )
    # Jaco's shape: kv=500 over a 0.63 rad/s range against forcerange 30.5.
    _gate[M_JC](
        "jaco_shape    ", XML_JACO_SHAPE, 0.7, -0.4, 0.0, 0.0, 0.62, 4.0,
        failures,
    )
    # ⚠ Same model, joint already moving so the velocity term pulls the force
    # back INSIDE the limit: 500*(0.62 - 0.58) = 20 < 30.5 and
    # 10*(4.0 - 3.95) = 0.5 < 1. Saturation is STATE-dependent, so a fixture
    # that only ever saturates would pass with the clamp applied
    # unconditionally.
    _gate[M_JC](
        "jaco_unsat    ", XML_JACO_SHAPE, 0.7, -0.4, 0.58, 3.95, 0.62, 4.0,
        failures,
    )

    assert_true(
        failures == 0, String(failures) + " forcerange case(s) failed"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
