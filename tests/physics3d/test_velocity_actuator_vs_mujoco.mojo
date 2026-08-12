"""`<velocity>` actuators against MuJoCo 3.10.0 — Phase 7 gap B.

MuJoCo expands `<velocity kv=K>` to gaintype=fixed / biastype=affine with
`gainprm = [K, 0, 0]` and `biasprm = [0, 0, -K]`, i.e.

    force = K * (ctrl - actuator_velocity)

⚠⚠ THE ONLY THING SEPARATING THIS FROM `<position>` IS `biasprm[1]`:
`-gainprm[0]` for a position servo, `0` for a velocity one. MuJoCo writes the
SAME gaintype and biastype for both, so a check that asks "is this a servo"
cannot tell them apart, and neither can a rollout started at qpos 0 — there the
length term is zero and both laws agree. Every fixture below is therefore
evaluated at a NON-ZERO qpos as well as a non-zero qvel, and
`no_length_term` exists purely to make the two laws disagree in SIGN.

This gates `apply_actions` against `d.qfrc_actuator` rather than rolling the
model forward: the actuator law is then the only thing under test, with no
integrator, no contact solver and no accumulated drift in the way.

⚠ What this does NOT cover: `forcerange`. It is unparsed everywhere in
physics3d (Phase 7 gap C) and no fixture here declares it, so these numbers say
nothing about force clamping. Jaco needs C before its arm can be gated.

Cases:
  1. `kv_explicit`    `<velocity kv=K>` on a hinge.
  2. `kv_default`     attribute-less `<velocity/>`. MuJoCo's kv default is 1,
                      NOT 0 (measured) — a 0 default would be a dead motor.
  3. `class_default`  `<default><velocity kv=.../></default>` reaching an
                      attribute-less element.
  4. `gear`           gear scales the transmission velocity AND the applied
                      torque, so it enters the force twice.
  5. `general_form`   the `<general>` spelling MuJoCo itself expands to. It was
                      REJECTED before this change (`bad_actuator` code 3
                      demanded `biasprm[1] == -gainprm[0]`). gainprm[0] and
                      -biasprm[2] are deliberately DIFFERENT here to prove they
                      are carried separately rather than collapsed.
  6. `no_length_term` the discriminator: a position reading of this actuator
                      drives the opposite sign.
  7. `mixed`          `<motor>`, `<position>` and `<velocity>` in one model, to
                      pin that adding the velocity tag to the scan did not
                      permute actuator indices.

Run: pixi run mojo run -I . tests/physics3d/test_velocity_actuator_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.fields import Data, Model

comptime DTYPE = DType.float64


# =============================================================================
# Fixtures
# =============================================================================

# Two hinges so the transmission has something to get wrong, and gravity off so
# `qfrc_actuator` is the actuator alone.
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

comptime XML_KV_EXPLICIT = (
    "<mujoco>"
    + _BODY
    + '<actuator><velocity ctrllimited="true" ctrlrange="-10 10" name="a1" joint="j1" kv="9.0"/>'
    + '<velocity ctrllimited="true" ctrlrange="-10 10" name="a2" joint="j2" kv="2.5"/></actuator></mujoco>'
)

comptime XML_KV_DEFAULT = (
    "<mujoco>"
    + _BODY
    + '<actuator><velocity ctrllimited="true" ctrlrange="-10 10" name="a1" joint="j1"/>'
    + '<velocity ctrllimited="true" ctrlrange="-10 10" name="a2" joint="j2"/></actuator></mujoco>'
)

comptime XML_CLASS_DEFAULT = (
    '<mujoco><default><velocity kv="7.0"/></default>'
    + _BODY
    + '<actuator><velocity ctrllimited="true" ctrlrange="-10 10" name="a1" joint="j1"/>'
    + '<velocity ctrllimited="true" ctrlrange="-10 10" name="a2" joint="j2" kv="3.0"/></actuator></mujoco>'
)

comptime XML_GEAR = (
    "<mujoco>"
    + _BODY
    + '<actuator><velocity ctrllimited="true" ctrlrange="-10 10" name="a1" joint="j1" kv="4.0" gear="2.5"/>'
    + '<velocity ctrllimited="true" ctrlrange="-10 10" name="a2" joint="j2" kv="4.0" gear="-1.5"/></actuator></mujoco>'
)

comptime XML_GENERAL_FORM = (
    "<mujoco>"
    + _BODY
    + '<actuator>'
    + '<general ctrllimited="true" ctrlrange="-10 10" name="a1" joint="j1" gaintype="fixed" biastype="affine"'
    + ' gainprm="6.0 0 0" biasprm="0 0 -9.0"/>'
    + '<general ctrllimited="true" ctrlrange="-10 10" name="a2" joint="j2" gaintype="fixed" biastype="affine"'
    + ' gainprm="2.0 0 0" biasprm="0 0 -0.5"/>'
    + '</actuator></mujoco>'
)

# ⚠ THE DISCRIMINATOR. `kv=30` with qpos driven to 1.2 rad below: a position
# reading gives `30*(ctrl - 1.2) - kv*vel`, a velocity reading `30*ctrl - 30*vel`.
# At ctrl = 1.0 those are -6 and +30 before the damping term — opposite signs,
# so this cannot pass by coincidence.
comptime XML_NO_LENGTH = (
    "<mujoco>"
    + _BODY
    + '<actuator><velocity ctrllimited="true" ctrlrange="-10 10" name="a1" joint="j1" kv="30.0"/>'
    + '<velocity ctrllimited="true" ctrlrange="-10 10" name="a2" joint="j2" kv="30.0"/></actuator></mujoco>'
)

# Index-order pin: three different tags interleaved. `<velocity>` was added to a
# scan that walks all tags in DOCUMENT order, and getting that wrong permutes
# `ctrl` rather than breaking anything visibly.
comptime XML_MIXED = (
    "<mujoco>"
    + _BODY
    + '<actuator>'
    + '<velocity ctrllimited="true" ctrlrange="-10 10" name="a1" joint="j1" kv="5.0"/>'
    + '<position ctrllimited="true" ctrlrange="-10 10" name="a2" joint="j2" kp="8.0" kv="0.3"/>'
    + '</actuator></mujoco>'
)

comptime XML_MIXED2 = (
    "<mujoco>"
    + _BODY
    + '<actuator>'
    + '<motor ctrllimited="true" ctrlrange="-10 10" name="a1" joint="j1" gear="3.0"/>'
    + '<velocity ctrllimited="true" ctrlrange="-10 10" name="a2" joint="j2" kv="5.0"/>'
    + '</actuator></mujoco>'
)


# =============================================================================
# `ModelDefFromXML[...]` is a comptime TYPE; the generic gate needs a VALUE.
# =============================================================================

comptime p_kve = parse_xml(XML_KV_EXPLICIT)
comptime p_kvd = parse_xml(XML_KV_DEFAULT)
comptime p_cls = parse_xml(XML_CLASS_DEFAULT)
comptime p_gear = parse_xml(XML_GEAR)
comptime p_gen = parse_xml(XML_GENERAL_FORM)
comptime p_nol = parse_xml(XML_NO_LENGTH)
comptime p_mix = parse_xml(XML_MIXED)
comptime p_mix2 = parse_xml(XML_MIXED2)


def _m_kve() -> ModelDefFromXML[
    xml=XML_KV_EXPLICIT, nbody=p_kve.NBODY, njoint=p_kve.NJOINT, nq=p_kve.NQ,
    nv=p_kve.NV, ngeom=p_kve.NGEOM, nact=p_kve.NACT, ntex=p_kve.NTEX,
    nmat=p_kve.NMAT, nlight=p_kve.NLIGHT, ncam=p_kve.NCAM, nsite=p_kve.NSITE,
    max_tendon=p_kve.NTENDON, cone_type=ConeType.PYRAMIDAL, max_contacts=8,
    max_condim=p_kve.MAX_CONDIM, nexclude=p_kve.NEXCLUDE, npair=p_kve.NPAIR,
    obs_dim_override=1, obs_qpos_skip=0, timestep=p_kve.TIMESTEP,
]:
    return {}


def _m_kvd() -> ModelDefFromXML[
    xml=XML_KV_DEFAULT, nbody=p_kvd.NBODY, njoint=p_kvd.NJOINT, nq=p_kvd.NQ,
    nv=p_kvd.NV, ngeom=p_kvd.NGEOM, nact=p_kvd.NACT, ntex=p_kvd.NTEX,
    nmat=p_kvd.NMAT, nlight=p_kvd.NLIGHT, ncam=p_kvd.NCAM, nsite=p_kvd.NSITE,
    max_tendon=p_kvd.NTENDON, cone_type=ConeType.PYRAMIDAL, max_contacts=8,
    max_condim=p_kvd.MAX_CONDIM, nexclude=p_kvd.NEXCLUDE, npair=p_kvd.NPAIR,
    obs_dim_override=1, obs_qpos_skip=0, timestep=p_kvd.TIMESTEP,
]:
    return {}


def _m_cls() -> ModelDefFromXML[
    xml=XML_CLASS_DEFAULT, nbody=p_cls.NBODY, njoint=p_cls.NJOINT, nq=p_cls.NQ,
    nv=p_cls.NV, ngeom=p_cls.NGEOM, nact=p_cls.NACT, ntex=p_cls.NTEX,
    nmat=p_cls.NMAT, nlight=p_cls.NLIGHT, ncam=p_cls.NCAM, nsite=p_cls.NSITE,
    max_tendon=p_cls.NTENDON, cone_type=ConeType.PYRAMIDAL, max_contacts=8,
    max_condim=p_cls.MAX_CONDIM, nexclude=p_cls.NEXCLUDE, npair=p_cls.NPAIR,
    obs_dim_override=1, obs_qpos_skip=0, timestep=p_cls.TIMESTEP,
]:
    return {}


def _m_gear() -> ModelDefFromXML[
    xml=XML_GEAR, nbody=p_gear.NBODY, njoint=p_gear.NJOINT, nq=p_gear.NQ,
    nv=p_gear.NV, ngeom=p_gear.NGEOM, nact=p_gear.NACT, ntex=p_gear.NTEX,
    nmat=p_gear.NMAT, nlight=p_gear.NLIGHT, ncam=p_gear.NCAM, nsite=p_gear.NSITE,
    max_tendon=p_gear.NTENDON, cone_type=ConeType.PYRAMIDAL, max_contacts=8,
    max_condim=p_gear.MAX_CONDIM, nexclude=p_gear.NEXCLUDE, npair=p_gear.NPAIR,
    obs_dim_override=1, obs_qpos_skip=0, timestep=p_gear.TIMESTEP,
]:
    return {}


def _m_gen() -> ModelDefFromXML[
    xml=XML_GENERAL_FORM, nbody=p_gen.NBODY, njoint=p_gen.NJOINT, nq=p_gen.NQ,
    nv=p_gen.NV, ngeom=p_gen.NGEOM, nact=p_gen.NACT, ntex=p_gen.NTEX,
    nmat=p_gen.NMAT, nlight=p_gen.NLIGHT, ncam=p_gen.NCAM, nsite=p_gen.NSITE,
    max_tendon=p_gen.NTENDON, cone_type=ConeType.PYRAMIDAL, max_contacts=8,
    max_condim=p_gen.MAX_CONDIM, nexclude=p_gen.NEXCLUDE, npair=p_gen.NPAIR,
    obs_dim_override=1, obs_qpos_skip=0, timestep=p_gen.TIMESTEP,
]:
    return {}


def _m_nol() -> ModelDefFromXML[
    xml=XML_NO_LENGTH, nbody=p_nol.NBODY, njoint=p_nol.NJOINT, nq=p_nol.NQ,
    nv=p_nol.NV, ngeom=p_nol.NGEOM, nact=p_nol.NACT, ntex=p_nol.NTEX,
    nmat=p_nol.NMAT, nlight=p_nol.NLIGHT, ncam=p_nol.NCAM, nsite=p_nol.NSITE,
    max_tendon=p_nol.NTENDON, cone_type=ConeType.PYRAMIDAL, max_contacts=8,
    max_condim=p_nol.MAX_CONDIM, nexclude=p_nol.NEXCLUDE, npair=p_nol.NPAIR,
    obs_dim_override=1, obs_qpos_skip=0, timestep=p_nol.TIMESTEP,
]:
    return {}


def _m_mix() -> ModelDefFromXML[
    xml=XML_MIXED, nbody=p_mix.NBODY, njoint=p_mix.NJOINT, nq=p_mix.NQ,
    nv=p_mix.NV, ngeom=p_mix.NGEOM, nact=p_mix.NACT, ntex=p_mix.NTEX,
    nmat=p_mix.NMAT, nlight=p_mix.NLIGHT, ncam=p_mix.NCAM, nsite=p_mix.NSITE,
    max_tendon=p_mix.NTENDON, cone_type=ConeType.PYRAMIDAL, max_contacts=8,
    max_condim=p_mix.MAX_CONDIM, nexclude=p_mix.NEXCLUDE, npair=p_mix.NPAIR,
    obs_dim_override=1, obs_qpos_skip=0, timestep=p_mix.TIMESTEP,
]:
    return {}


def _m_mix2() -> ModelDefFromXML[
    xml=XML_MIXED2, nbody=p_mix2.NBODY, njoint=p_mix2.NJOINT, nq=p_mix2.NQ,
    nv=p_mix2.NV, ngeom=p_mix2.NGEOM, nact=p_mix2.NACT, ntex=p_mix2.NTEX,
    nmat=p_mix2.NMAT, nlight=p_mix2.NLIGHT, ncam=p_mix2.NCAM,
    nsite=p_mix2.NSITE, max_tendon=p_mix2.NTENDON,
    cone_type=ConeType.PYRAMIDAL, max_contacts=8,
    max_condim=p_mix2.MAX_CONDIM, nexclude=p_mix2.NEXCLUDE, npair=p_mix2.NPAIR,
    obs_dim_override=1, obs_qpos_skip=0, timestep=p_mix2.TIMESTEP,
]:
    return {}


comptime M_KVE = _m_kve()
comptime M_KVD = _m_kvd()
comptime M_CLS = _m_cls()
comptime M_GEAR = _m_gear()
comptime M_GEN = _m_gen()
comptime M_NOL = _m_nol()
comptime M_MIX = _m_mix()
comptime M_MIX2 = _m_mix2()


# =============================================================================
# The gate
# =============================================================================


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
    """Diff our `qfrc` after `apply_actions` against MuJoCo `qfrc_actuator`.

    ⚠ `q0`/`q1` are non-zero on purpose. At qpos 0 the `<position>` and
    `<velocity>` laws produce the SAME force, so a fixture evaluated at the
    origin would pass with the length term wrongly included.
    """
    comptime Dat = Data[
        DTYPE, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1
    ]
    comptime Mod = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0, M.NPAIR,
    ]

    var ctx = DeviceContext()
    var mf = Mod()
    M.init_fields[DTYPE, 0](ctx, mf)
    var d = Dat()
    M.reset_data(d)

    d.qpos.data[0] = Scalar[DTYPE](q0)
    d.qpos.data[1] = Scalar[DTYPE](q1)
    d.qvel.data[0] = Scalar[DTYPE](v0)
    d.qvel.data[1] = Scalar[DTYPE](v1)

    var actions = List[Float64]()
    actions.append(c0)
    actions.append(c1)
    var act = List[Scalar[DTYPE]]()
    M.apply_actions[DTYPE](d, actions, act)

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
        var ours = Float64(d.qfrc.data[i])
        var theirs = Float64(py=md.qfrc_actuator[i])
        var diff = abs(ours - theirs)
        if diff > worst:
            worst = diff

    var ok = worst < 1e-12
    print(
        "  ",
        label,
        " qfrc ours[0]=",
        Float64(d.qfrc.data[0]),
        " mj[0]=",
        Float64(py=md.qfrc_actuator[0]),
        " worst|d|=",
        worst,
        " -> ",
        "PASS" if ok else "FAIL",
    )
    if not ok:
        failures += 1


def test_velocity_actuator_vs_mujoco() raises:
    print("=== <velocity> actuator vs MuJoCo 3.10.0 (qfrc_actuator) ===")
    var failures = 0

    # q != 0 everywhere — see the gate's docstring.
    _gate[M_KVE](
        "kv_explicit  ", XML_KV_EXPLICIT, 0.7, -0.4, 0.3, -0.9, 1.5, -0.6,
        failures,
    )
    _gate[M_KVD](
        "kv_default   ", XML_KV_DEFAULT, 0.7, -0.4, 0.3, -0.9, 1.5, -0.6,
        failures,
    )
    _gate[M_CLS](
        "class_default", XML_CLASS_DEFAULT, 0.7, -0.4, 0.3, -0.9, 1.5, -0.6,
        failures,
    )
    _gate[M_GEAR](
        "gear         ", XML_GEAR, 0.7, -0.4, 0.3, -0.9, 0.8, 0.5, failures
    )
    _gate[M_GEN](
        "general_form ", XML_GENERAL_FORM, 0.7, -0.4, 0.3, -0.9, 1.5, -0.6,
        failures,
    )
    # 1.2 rad of "position error" that a velocity servo must ignore.
    _gate[M_NOL](
        "no_length_term", XML_NO_LENGTH, 1.2, 1.2, 0.0, 0.0, 1.0, 1.0, failures
    )
    _gate[M_MIX](
        "mixed_pos_vel ", XML_MIXED, 0.7, -0.4, 0.3, -0.9, 1.5, -0.6, failures
    )
    _gate[M_MIX2](
        "mixed_mot_vel ", XML_MIXED2, 0.7, -0.4, 0.3, -0.9, 1.5, -0.6, failures
    )

    assert_true(
        failures == 0,
        String(failures) + " velocity-actuator case(s) failed",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
