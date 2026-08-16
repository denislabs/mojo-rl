"""`mj_rnePostConstraint` + the acceleration-stage site sensors vs LIVE MuJoCo.

Gates `dynamics/rne_post.mojo` and `sensors/site_acc.mojo` on the dm_control
quadruped — the first model in the tree that declares `accelerometer`,
`force` and `torque` sensors. Both engines are fed the SAME merged XML, so
this is a comparison of arithmetic, not of two model builds.

WHERE THE VALUES COME FROM ON OUR SIDE. The stage runs inside
`EulerIntegrator.step` between the constraint solve and the integration —
MuJoCo's `mj_sensorAcc` point. So after one `step()` call, `d.cacc` /
`d.cfrc_int` (and the FK products, which the step does not refresh) all
describe the PRE-integration state. That is the state `mj_forward` is
evaluated at here, which is why nothing needs to be stepped on the MuJoCo
side at all.

THREE LAYERS, deliberately separated:

  1. STRUCTURAL — our body and site ORDER equals MuJoCo's for this XML, so
     `mj_name2id` indices are usable on our records. Nothing else in this
     file is meaningful if this fails, and the failure mode is a silently
     mismatched sensor rather than an error.
  2. FREE FLIGHT — no contacts, no active limits, so `cfrc_ext == 0` and
     `qacc` agrees with MuJoCo to machine precision. This isolates the
     cacc recursion and the cfrc_int backward pass from the contact solver.
  3. STANDING — four toes on the floor. Now `cfrc_ext` carries real contact
     forces, which is the only way to catch a sign error in the mapping (the
     one pre-existing consumer, Ant's contact_cost, takes a NORM and is
     blind to it). Tolerance here is the contact solver's, not the sensor's.

Run: pixi run mojo run -I . tests/physics3d/test_rne_post_sensors_vs_mujoco.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python, PythonObject
from std.math import abs, sqrt
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.fields import Data, Model, Dims
from mojo_rl.physics3d.sensors import (
    site_accelerometer,
    site_force_torque,
    site_frame_velocity,
)
from mojo_rl.envs.dm_control.quadruped.quadruped_xml import (
    DMQuadrupedWalkModel,
)

comptime DTYPE = DType.float64
comptime Mdl = DMQuadrupedWalkModel

comptime NQ = Mdl.NQ
comptime NV = Mdl.NV
comptime NBODY = Mdl.NBODY
comptime NJOINT = Mdl.NJOINT
comptime NGEOM = Mdl.NGEOM
comptime NSITE = Mdl.NSITE
comptime MAX_CONTACTS = Mdl.MAX_CONTACTS
comptime NEQ = Mdl.MAX_EQUALITY
comptime NTEN = Mdl.MAX_TENDON
comptime NEXCL = Mdl.NEXCLUDE
comptime NA = Mdl.NA

comptime Integ = EulerIntegrator[
    DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, NEQ, NTEN, NSITE,
    NEXCL, 0, Mdl.CONE_TYPE, 1, SOLVER="newton", RNE_POST=True,
]
comptime Dat = Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1]
comptime Mod = Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQ, ntendon=NTEN, nsite=NSITE, nexclude=NEXCL, nmesh_verts=0]]

# ⚠ ALL FOUR BOUNDS WERE RE-PINNED 2026-08-03, two to three orders tighter,
# and the reason is worth reading before loosening any of them again.
#
# They used to sit at 1e-8 / 1e-9 / 1e-9 / 5e-10 around a residual of ~1e-10
# that FK_TOL's own comment called "worth its own investigation". It was:
# `kinematics/quat_math.mojo` normalized quaternions as
# `1/sqrt(norm_sq + 1e-10)` — a divide-by-zero guard INSIDE the sqrt — which
# returns 0.99999999995 for an already-unit quaternion, so every body
# quaternion came out 5e-11 short of unit and every vector rotated by one was
# scaled by 1 - 1e-10. Every quantity below inherited it, which is exactly why
# the gap "showed up identically in cvel, qfrc_bias and tendon_invweight0".
#
# With that fixed the whole file agrees with MuJoCo at float64 rounding. A
# tolerance left at the old value would now pass while testing nothing — which
# is the failure mode that let the bug survive under six quadruped gates.
#
# Free flight has no solver in the loop: the only error is float rounding
# through two independent implementations of the same recursion. Observed
# 1.00e-14 (was ~1e-10).
comptime FLIGHT_TOL: Float64 = 1e-13
# Standing rides on the contact solve. Observed 3.55e-15; the total vertical
# toe force matches MuJoCo to 9.99e-16 (was 1.4e-11).
comptime STAND_REL_TOL: Float64 = 1e-13
# Worst single force/torque component, standing. Observed 4.07e-15. This was
# 0.221 until the tangential direction was fixed — `cfrc_ext` was reading the
# contact record's FRAME_T1 HINT as if it were the tangent, so the horizontal
# force landed along an arbitrary axis while the vertical one (which needs
# only the normal) stayed exact. See collision/contact_frame.mojo. It was then
# 5.06e-11 until the quaternion normalizer above.
comptime STAND_COMPONENT_TOL: Float64 = 1e-13
# Forward kinematics itself. Observed 4.44e-16 on both `xpos` and `site_xpos`
# — i.e. float64 rounding, which is what it should always have been.
comptime FK_TOL: Float64 = 1e-14

def _toe_names() -> List[String]:
    return [
        String("toe_front_left"),
        String("toe_front_right"),
        String("toe_back_right"),
        String("toe_back_left"),
    ]


def _sensor_slice(mujoco: PythonObject, m: PythonObject, name: String,
                  dat: PythonObject) raises -> List[Float64]:
    """`sensordata` for one named sensor, as a plain list."""
    var sid = Int(py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name))
    assert_true(sid >= 0, "no such sensor: " + name)
    var adr = Int(py=m.sensor_adr[sid])
    var dim = Int(py=m.sensor_dim[sid])
    var out = List[Float64]()
    for k in range(dim):
        out.append(Float64(py=dat.sensordata[adr + k]))
    return out^


# quadruped's joint layout, pinned by `test_body_and_site_order_match_mujoco`
# (which proves our numbering equals MuJoCo's) and by the tendon probe: joint 0
# is the free root, then four legs of four hinges each — yaw, then the pitch /
# knee / ankle triple that leg L's `coupling_L` tendon ties together. A hinge
# joint j therefore owns qpos[6+j] and dof 5+j.
comptime N_LEGS: Int = 4


def _setup(
    mut d: Dat, mut mf: Mod, ctx: DeviceContext, z: Float64, tilt: Bool,
    bend: Float64, torso_vel: Float64, hinge_vel: Float64,
) raises -> List[Float64]:
    """Reference pose moved to `z`, optionally tilted, with a per-leg bend and
    motion that leaves EVERY coupling tendon exactly satisfied in both length
    and length-rate.

    That last part is deliberate. The four `<equality><tendon>` rows are
    always active, and driving them far off their setpoint (all three hinges
    bent the same way) puts the model in a stiff regime — 6 cm of violation
    against solref `.005 .5` asks for ~1e4 rad/s^2 — where the gate stops
    measuring `rne_post` and starts measuring how two different solvers cope
    with a near-singular row. Bending pitch and knee OPPOSITE ways keeps
    `.333*(pitch + knee + ankle)` at zero, so the constraint only has to
    cancel the differential Coriolis and servo accelerations, which is both
    well conditioned and what the real task actually does.
    """
    var sf = Mdl.make_spec_fields[DTYPE]()
    Mdl.init_fields[DTYPE, 0](ctx, mf)
    Mdl.reset_data(sf, d)

    d.qpos.data[2] = Scalar[DTYPE](z)
    # w first, matching the free-joint qpos layout both engines use.
    d.qpos.data[3] = Scalar[DTYPE](0.9762960071199334 if tilt else 1.0)
    d.qpos.data[4] = Scalar[DTYPE](0.0)
    d.qpos.data[5] = Scalar[DTYPE](0.2164396139381029 if tilt else 0.0)
    d.qpos.data[6] = Scalar[DTYPE](0.0)
    for i in range(7, NQ):
        d.qpos.data[i] = Scalar[DTYPE](0)
    for i in range(NV):
        d.qvel.data[i] = Scalar[DTYPE](0)

    if torso_vel != 0.0:
        d.qvel.data[0] = Scalar[DTYPE](0.7 * torso_vel)
        d.qvel.data[2] = Scalar[DTYPE](-0.4 * torso_vel)
        d.qvel.data[3] = Scalar[DTYPE](1.3 * torso_vel)
        d.qvel.data[5] = Scalar[DTYPE](0.9 * torso_vel)

    for leg in range(N_LEGS):
        var sign = 1.0 if (leg % 2) == 0 else -1.0
        var yaw = 1 + 4 * leg
        var pitch = 2 + 4 * leg
        var knee = 3 + 4 * leg
        d.qpos.data[6 + yaw] = Scalar[DTYPE](0.4 * bend * sign)
        d.qpos.data[6 + pitch] = Scalar[DTYPE](bend * sign)
        d.qpos.data[6 + knee] = Scalar[DTYPE](-bend * sign)
        d.qvel.data[5 + yaw] = Scalar[DTYPE](0.5 * hinge_vel * sign)
        d.qvel.data[5 + pitch] = Scalar[DTYPE](hinge_vel * sign)
        d.qvel.data[5 + knee] = Scalar[DTYPE](-hinge_vel * sign)

    var state = List[Float64]()
    for i in range(NQ):
        state.append(Float64(d.qpos.data[i]))
    for i in range(NV):
        state.append(Float64(d.qvel.data[i]))
    return state^


def _mj_setup(state: List[Float64]) raises -> Tuple[PythonObject, PythonObject,
                                                    PythonObject]:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path("mojo_rl/envs/dm_control/assets/quadruped_walk.xml")
    var dat = mujoco.MjData(m)
    for i in range(NQ):
        dat.qpos[i] = state[i]
    for i in range(NV):
        dat.qvel[i] = state[NQ + i]
    for i in range(Int(py=m.nu)):
        dat.ctrl[i] = 0.0
    for i in range(Int(py=m.na)):
        dat.act[i] = 0.0
    mujoco.mj_forward(m, dat)
    return (mujoco, m, dat)


def _our_step(mut d: Dat, mut mf: Mod, mut integ: Integ) raises:
    """One step, which fills cacc/cfrc_int at the PRE-integration state."""
    var zero_ctrl = List[Float64]()
    for _ in range(Mdl.ACTION_DIM):
        zero_ctrl.append(0.0)
    var act = List[Scalar[DTYPE]]()
    for _ in range(NA if NA > 0 else 1):
        act.append(Scalar[DTYPE](0))
    for i in range(NV):
        d.qfrc.data[i] = Scalar[DTYPE](0)
    var sf = Mdl.make_spec_fields[DTYPE]()
    Mdl.apply_actions(sf, d, zero_ctrl, act)
    integ.step["cpu"](d, mf)


def test_body_and_site_order_match_mujoco() raises:
    """Layer 1. Our records and MuJoCo's agree element-for-element, so an
    `mj_name2id` index is a valid index into ours."""
    print("--- quadruped: body / site ordering vs MuJoCo ---")
    var ctx = DeviceContext()
    var mf = Mod()
    var d = Dat()
    var state = _setup(d, mf, ctx, 3.0, True, 0.15, 0.5, 0.5)
    var integ = Integ()
    _our_step(d, mf, integ)

    var mj = _mj_setup(state)
    var mujoco = mj[0]
    var m = mj[1]
    var dat = mj[2]

    assert_true(Int(py=m.nbody) == NBODY, "nbody mismatch")
    assert_true(Int(py=m.nsite) == NSITE, "nsite mismatch")
    assert_true(Int(py=m.nq) == NQ, "nq mismatch")
    assert_true(Int(py=m.nv) == NV, "nv mismatch")
    assert_true(Int(py=m.na) == NA, "na mismatch (activation count)")

    var worst_body = Float64(0)
    var worst_body_i = 0
    for b in range(NBODY):
        for k in range(3):
            var e = abs(
                Float64(d.xpos.data[b * 3 + k])
                - Float64(py=dat.xpos[b][k])
            )
            if e > worst_body:
                worst_body = e
                worst_body_i = b
    # ALL sites, not just the five the sensors read. This loop was restricted
    # to those five until 2026-08-01 because quadruped's twenty `rf_*`
    # rangefinder sites are declared with `fromto=`, which the parser did not
    # implement for sites — their local pos stayed (0,0,0), so `site_xpos`
    # landed on the torso origin, up to 0.4 m out. Now that sites honour
    # `fromto` it covers all NSITE, which is what makes it a site-ORDER gate
    # and not just a spot check.
    var sensor_sites = _toe_names()
    sensor_sites.append(String("torso"))
    var nsite_mj = Int(py=m.nsite)
    var worst_site = Float64(0)
    var worst_site_i = 0
    for s in range(nsite_mj):
        for k in range(3):
            var e = abs(
                Float64(d.site_xpos.data[s * 3 + k])
                - Float64(py=dat.site_xpos[s][k])
            )
            if e > worst_site:
                worst_site = e
                worst_site_i = s
    print("  worst |xpos err| =", worst_body, "at body", worst_body_i,
          " worst |site_xpos err| =", worst_site, "at site", worst_site_i)
    print("    body", worst_body_i, "ours =",
          Float64(d.xpos.data[worst_body_i * 3 + 0]),
          Float64(d.xpos.data[worst_body_i * 3 + 1]),
          Float64(d.xpos.data[worst_body_i * 3 + 2]))
    print("    body", worst_body_i, "mj   =",
          Float64(py=dat.xpos[worst_body_i][0]),
          Float64(py=dat.xpos[worst_body_i][1]),
          Float64(py=dat.xpos[worst_body_i][2]))
    print("    site", worst_site_i, "ours =",
          Float64(d.site_xpos.data[worst_site_i * 3 + 0]),
          Float64(d.site_xpos.data[worst_site_i * 3 + 1]),
          Float64(d.site_xpos.data[worst_site_i * 3 + 2]))
    print("    site", worst_site_i, "mj   =",
          Float64(py=dat.site_xpos[worst_site_i][0]),
          Float64(py=dat.site_xpos[worst_site_i][1]),
          Float64(py=dat.site_xpos[worst_site_i][2]))
    assert_true(worst_body < FK_TOL, "body order/pose diverges from MuJoCo")
    assert_true(worst_site < FK_TOL, "site order/pose diverges from MuJoCo")

    # The site-frame == body-frame assumption in sensors/site_acc.mojo.
    for name in sensor_sites:
        var sid = Int(
            py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
        )
        assert_true(sid >= 0, "missing site " + name)
        var w = abs(Float64(py=m.site_quat[sid][0]) - 1.0)
        for k in range(1, 4):
            w += abs(Float64(py=m.site_quat[sid][k]))
        assert_true(
            w < 1e-15,
            "site " + name + " is rotated w.r.t. its body — "
            "sensors/site_acc.mojo assumes identity (no site_xmat)",
        )
    print("  PASS: ordering + identity site frames")


def _find_standing_z() raises -> Float64:
    """Lowest torso height, in 5 mm steps, that puts at least four toes on the
    floor at the reference pose.

    Searched rather than hardcoded: the resting height follows from the leg
    geometry, and a literal would go stale the moment the model does — quietly,
    by making the "standing" case a second free-flight case.
    """
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path("mojo_rl/envs/dm_control/assets/quadruped_walk.xml")
    var dat = mujoco.MjData(m)
    var z = 0.70
    for _ in range(120):
        mujoco.mj_resetData(m, dat)
        dat.qpos[2] = z
        mujoco.mj_forward(m, dat)
        if Int(py=dat.ncon) >= 4:
            print("  standing height found at z =", z,
                  " ncon =", Int(py=dat.ncon))
            return z
        z -= 0.005
    return -1.0


def _compare_sensors(
    label: String, z: Float64, tilt: Bool, bend: Float64, torso_vel: Float64,
    hinge_vel: Float64, tol: Float64, expect_contacts: Bool,
) raises:
    """`tol` is a RELATIVE bound on the sensor readings. With contacts it also
    switches the force/torque check to sign + magnitude only — see
    `test_standing_sensors`."""
    print("--- quadruped acc-stage sensors vs MuJoCo:", label, "---")
    var ctx = DeviceContext()
    var mf = Mod()
    var d = Dat()
    var state = _setup(d, mf, ctx, z, tilt, bend, torso_vel, hinge_vel)
    var integ = Integ()
    _our_step(d, mf, integ)

    var mj = _mj_setup(state)
    var mujoco = mj[0]
    var m = mj[1]
    var dat = mj[2]

    var mj_ncon = Int(py=dat.ncon)
    var our_ncon = Int(d.meta.data[0])
    print("  ncon: ours =", our_ncon, " MuJoCo =", mj_ncon)
    if expect_contacts:
        assert_true(mj_ncon > 0, "expected contacts in the standing case")
        assert_true(our_ncon > 0, "our engine found no contacts")
    else:
        assert_true(mj_ncon == 0, "expected contact-free flight")
        assert_true(our_ncon == 0, "our engine invented a contact")
        # nefc is NOT zero here: the four <equality><tendon> rows are always
        # active. Only contact rows must be absent.
        assert_true(
            Int(py=dat.nefc) == Int(py=dat.ne),
            "expected equality rows only (no contact/limit rows)",
        )

    # --- the raw rnePostConstraint products, before any sensor read -------
    var worst_qacc = Float64(0)
    var wi_qacc = 0
    for i in range(NV):
        var e = abs(
            Float64(integ.scratch.qacc_constrained.data[i])
            - Float64(py=dat.qacc[i])
        )
        if e > worst_qacc:
            worst_qacc = e
            wi_qacc = i
    print("  worst qacc dof =", wi_qacc,
          " ours =", Float64(integ.scratch.qacc_constrained.data[wi_qacc]),
          " mj =", Float64(py=dat.qacc[wi_qacc]),
          " mj qfrc_passive =", Float64(py=dat.qfrc_passive[wi_qacc]),
          " mj qfrc_constraint =", Float64(py=dat.qfrc_constraint[wi_qacc]))
    print("  mj ne =", Int(py=dat.ne), " nefc =", Int(py=dat.nefc))
    var worst_act = Float64(0)
    var worst_bias = Float64(0)
    var wi_act = 0
    var wi_bias = 0
    for i in range(NV):
        var ea = abs(
            Float64(d.qfrc.data[i]) - Float64(py=dat.qfrc_actuator[i])
        )
        if ea > worst_act:
            worst_act = ea
            wi_act = i
        var eb = abs(
            Float64(integ.scratch.bias.data[i]) - Float64(py=dat.qfrc_bias[i])
        )
        if eb > worst_bias:
            worst_bias = eb
            wi_bias = i
    print("  worst |qfrc_actuator| =", worst_act, "at dof", wi_act,
          " ours =", Float64(d.qfrc.data[wi_act]),
          " mj =", Float64(py=dat.qfrc_actuator[wi_act]))
    print("  worst |qfrc_bias| =", worst_bias, "at dof", wi_bias,
          " ours =", Float64(integ.scratch.bias.data[wi_bias]),
          " mj =", Float64(py=dat.qfrc_bias[wi_bias]))
    var wp = Float64(0)
    for i in range(NV):
        wp = max(wp, abs(Float64(py=dat.qfrc_passive[i])))
    var wc = Float64(0)
    for i in range(NV):
        wc = max(wc, abs(Float64(py=dat.qfrc_constraint[i])))
    print("  mj max |qfrc_passive| =", wp, " mj max |qfrc_constraint| =", wc)

    var worst_cvel = Float64(0)
    var worst_cacc = Float64(0)
    var worst_cext = Float64(0)
    var worst_cint = Float64(0)
    var wb_acc = 0
    for b in range(NBODY):
        for k in range(6):
            worst_cvel = max(worst_cvel, abs(
                Float64(d.cvel.data[b * 6 + k]) - Float64(py=dat.cvel[b][k])
            ))
            var ea = abs(
                Float64(d.cacc.data[b * 6 + k]) - Float64(py=dat.cacc[b][k])
            )
            if ea > worst_cacc:
                worst_cacc = ea
                wb_acc = b
            worst_cext = max(worst_cext, abs(
                Float64(d.cfrc_ext.data[b * 6 + k])
                - Float64(py=dat.cfrc_ext[b][k])
            ))
            worst_cint = max(worst_cint, abs(
                Float64(d.cfrc_int.data[b * 6 + k])
                - Float64(py=dat.cfrc_int[b][k])
            ))
    var qacc_scale = Float64(0)
    for i in range(NV):
        qacc_scale = max(qacc_scale, abs(Float64(py=dat.qacc[i])))
    print("  |qacc| scale =", qacc_scale)
    print("  worst |qacc| =", worst_qacc, " |cvel| =", worst_cvel,
          " |cacc| =", worst_cacc, "(body", wb_acc, ")",
          " |cfrc_ext| =", worst_cext, " |cfrc_int| =", worst_cint)

    # ⚠ These two were COMPUTED AND PRINTED here from the day this file was
    # written, and never asserted. That is precisely how bug 30 (the contact
    # frame hint read as a tangent) survived inside the one file that was
    # already measuring it: this line printed |cfrc_ext| = 3.88 against a
    # ~100 N scale while every assertion in the file passed. A printed
    # diagnostic is not a gate. Asserting them also makes this the only
    # numeric check on `cfrc_ext` anywhere in the suite.
    var cext_scale = Float64(1.0)
    var cint_scale = Float64(1.0)
    for b in range(NBODY):
        for k in range(6):
            cext_scale = max(cext_scale, abs(Float64(py=dat.cfrc_ext[b][k])))
            cint_scale = max(cint_scale, abs(Float64(py=dat.cfrc_int[b][k])))
    print("    cfrc_ext rel =", worst_cext / cext_scale,
          " cfrc_int rel =", worst_cint / cint_scale,
          " (scales", cext_scale, cint_scale, ")")
    assert_true(
        worst_cext / cext_scale < tol, "cfrc_ext diverges from MuJoCo"
    )
    assert_true(
        worst_cint / cint_scale < tol, "cfrc_int diverges from MuJoCo"
    )
    print("    cacc[", wb_acc, "] ours =",
          Float64(d.cacc.data[wb_acc * 6 + 0]),
          Float64(d.cacc.data[wb_acc * 6 + 1]),
          Float64(d.cacc.data[wb_acc * 6 + 2]),
          Float64(d.cacc.data[wb_acc * 6 + 3]),
          Float64(d.cacc.data[wb_acc * 6 + 4]),
          Float64(d.cacc.data[wb_acc * 6 + 5]))
    print("    cacc[", wb_acc, "] mj   =",
          Float64(py=dat.cacc[wb_acc][0]), Float64(py=dat.cacc[wb_acc][1]),
          Float64(py=dat.cacc[wb_acc][2]), Float64(py=dat.cacc[wb_acc][3]),
          Float64(py=dat.cacc[wb_acc][4]), Float64(py=dat.cacc[wb_acc][5]))

    var torso_sid = Int(
        py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "torso")
    )
    var torso_bid = Int(py=m.site_bodyid[torso_sid])

    # --- accelerometer -----------------------------------------------------
    var acct = site_accelerometer[DTYPE](
        d.cvel.data, d.cacc.data, d.subtree_com.data, d.site_xpos.data,
        d.xquat.data, mf.bodies.data, mf.sites.data, torso_bid, torso_sid,
    )
    var acc = [acct[0], acct[1], acct[2]]
    var mj_acc = _sensor_slice(mujoco, m, "imu_accel", dat)
    var worst_acc = abs(acc[0] - mj_acc[0])
    worst_acc = max(worst_acc, abs(acc[1] - mj_acc[1]))
    worst_acc = max(worst_acc, abs(acc[2] - mj_acc[2]))
    var acc_scale = Float64(1)
    for k in range(3):
        acc_scale = max(acc_scale, abs(mj_acc[k]))
    print("  accelerometer ours =", acc[0], acc[1], acc[2])
    print("                 mj  =", mj_acc[0], mj_acc[1], mj_acc[2])
    print("                 worst rel err =", worst_acc / acc_scale)
    assert_true(
        worst_acc / acc_scale < tol, "accelerometer diverges from MuJoCo"
    )

    # --- gyro / velocimeter (existing sensors, free cross-check) -----------
    var fvt = site_frame_velocity[DTYPE](
        d.xvel.data, d.xangvel.data, d.xipos.data, d.xquat.data,
        d.site_xpos.data, mf.sites.data, torso_bid, torso_sid,
    )
    var fv = [fvt[0], fvt[1], fvt[2], fvt[3], fvt[4], fvt[5]]
    var mj_gyro = _sensor_slice(mujoco, m, "imu_gyro", dat)
    var mj_vel = _sensor_slice(mujoco, m, "velocimeter", dat)
    var worst_gv = Float64(0)
    for k in range(3):
        worst_gv = max(worst_gv, abs(fv[k] - mj_vel[k]))
        worst_gv = max(worst_gv, abs(fv[3 + k] - mj_gyro[k]))
    print("  gyro+velocimeter worst abs err =", worst_gv)
    assert_true(worst_gv < FK_TOL, "velocimeter/gyro diverge from MuJoCo")

    # --- force / torque at the four toes -----------------------------------
    var worst_ft = Float64(0)
    var ft_scale = Float64(1)
    var sign_flips = 0
    var our_fz_sum = Float64(0)
    var mj_fz_sum = Float64(0)
    var fz_min = Float64(1e30)
    var fz_max = Float64(-1e30)
    var toes = _toe_names()
    for ti in range(4):
        var nm = toes[ti]
        var sid = Int(py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, nm))
        var bid = Int(py=m.site_bodyid[sid])
        var ftt = site_force_torque[DTYPE](
            d.cfrc_int.data, d.subtree_com.data, d.site_xpos.data,
            d.xquat.data, mf.bodies.data, mf.sites.data, bid, sid,
        )
        var ft = [ftt[0], ftt[1], ftt[2], ftt[3], ftt[4], ftt[5]]
        var mj_f = _sensor_slice(mujoco, m, "force_" + nm, dat)
        var mj_t = _sensor_slice(mujoco, m, "torque_" + nm, dat)
        our_fz_sum += ft[2]
        mj_fz_sum += mj_f[2]
        fz_min = min(fz_min, ft[2])
        fz_max = max(fz_max, ft[2])
        for k in range(3):
            ft_scale = max(ft_scale, abs(mj_f[k]))
            ft_scale = max(ft_scale, abs(mj_t[k]))
            worst_ft = max(worst_ft, abs(ft[k] - mj_f[k]))
            worst_ft = max(worst_ft, abs(ft[3 + k] - mj_t[k]))
            # SIGN is what the contact -> cfrc_ext mapping can get wrong; only
            # components MuJoCo reports as loaded are meaningful.
            if abs(mj_f[k]) > 1.0 and (ft[k] * mj_f[k]) < 0.0:
                sign_flips += 1
        print("  force ", nm, " ours =", ft[0], ft[1], ft[2],
              " mj =", mj_f[0], mj_f[1], mj_f[2])
        print("  torque", nm, " ours =", ft[3], ft[4], ft[5],
              " mj =", mj_t[0], mj_t[1], mj_t[2])
    print("  force/torque worst rel err =", worst_ft / ft_scale,
          " sign flips =", sign_flips)
    var spread = abs(fz_max - fz_min)
    print("  sum force_z ours =", our_fz_sum, " mj =", mj_fz_sum,
          " toe spread =", spread)

    assert_true(sign_flips == 0, "a toe force sensor has the WRONG SIGN")

    if expect_contacts:
        # TOTAL VERTICAL FORCE — the whole robot's weight through four toes.
        # This is the gate that says the contact -> cfrc_ext -> cfrc_int ->
        # sensor chain is right end to end, and it is tight (1e-11 observed).
        # It took three fixes to get here, all silent before quadruped:
        #   * the four <equality><tendon> rows are now rows of the Newton
        #     system rather than a post-solve Gauss-Seidel pass, which took
        #     qacc from 45% off to 4e-9 (constraints/tendon_limit.mojo);
        #   * `mju_decodePyramid` says a pyramidal contact's normal force is
        #     the SUM of its four edge forces; we were halving it, so every
        #     contact force RECORD on the pyramidal path read half true. Ant's
        #     contact_cost — the only other consumer — is a squared norm, so
        #     it had been costing a quarter of what it should.
        #   * `cfrc_ext` read the contact record's FRAME_T1 field as the
        #     tangent when it is only a HINT, so the TANGENTIAL force pointed
        #     somewhere arbitrary (collision/contact_frame.mojo).
        var rel = abs(our_fz_sum - mj_fz_sum) / max(abs(mj_fz_sum), 1.0)
        print("  sum force_z rel err =", rel)
        assert_true(
            our_fz_sum < 0.0 and mj_fz_sum < 0.0,
            "the toes are not pushing back at all",
        )
        assert_true(
            rel < tol, "total vertical toe force is not the right size"
        )
        # The pose is four-fold symmetric, so the four toes must read the same
        # magnitude — this would catch a per-leg indexing slip that a total
        # averages away.
        assert_true(
            spread < 1e-6,
            "the four toes disagree under a symmetric pose",
        )
        # Per-component too, which is the part that pins the TANGENTIAL
        # direction. The four legs splay at 90 degrees, so a common
        # inward/outward toe force cancels in the net wrench: equilibrium does
        # not constrain it and `qacc` cannot see it. That made this the only
        # assertion in the suite able to catch the FRAME_T1-hint bug, which
        # sat here at 0.221 while every other number in the file was at 1e-9.
        assert_true(
            worst_ft / ft_scale < STAND_COMPONENT_TOL,
            "force/torque components diverge from MuJoCo",
        )
    else:
        assert_true(
            worst_ft / ft_scale < tol,
            "force/torque sensors diverge from MuJoCo",
        )


def test_free_flight_sensors() raises:
    """Layer 2 — contact-free, so cfrc_ext == 0 and only the recursion is
    under test."""
    _compare_sensors(
        "free flight", 3.0, True, 0.15, 0.5, 0.5, FLIGHT_TOL, False
    )


def test_standing_sensors() raises:
    """Layer 3 — toes loaded, so the contact -> cfrc_ext mapping (and its
    SIGN) is under test."""
    var z = _find_standing_z()
    assert_true(z > 0.0, "no standing height found — model geometry changed?")
    _compare_sensors(
        "standing on the floor", z, False, 0.0, 0.0, 0.0,
        STAND_REL_TOL, True,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
