"""Behavioral gate: the per-field Euler step vs LIVE MuJoCo (tumbling free-flight
Ant, contact-free, float64, opt.integrator=0).

Drives the per-field path `EulerIntegrator.step["cpu"]` over
Data/Model. The MuJoCo comparison is the primary (golden) reference.
The former legacy-CPU cross-check was replaced during Phase-0 of the physics3d
sunset: the active-limit sub-test now checks the fields-CPU trajectory against a
frozen GOLDEN fingerprint (it previously compared to the legacy CPU Euler step),
so this gate survives deletion of the legacy Model/Data slab.

Regenerate the active-limit golden after an INTENTIONAL physics change:
HARVEST=True, run on Apple, paste, HARVEST=False.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_euler_fields_vs_mujoco.mojo
"""

from std.testing import assert_true, TestSuite
from std.python import Python
from std.math import abs
from std.collections import InlineArray
from std.gpu.host import DeviceContext

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.envs.ant.ant_xml import AntModel

comptime DTYPE = DType.float64
comptime NQ = AntModel.NQ  # 15
comptime NV = AntModel.NV  # 14
comptime NBODY = AntModel.NBODY
comptime NJOINT = AntModel.NJOINT
comptime NGEOM = AntModel.NGEOM
comptime MAX_CONTACTS = AntModel.MAX_CONTACTS
comptime NSITE = AntModel.NSITE
comptime NEQ = AntModel.MAX_EQUALITY
comptime NTEN = AntModel.MAX_TENDON
comptime NEXCL = AntModel.NEXCLUDE

# ⚠ These were inherited from the legacy free-joint gate at 1e-4 / 5e-3 —
# NINE orders of magnitude looser than what the path actually achieves, so
# they could not fail for any reason short of a total break. That slack is
# precisely what hid a real bug: `quat_integrate` was a first-order
# approximation where MuJoCo's `mju_quatIntegrate` is the exact exponential
# map, which made this a DIFFERENT INTEGRATOR for every free-rooted model.
# Ant sat comfortably inside 1e-4 the whole time. Fixed 2026-07-30; observed
# is now 4.8e-13 / 4.9e-13 (1 step) and 4.7e-13 / 4.7e-12 (10 steps), so the
# budgets are set ~20x above that. Do NOT loosen these back to "safe" values;
# if a change moves them, that is the gate doing its job.
comptime QPOS_ABS_TOL_1: Float64 = 1e-11
comptime QVEL_ABS_TOL_1: Float64 = 1e-11
comptime QPOS_ABS_TOL_10: Float64 = 1e-11
comptime QVEL_ABS_TOL_10: Float64 = 1e-10

# --- GOLDEN fingerprint for the active-limit fields-CPU trajectory -----------
comptime HARVEST = False  # True => print fingerprint + skip asserts (regen)
comptime GOLD_RTOL = 1e-6  # fields-CPU f64 is deterministic across devices
# ⚠ This golden pins our OWN output, and there is no MuJoCo comparison in this
# file for the active-limit case (the two sub-tests above assert nefc == 0).
# It therefore CANNOT catch an error in the constraint response — it froze one
# for months: `dof_invweight0` was ~1% wrong, and this number faithfully
# preserved it. The authority for constrained behaviour is
# tests/physics3d/test_constraints_vs_mujoco.mojo, which compares against
# MuJoCo directly. Treat this fingerprint as a determinism check only.
#
# Regenerated 2026-07-30 (b) for bug 18 (dof_invweight0 built at the env reset
# pose instead of qpos0, and missing MuJoCo's free/ball dof-group averaging).
# Moved 1.2e-6 (qpos) / 2.3e-5 (qvel) — small because the limit rows are only
# briefly active over these 10 steps. What says this is a fix and not a
# regression: the MuJoCo-gated sub-tests above did NOT move (4.8e-13 / 4.7e-12,
# they run limit-free), while the active-limit case measured against MuJoCo in
# test_constraints_vs_mujoco went 9.4e-6 -> 4.4e-13 in the same change.
# Previously regenerated 2026-07-30 (a) for the exact `quat_integrate` (bug 17).
comptime GOLD_LIM_QPOS = 12.238441724124275
comptime GOLD_LIM_QVEL = 17.46150375918525


def _tumbling_qpos() -> InlineArray[Float64, NQ]:
    """Contact-free AND limit-free: torso high up, legs mid-range."""
    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[2] = 2.0  # z — contact-free
    qpos[3] = 0.9659258262890683  # qw (30 deg tilt about y)
    qpos[5] = 0.25881904510252074  # qy
    qpos[8] = 0.9  # ankle_1
    qpos[10] = -0.9  # ankle_2
    qpos[12] = -0.9  # ankle_3
    qpos[14] = 0.9  # ankle_4
    return qpos


def _tumbling_qvel() -> InlineArray[Float64, NV]:
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[3] = 2.0
    qvel[4] = 1.0
    qvel[5] = 0.5
    return qvel


def _make_model(
    ctx: DeviceContext,
) raises -> Model[DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL, 0]:
    """Build the model into Model via the model-def init + flattening."""
    var mf = Model[DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTEN, NSITE, NEXCL, 0]()
    AntModel.init_fields[DTYPE, 0](ctx, mf)
    return mf^


def _compare_vs_mujoco(
    num_steps: Int, qpos_tol: Float64, qvel_tol: Float64
) raises:
    print(
        "--- fields-CPU Euler vs MuJoCo, tumbling Ant,", num_steps, "steps ---"
    )
    var qpos_init = _tumbling_qpos()
    var qvel_init = _tumbling_qvel()
    var ctx = DeviceContext()
    var mf = _make_model(ctx)

    # Fields path (f64, CPU target, BATCH=1).
    var d = Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1]()
    for i in range(NQ):
        d.qpos.data[i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        d.qvel.data[i] = Scalar[DTYPE](qvel_init[i])
    var integ = EulerIntegrator[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, NEQ, NTEN, NSITE,
        0, 0, BATCH=1,
    ]()
    for _ in range(num_steps):
        for i in range(NV):
            d.qfrc.data[i] = Scalar[DTYPE](0)
        integ.step["cpu"](d, mf)

    # Live MuJoCo reference.
    var mujoco = Python.import_module("mujoco")
    var xml_path = (
        "./references/Gymnasium-main/gymnasium/envs/mujoco/assets/ant.xml"
    )
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.integrator = 0
    mj_model.opt.solver = 2
    var mj_data = mujoco.MjData(mj_model)
    for i in range(NQ):
        mj_data.qpos[i] = qpos_init[i]
    for i in range(NV):
        mj_data.qvel[i] = qvel_init[i]
    for _ in range(num_steps):
        mujoco.mj_step(mj_model, mj_data)
        # nefc == 0 => NO active constraints (contacts NOR joint limits) —
        # the unconstrained comparison is only valid under this.
        assert_true(
            Int(py=mj_data.nefc) == 0,
            "expected constraint-free config (no contacts, no active limits)",
        )

    var mj_qpos = mj_data.qpos.flatten().tolist()
    var mj_qvel = mj_data.qvel.flatten().tolist()
    var max_qpos_err: Float64 = 0.0
    var max_qvel_err: Float64 = 0.0
    for i in range(NQ):
        var e = abs(Float64(d.qpos.data[i]) - Float64(py=mj_qpos[i]))
        if e > max_qpos_err:
            max_qpos_err = e
    for i in range(NV):
        var e = abs(Float64(d.qvel.data[i]) - Float64(py=mj_qvel[i]))
        if e > max_qvel_err:
            max_qvel_err = e
    print(
        "  max |qpos err| vs mj =", max_qpos_err,
        " max |qvel err| vs mj =", max_qvel_err,
    )
    assert_true(
        max_qpos_err < qpos_tol, "fields Euler qpos diverged from MuJoCo"
    )
    assert_true(
        max_qvel_err < qvel_tol, "fields Euler qvel diverged from MuJoCo"
    )


def test_fields_euler_vs_mujoco_1_step() raises:
    _compare_vs_mujoco(1, QPOS_ABS_TOL_1, QVEL_ABS_TOL_1)


def test_fields_euler_vs_mujoco_10_steps() raises:
    _compare_vs_mujoco(10, QPOS_ABS_TOL_10, QVEL_ABS_TOL_10)


def test_fields_euler_active_limits_golden() raises:
    """Active-limit dynamics through the fields path (ankles at qpos=0, VIOLATING
    their ranges) vs a frozen GOLDEN fingerprint of the fields-CPU trajectory.

    Originally compared to the legacy CPU Euler step; the legacy reference was
    frozen here during the sunset. fields-CPU f64 is deterministic, so the
    fingerprint is device-independent."""
    print("--- fields-CPU active limits GOLDEN, tumbling Ant ---")
    var qpos_init = _tumbling_qpos()
    for k in range(4):
        qpos_init[8 + 2 * k] = 0.0  # ankles back to 0 -> limits ACTIVE
    var qvel_init = _tumbling_qvel()
    var ctx = DeviceContext()
    var mf = _make_model(ctx)

    var d = Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1]()
    for i in range(NQ):
        d.qpos.data[i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        d.qvel.data[i] = Scalar[DTYPE](qvel_init[i])
    var integ = EulerIntegrator[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, NEQ, NTEN, NSITE,
        0, 0, BATCH=1,
    ]()
    for _ in range(10):
        for i in range(NV):
            d.qfrc.data[i] = Scalar[DTYPE](0)
        integ.step["cpu"](d, mf)

    var fp_qpos = Float64(0)
    var fp_qvel = Float64(0)
    for i in range(NQ):
        fp_qpos += Float64(d.qpos.data[i]) * Float64(i + 1)
    for i in range(NV):
        fp_qvel += Float64(d.qvel.data[i]) * Float64(i + 1)

    if HARVEST:
        print("  HARVEST GOLD_LIM_QPOS =", fp_qpos)
        print("  HARVEST GOLD_LIM_QVEL =", fp_qvel)
    else:
        var dqp = abs(fp_qpos - GOLD_LIM_QPOS) / (
            abs(GOLD_LIM_QPOS) if abs(GOLD_LIM_QPOS) > 1e-9 else 1.0
        )
        var dqv = abs(fp_qvel - GOLD_LIM_QVEL) / (
            abs(GOLD_LIM_QVEL) if abs(GOLD_LIM_QVEL) > 1e-9 else 1.0
        )
        print("  active-limit fp qpos=", fp_qpos, " qvel=", fp_qvel)
        assert_true(
            dqp < GOLD_RTOL and dqv < GOLD_RTOL,
            "fields active-limit trajectory diverged from golden",
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
