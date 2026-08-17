"""Reach parity measured WHERE THE TASK ACTUALLY OPERATES.

⚠⚠ EVERY REACH PARITY NUMBER BEFORE THIS FILE WAS MEASURED OFF-DISTRIBUTION,
and that is the defect this gate exists to prevent recurring. The 40-pose sweep
those numbers came from draws `qpos` uniformly from dm_control's sampling
bounds and steps whatever comes out. dm_control's OWN reset draws from the same
bounds and then REJECTS anything in contact — measured, 10 of 10 episodes reset
with zero contacts. Omitting the rejection does not sample the task's states
slightly wrong, it samples a different regime entirely:

                            the sweep          the task
    deepest penetration     -317 mm            -0.55 mm      (575x)
    simultaneous contacts   up to 45           up to 2
    worst |d(qvel)|         62.3               see below

`qpos0` — the sweep's pose 0, and the one a task-#60 was filed on as "the pose
every reset starts from" — has the arm 178 mm through the floor with 55
penetrating contacts. No episode ever visits it. A parity conclusion drawn
there is a conclusion about deep interpenetration, which is a regime where the
contact set is ill-conditioned and any two implementations disagree.

WHAT THIS FILE MEASURES INSTEAD:

  1. CONTACT-FREE dynamics over a long random-control rollout. This is the bulk
     of any episode and it isolates smooth dynamics + integrator + actuators.
  2. SHALLOW-CONTACT poses, bisected into the [-2 mm, 0) band the task really
     reaches, with contact COUNTS asserted against MuJoCo's.

⚠ MUJOCO'S `qfrc_actuator` IS COPIED IN rather than our actuators being run, so
this gate measures the CONTACT and SOLVER path. Actuator fidelity is gated by
`test_manipulation_reach_def` and the `<velocity>` work; mixing them here would
make a failure ambiguous between two subsystems.

⚠ SHARED STATE, ONE STEP AT A TIME — never a free-running rollout. Contact
dynamics are chaotic, so two engines released together diverge exponentially
no matter how accurate each step is, and the number that falls out measures a
Lyapunov exponent rather than this engine
(`feedback_ab_arms_must_share_the_warmup_state`).

⚠ COMPARED AFTER A FULL `mj_step` ON BOTH SIDES. `mj_forward`'s `qacc` is not
what `mj_step` integrates — `mj_Euler` treats `dof_damping` implicitly — and
that trap has already produced one phantom 1.5% "solver residual" on this exact
model (`feedback_mj_forward_qacc_is_not_what_mj_step_integrates`).

Run with:
    pixi run mojo run -I . tests/dm_control/test_reach_parity_in_distribution.mojo
"""

from std.math import abs
from std.python import Python
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.fields import Model, Data, Dims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.envs.dm_control.manipulation_reach_def import (
    ReachSiteFeaturesModel as MD,
)
from mojo_rl.physics3d.gpu.constants import META_IDX_NUM_CONTACTS
from max.gpu.host import DeviceContext
from mojo_rl.physics3d.model.model_dims import ModelDims

comptime DTYPE = DType.float64
comptime NMV: Int = 8000
comptime MD_2 = ModelDims[MD, 8000]

# The band a real rollout reaches. Measured under random control across 5
# episodes x 200 dm_control steps: deepest 0.55 mm, at most 2 simultaneous
# penetrating contacts. 2 mm is a few times that, so the gate is not tuned to
# the easiest possible case.
comptime SHALLOW_M: Float64 = 2e-3
comptime N_SHALLOW: Int = 12
comptime N_FREE_STEPS: Int = 400

# Contact-free steps exercise no solver, so this is integrator + smooth
# dynamics against MuJoCo's: measured 5.6e-17 over 1500 steps.
comptime TOL_FREE: Float64 = 1e-12
# Shallow-contact single-step: measured worst 4.2e-4 over 25 poses, typical
# ~1e-7. 5e-3 leaves an order of headroom while still being 4 ORDERS below the
# 62.3 the off-distribution sweep reported — i.e. it would not be passed by
# re-introducing the old regime.
comptime TOL_CONTACT: Float64 = 5e-3


def test_contact_free_dynamics_match_mujoco() raises:
    """The bulk of an episode: no contacts, so no solver."""
    var sf = MD.make_spec_fields[DTYPE]()
    print("=== contact-free reach dynamics vs MuJoCo ===")
    var sys = Python.import_module("sys")
    _ = sys.path.insert(0, "tests/dm_control")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var refmod = Python.import_module("manipulation_ref")

    var m = mujoco.MjModel.from_xml_path("mojo_rl/envs/dm_control/assets/manipulation/reach_site_features.xml")
    var md = mujoco.MjData(m)
    var nu = Int(py=m.nu)

    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD_2]()
    MD.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MD_2, 1]()
    MD.reset_data[DTYPE](sf, d)
    var integ = EulerIntegrator[DTYPE, MD_2, MD.CONE_TYPE, 1, SOLVER="newton", MAX_CONDIM=MD.MAX_CONDIM, NOSLIP_ITER=MD.NOSLIP_ITER]()

    var lo = refmod.arm_joint_bounds()[0]
    var hi = refmod.arm_joint_bounds()[1]
    var rng = np.random.default_rng(11)

    # dm_control's reset: draw, then REJECT anything in contact.
    var settled = False
    for _t in range(200):
        for i in range(MD.NQ):
            var v: Float64
            if i < 6:
                v = Float64(
                    py=rng.uniform(
                        Python.evaluate("float")(lo[i]),
                        Python.evaluate("float")(hi[i]),
                    )
                )
            else:
                v = Float64(py=rng.uniform(0.15, 1.35))
            md.qpos[i] = v
        for i in range(MD.NV):
            md.qvel[i] = 0.0
        mujoco.mj_forward(m, md)
        if Int(py=md.ncon) == 0:
            settled = True
            break
    assert_true(
        settled,
        "could not draw a contact-free start in 200 tries — the rejection"
        " step dm_control's reset applies is not reproducible here, so this"
        " gate is not sampling the task's distribution",
    )

    var worst = Float64(0)
    var n_free = 0
    for _step in range(N_FREE_STEPS):
        for a in range(nu):
            md.ctrl[a] = Float64(py=rng.uniform(-1.0, 1.0))
        var mq = md.qpos.flatten().tolist()
        var mv = md.qvel.flatten().tolist()
        for i in range(MD.NQ):
            d.qpos.data[i] = Scalar[DTYPE](Float64(py=mq[i]))
        for i in range(MD.NV):
            d.qvel.data[i] = Scalar[DTYPE](Float64(py=mv[i]))
        mujoco.mj_forward(m, md)
        var mfa = md.qfrc_actuator.flatten().tolist()
        for i in range(MD.NV):
            d.qfrc.data[i] = Scalar[DTYPE](Float64(py=mfa[i]))
        if Int(py=md.ncon) != 0:
            mujoco.mj_step(m, md)
            continue
        n_free += 1
        forward_kinematics["cpu"](d, mf)
        detect_contacts["cpu"](d, mf)
        integ.step["cpu"](d, mf)
        mujoco.mj_step(m, md)
        var mv2 = md.qvel.flatten().tolist()
        for i in range(MD.NV):
            var e = abs(Float64(d.qvel.data[i]) - Float64(py=mv2[i]))
            if e > worst:
                worst = e

    print("  contact-free steps:", n_free, "  worst |d(qvel)| =", worst)
    assert_true(
        n_free > N_FREE_STEPS // 2,
        "only " + String(n_free) + " of " + String(N_FREE_STEPS)
        + " steps were contact-free, so this is not the contact-free gate it"
        " claims to be",
    )
    assert_true(
        worst <= TOL_FREE,
        "contact-free reach dynamics diverge from MuJoCo by " + String(worst)
        + ". No solver runs on these steps, so this is smooth dynamics, the"
        " integrator, or the actuator force we copied in — NOT contacts",
    )
    print("  PASS")


def test_shallow_contact_parity() raises:
    """Contacts at the depths the task reaches — counts AND dynamics."""
    var sf = MD.make_spec_fields[DTYPE]()
    print("=== shallow-contact reach parity vs MuJoCo ===")
    var sys = Python.import_module("sys")
    _ = sys.path.insert(0, "tests/dm_control")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var refmod = Python.import_module("manipulation_ref")

    var m = mujoco.MjModel.from_xml_path("mojo_rl/envs/dm_control/assets/manipulation/reach_site_features.xml")
    var md = mujoco.MjData(m)
    var nu = Int(py=m.nu)

    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD_2]()
    MD.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MD_2, 1]()
    MD.reset_data[DTYPE](sf, d)
    var integ = EulerIntegrator[DTYPE, MD_2, MD.CONE_TYPE, 1, SOLVER="newton", MAX_CONDIM=MD.MAX_CONDIM, NOSLIP_ITER=MD.NOSLIP_ITER]()

    var lo = refmod.arm_joint_bounds()[0]
    var hi = refmod.arm_joint_bounds()[1]
    var rng = np.random.default_rng(11)

    var qa = InlineArray[Float64, MD.NQ](fill=0.0)
    var qb = InlineArray[Float64, MD.NQ](fill=0.0)
    var found = 0
    var worst = Float64(0)
    var worst_depth = Float64(0)
    var count_mismatches = 0
    var deepest_used = Float64(0)

    for _draw in range(400):
        if found >= N_SHALLOW:
            break
        # A: contact-free (the reset's rejection).  B: contacting.
        var ok_a = False
        for _t in range(40):
            for i in range(MD.NQ):
                var v: Float64
                if i < 6:
                    v = Float64(
                        py=rng.uniform(
                            Python.evaluate("float")(lo[i]),
                            Python.evaluate("float")(hi[i]),
                        )
                    )
                else:
                    v = Float64(py=rng.uniform(0.15, 1.35))
                qa[i] = v
                md.qpos[i] = v
            for i in range(MD.NV):
                md.qvel[i] = 0.0
            mujoco.mj_forward(m, md)
            if Int(py=md.ncon) == 0:
                ok_a = True
                break
        if not ok_a:
            continue
        var ok_b = False
        for _t in range(40):
            for i in range(MD.NQ):
                var v: Float64
                if i < 6:
                    v = Float64(
                        py=rng.uniform(
                            Python.evaluate("float")(lo[i]),
                            Python.evaluate("float")(hi[i]),
                        )
                    )
                else:
                    v = Float64(py=rng.uniform(0.15, 1.35))
                qb[i] = v
                md.qpos[i] = v
            mujoco.mj_forward(m, md)
            if Int(py=md.ncon) > 0:
                ok_b = True
                break
        if not ok_b:
            continue

        # Bisect the segment A->B for a penetration inside the task's band.
        var alo = Float64(0)
        var ahi = Float64(1)
        var got = False
        var depth = Float64(0)
        for _b in range(40):
            var a = (alo + ahi) * 0.5
            for i in range(MD.NQ):
                md.qpos[i] = qa[i] * (1.0 - a) + qb[i] * a
            for i in range(MD.NV):
                md.qvel[i] = 0.0
            mujoco.mj_forward(m, md)
            depth = 0.0
            for c in range(Int(py=md.ncon)):
                var dv = Float64(py=md.contact[c].dist)
                if dv < depth:
                    depth = dv
            if depth < -SHALLOW_M:
                ahi = a
            elif depth >= 0.0:
                alo = a
            else:
                got = True
                break
        if not got:
            continue
        found += 1
        if depth < deepest_used:
            deepest_used = depth

        for a in range(nu):
            md.ctrl[a] = 0.0
        var mq = md.qpos.flatten().tolist()
        for i in range(MD.NQ):
            d.qpos.data[i] = Scalar[DTYPE](Float64(py=mq[i]))
        for i in range(MD.NV):
            d.qvel.data[i] = 0
        mujoco.mj_forward(m, md)
        var mfa = md.qfrc_actuator.flatten().tolist()
        for i in range(MD.NV):
            d.qfrc.data[i] = Scalar[DTYPE](Float64(py=mfa[i]))

        forward_kinematics["cpu"](d, mf)
        detect_contacts["cpu"](d, mf)
        var our_n = Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
        var mj_n = Int(py=md.ncon)
        if our_n != mj_n:
            count_mismatches += 1

        integ.step["cpu"](d, mf)
        mujoco.mj_step(m, md)
        var mv2 = md.qvel.flatten().tolist()
        var e = Float64(0)
        for i in range(MD.NV):
            var x = abs(Float64(d.qvel.data[i]) - Float64(py=mv2[i]))
            if x > e:
                e = x
        if e > worst:
            worst = e
            worst_depth = depth
        print("   depth", depth, " ncon ours/mj", our_n, "/", mj_n,
              "  |dqvel|", e)

    print("  shallow poses:", found, " worst |d(qvel)| =", worst,
          " at depth", worst_depth)
    print("  contact-count mismatches:", count_mismatches,
          "  deepest used:", deepest_used, "m")

    assert_true(
        found >= N_SHALLOW,
        "only " + String(found) + " shallow-contact poses were constructed;"
        " the gate needs " + String(N_SHALLOW) + " or it is not sampling the"
        " contact regime at all",
    )
    # A pose that never penetrates would make the dynamics check vacuous.
    assert_true(
        deepest_used < -1e-6,
        "the constructed poses barely touch (deepest " + String(deepest_used)
        + " m), so no contact force was exercised",
    )
    assert_true(
        count_mismatches == 0,
        String(count_mismatches) + " of " + String(found) + " shallow poses"
        " disagree with MuJoCo on the CONTACT COUNT. ⚠ counts DO match in this"
        " regime — the 57-vs-55 that task #60 was filed on came from a pose"
        " with 178 mm of interpenetration, which no episode reaches",
    )
    assert_true(
        worst <= TOL_CONTACT,
        "shallow-contact reach dynamics diverge from MuJoCo by " + String(worst)
        + " at depth " + String(worst_depth) + " m",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
