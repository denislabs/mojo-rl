"""dog step divergence — a STAGED probe, not a gate.

`test_dog_rollout_matches_mujoco` reports `|d(qvel)| = 20.6` on the first
contacting step from a MuJoCo-settled pose. That number says the step is
wrong; it says nothing about WHICH stage of the step is wrong, and the model
constants are 8/8 so it is not upstream of the step.

This file runs ONE substep at ONE state and compares the stages in the order
the engine computes them, so the first line that disagrees IS the defect:

    1. qfrc_bias        RNE                       vs  dat.qfrc_bias
    2. fnet             qfrc - bias + passive     vs  dat.qfrc_smooth
    3. qacc_ws          M^-1 fnet                 vs  dat.qacc_smooth
    4. the CONTACT SET  ncon, body pairs, dist    vs  dat.contact[...]
    5. qacc_constrained the solve + noslip        vs  dat.qacc

Stage 4 is deliberately placed before any solver comparison. A solver fed a
different set of rows is not a solver bug, and reading stage 5 first would
file a mechanism against a symptom — the standing rule after the condim>=4
arc is that "is X correct" gets answered with a number.

WHY ctrl = 0 AND act = 0

All 38 actuators are `<general dyntype="filter">` with force `gainprm[0]*act`,
so at `act = 0` the actuator force is identically zero on BOTH sides and the
probe isolates bias + passive + contacts. The rollout's own divergence appears
at step 0, where `act` is still zero to within 1e-12 (measured: `|d(act)|` is
exactly 0.0), so nothing is being excluded that the failure depends on. The
probe asserts `dat.qfrc_actuator == 0` rather than assuming it.

WHY A SINGLE SUBSTEP, BUILT WITHOUT `Phyics3dEnv`

`Phyics3dEnv.step` takes `FRAME_SKIP = 3` substeps and each one overwrites the
scratch the earlier stages live in, so the intermediates of substep 1 are
unreadable through the env. Driving `EulerIntegrator` directly gives one
substep, and `mj_forward` at the same state gives MuJoCo's answer for exactly
that substep's inputs — no integration on either side.

Run with:
    pixi run mojo run -I . tests/dm_control/test_dog_step_probe.mojo
"""

from std.math import abs, sqrt, sin
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.dog import (
    DMDogStandWalkModel,
    dm_dog_stand_walk_xml,
)
from mojo_rl.physics3d.fields import Model, Data
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE,
    CONTACT_IDX_CONDIM,
    CONTACT_IDX_FORCE_N,
    CONTACT_IDX_FORCE_T1,
    CONTACT_IDX_FORCE_T2,
    CONTACT_IDX_FRICTION,
    CONTACT_IDX_FRICTION_SPIN,
    CONTACT_IDX_FRICTION_ROLL,
    CONTACT_IDX_SOLREF_0,
    CONTACT_IDX_SOLREF_1,
    CONTACT_IDX_SOLIMP_0,
    CONTACT_IDX_SOLIMP_1,
    CONTACT_IDX_SOLIMP_2,
    CONTACT_IDX_SOLIMP_3,
    CONTACT_IDX_SOLIMP_4,
    META_IDX_NUM_CONTACTS,
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_DAMPING,
    JOINT_IDX_STIFFNESS,
    JOINT_IDX_SPRINGREF,
    JOINT_IDX_FRICTIONLOSS,
    MODEL_META_SIZE,
    MODEL_META_IDX_DENSITY,
    MODEL_META_IDX_VISCOSITY,
)

comptime DTYPE = DType.float64
comptime M = DMDogStandWalkModel
comptime NQ = M.NQ
comptime NV = M.NV
comptime MC = M.MAX_CONTACTS

# MuJoCo-side only: it just produces a loaded pose on four feet.
comptime N_SETTLE: Int = 400


def _worst(a: List[Float64], b: PythonObject, n: Int) raises -> Tuple[Float64, Int]:
    var w = 0.0
    var at = -1
    for i in range(n):
        var e = abs(a[i] - Float64(py=b[i]))
        if e > w:
            w = e
            at = i
    return (w, at)


def _report(
    label: String, a: List[Float64], b: PythonObject, n: Int, top: Int
) raises -> Float64:
    """Print the `top` worst dofs with BOTH values.

    A max and an index localize nothing: `|d| = 101` at dof 2 is consistent
    with "ours is zero", "ours is doubled" and "ours has the wrong sign", and
    those are three different bugs. Printing both sides settles it in the same
    run instead of costing another nine-minute build.
    """
    var err = List[Float64]()
    var used = List[Bool]()
    for i in range(n):
        err.append(abs(a[i] - Float64(py=b[i])))
        used.append(False)
    for _r in range(top):
        var best = -1.0
        var at = -1
        for i in range(n):
            if not used[i] and err[i] > best:
                best = err[i]
                at = i
        if at < 0 or best <= 0.0:
            break
        used[at] = True
        print(
            "      ", label, "dof", at,
            " ours", a[at], " mj", Float64(py=b[at]),
            " |d|", best,
        )
    var w = 0.0
    for i in range(n):
        if err[i] > w:
            w = err[i]
    return w


def test_dog_step_stages_vs_mujoco() raises:
    print("=== dog: staged single-substep probe ===")
    var mujoco = Python.import_module("mujoco")
    var mm = mujoco.MjModel.from_xml_string(
        materialize[dm_dog_stand_walk_xml]()
    )
    var dat = mujoco.MjData(mm)

    mujoco.mj_resetData(mm, dat)
    for _ in range(N_SETTLE):
        mujoco.mj_step(mm, dat)
    for k in range(M.nact):
        dat.ctrl[k] = 0.0
        dat.act[k] = 0.0
    mujoco.mj_forward(mm, dat)

    # Non-vacuity: a probe run at a floating pose would gate nothing.
    var mj_ncon = Int(py=dat.ncon)
    print("  MuJoCo at the settled pose: ncon =", mj_ncon,
          " nefc =", Int(py=dat.nefc))
    assert_true(
        mj_ncon >= 4,
        "the settled pose has fewer than four contacts — it is not loaded, so"
        " nothing below tests the contact path",
    )

    var qfa = 0.0
    for i in range(NV):
        var v = abs(Float64(py=dat.qfrc_actuator[i]))
        if v > qfa:
            qfa = v
    print("  |qfrc_actuator| =", qfa, "(must be 0: act = 0 on both sides)")
    assert_true(
        qfa < 1e-14,
        "the actuator force is not zero at act = 0 — the probe's premise that"
        " actuators are excluded is false, so every stage below is confounded",
    )

    # --- our side, one substep from the same state -----------------------
    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DTYPE, 0](ctx, mf)
    var d = Data[DTYPE, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1]()
    M.reset_data[DTYPE](d)

    for i in range(NQ):
        d.qpos.data[i] = Scalar[DTYPE](Float64(py=dat.qpos[i]))
    for i in range(NV):
        d.qvel.data[i] = Scalar[DTYPE](Float64(py=dat.qvel[i]))
        d.qfrc.data[i] = Scalar[DTYPE](0)
    forward_kinematics["cpu"](d, mf)

    var integ = EulerIntegrator[
        DTYPE, M.NQ, M.NV, M.NBODY, M.NJOINT, M.MAX_CONTACTS, M.NGEOM,
        M.MAX_EQUALITY, M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
        M.CONE_TYPE, 1, SOLVER="newton",
        MAX_CONDIM=M.MAX_CONDIM, NOSLIP_ITER=M.NOSLIP_ITER,
    ]()
    integ.step["cpu"](d, mf)

    # ⚠ CAPTURE IMMEDIATELY. One `EulerIntegrator` owns one scratch, so the
    # contact-free runs below overwrite `qacc_constrained` — reading it after
    # them compares the CONTACT-FREE result against MuJoCo's contacting one and
    # reports a divergence that is entirely the probe's own.
    var qacc_full = List[Float64]()
    for i in range(NV):
        qacc_full.append(Float64(integ.scratch.qacc_constrained.data[i]))
    var bias_full = List[Float64]()
    for i in range(NV):
        bias_full.append(Float64(integ.scratch.bias.data[i]))

    # --- stage 0: the joint tables the model gate never compared ----------
    # ⚠ `test_dog_model_matches_dm_control` compares `jnt_stiffness` and
    # `armature` but NOT `dof_damping`, `jnt_springref` or `frictionloss`.
    # `AutoSpringDamper` writes stiffness AND damping from two DIFFERENT
    # formulas, so a matching stiffness is no evidence at all about damping.
    print("  [0] joint tables (damping / springref / frictionloss)")
    var worst_damp = 0.0
    var worst_damp_j = -1
    var worst_sref = 0.0
    var worst_sref_j = -1
    var worst_floss = 0.0
    var worst_floss_j = -1
    for j in range(M.NJOINT):
        var o = j * MODEL_JOINT_SIZE
        var dof = Int(Float64(mf.joints.data[o + JOINT_IDX_DOF_ADR]))
        var jt = Int(Float64(mf.joints.data[o + JOINT_IDX_TYPE]))
        var nd = 1
        if jt == 0:  # JNT_FREE
            nd = 6
        elif jt == 1:  # JNT_BALL
            nd = 3
        var ours_damp = Float64(mf.joints.data[o + JOINT_IDX_DAMPING])
        for k in range(nd):
            var e = abs(ours_damp - Float64(py=mm.dof_damping[dof + k]))
            if e > worst_damp:
                worst_damp = e
                worst_damp_j = j
        var e_s = abs(
            Float64(mf.joints.data[o + JOINT_IDX_SPRINGREF])
            - Float64(py=mm.qpos_spring[
                Int(Float64(mf.joints.data[o + JOINT_IDX_QPOS_ADR]))
            ])
        )
        if e_s > worst_sref:
            worst_sref = e_s
            worst_sref_j = j
        for k in range(nd):
            var e_f = abs(
                Float64(mf.joints.data[o + JOINT_IDX_FRICTIONLOSS])
                - Float64(py=mm.dof_frictionloss[dof + k])
            )
            if e_f > worst_floss:
                worst_floss = e_f
                worst_floss_j = j
    print("      dof_damping     max|d| =", worst_damp, " at joint",
          worst_damp_j)
    print("      springref       max|d| =", worst_sref, " at joint",
          worst_sref_j)
    print("      frictionloss    max|d| =", worst_floss, " at joint",
          worst_floss_j)
    print("      joint 0: type",
          Int(Float64(mf.joints.data[JOINT_IDX_TYPE])),
          " armature", Float64(mf.joints.data[JOINT_IDX_ARMATURE]),
          " damping", Float64(mf.joints.data[JOINT_IDX_DAMPING]),
          " stiffness", Float64(mf.joints.data[JOINT_IDX_STIFFNESS]),
          " springref", Float64(mf.joints.data[JOINT_IDX_SPRINGREF]),
          " (MuJoCo: armature", Float64(py=mm.dof_armature[0]),
          " damping", Float64(py=mm.dof_damping[0]),
          " stiffness", Float64(py=mm.jnt_stiffness[0]), ")")
    print("      fluid meta: density",
          Float64(mf.meta.data[MODEL_META_IDX_DENSITY]),
          " viscosity", Float64(mf.meta.data[MODEL_META_IDX_VISCOSITY]),
          " (MuJoCo:", Float64(py=mm.opt.density),
          Float64(py=mm.opt.viscosity), ")")

    # --- stage 1: RNE bias ------------------------------------------------
    # ⚠ ONLY READ SCRATCH THAT `_finalize_env` LEAVES ALONE. The last stage of
    # `EulerIntegrator.step` reuses `scratch.fnet` for `M * qacc_constrained`
    # and `scratch.qacc_ws` for the implicit-damped result, so reading either
    # after `step()` compares a post-solve quantity against a pre-solve one and
    # reports a large, entirely fictitious divergence. `bias`, `m_inv` and
    # `qacc_constrained` survive; `fnet`, `qacc_ws`, `M`, `L` and `D` do not.
    print("  [1] qfrc_bias")
    var r1 = _report("bias  ", bias_full, dat.qfrc_bias, NV, 4)
    print("      max|d| =", r1)

    # --- stage 2: PURE SMOOTH DYNAMICS, no contacts and no active limits ---
    # A floating dog at `qpos0` with zero velocity: gravity and the joint
    # springs are the only drivers, every hinge sits inside its range, and
    # nothing touches. `qacc_constrained == qacc_smooth` on both sides, so
    # this is the mass matrix + passive forces + LDL solve and nothing else.
    # The contacts-disabled run below still carries limit rows, which is why
    # it cannot play this role.
    var d2 = Data[DTYPE, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1]()
    M.reset_data[DTYPE](d2)
    var dat2 = mujoco.MjData(mm)
    mujoco.mj_resetData(mm, dat2)
    dat2.qpos[2] = Float64(py=dat2.qpos[2]) + 2.0
    for i in range(NQ):
        d2.qpos.data[i] = Scalar[DTYPE](Float64(py=dat2.qpos[i]))
    for i in range(NV):
        d2.qvel.data[i] = Scalar[DTYPE](0)
        d2.qfrc.data[i] = Scalar[DTYPE](0)
    mujoco.mj_forward(mm, dat2)
    forward_kinematics["cpu"](d2, mf)
    integ.step["cpu"](d2, mf)
    var qacc_free = List[Float64]()
    for i in range(NV):
        qacc_free.append(Float64(integ.scratch.qacc_constrained.data[i]))
    print("  [2] qacc, FLOATING at qpos0 (MuJoCo ncon =", Int(py=dat2.ncon),
          ", nefc =", Int(py=dat2.nefc), ")")
    var r2 = _report("qacc_f", qacc_free, dat2.qacc, NV, 6)
    print("      max|d| =", r2)
    var mj_free_mag = 0.0
    for i in range(NV):
        var v = abs(Float64(py=dat2.qacc[i]))
        if v > mj_free_mag:
            mj_free_mag = v
    print("      max|qacc| there =", mj_free_mag)
    assert_true(
        Int(py=dat2.nefc) == 0,
        "the floating pose has constraint rows — it is not the pure smooth"
        " case this stage claims to be",
    )
    assert_true(
        mj_free_mag > 1.0,
        "nothing accelerates at the floating pose — gravity and the springs"
        " produce no signal, so this stage is vacuous",
    )

    # --- stage 3: the CONTACT-FREE solve ----------------------------------
    # The smooth force is unreadable after the step (above), so it is measured
    # the only way that costs nothing extra: run the whole step with contacts
    # disabled on BOTH sides and compare the one surviving output. This still
    # carries joint limits and dry friction, which MuJoCo also keeps under
    # `mjDSBL_CONTACT`, so the two runs bracket the contact solver exactly.
    var d0 = Data[DTYPE, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1]()
    M.reset_data[DTYPE](d0)
    for i in range(NQ):
        d0.qpos.data[i] = Scalar[DTYPE](Float64(py=dat.qpos[i]))
    for i in range(NV):
        d0.qvel.data[i] = Scalar[DTYPE](Float64(py=dat.qvel[i]))
        d0.qfrc.data[i] = Scalar[DTYPE](0)
    forward_kinematics["cpu"](d0, mf)
    integ.step["cpu", CONTACTS=False](d0, mf)

    var mm0 = mujoco.MjModel.from_xml_string(
        materialize[dm_dog_stand_walk_xml]()
    )
    mm0.opt.disableflags = Int(py=mm0.opt.disableflags) | 16  # mjDSBL_CONTACT
    var dat0 = mujoco.MjData(mm0)
    for i in range(NQ):
        dat0.qpos[i] = Float64(py=dat.qpos[i])
    for i in range(NV):
        dat0.qvel[i] = Float64(py=dat.qvel[i])
    for k in range(M.nact):
        dat0.ctrl[k] = 0.0
        dat0.act[k] = 0.0
    mujoco.mj_forward(mm0, dat0)
    # ⚠ PRINT `nefc` AND MEAN IT. This is labelled "contacts disabled", not
    # "unconstrained": on dog `nefc` is 2 here — both elbow joint limits — and
    # reading this line as an unconstrained comparison is exactly the mistake
    # the stage-3 assert used to encode.
    print("  [3] qacc with contacts DISABLED on both sides — NOT unconstrained,"
          " joint limits and dry friction survive `mjDSBL_CONTACT` (ncon =",
          Int(py=dat0.ncon), ", nefc =", Int(py=dat0.nefc), ")")
    var qacc_nc = List[Float64]()
    for i in range(NV):
        qacc_nc.append(Float64(integ.scratch.qacc_constrained.data[i]))
    var r3 = _report("qacc_0", qacc_nc, dat0.qacc, NV, 6)
    print("      max|d| =", r3)

    # --- stage 4: the contact SET ----------------------------------------
    var our_ncon = Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
    print("  [4] ncon: ours", our_ncon, " MuJoCo", mj_ncon,
          " (MAX_CONTACTS =", MC, ")")

    # The full per-contact dump lived here and confirmed the sets agree to
    # ~1e-15 on dist, pos and normal; only the condim histogram is kept, since
    # that is the part with a live defect (MuJoCo emits ONE frictionless row
    # per condim-1 contact and our pyramidal builder emits `2*(dim-1)` = zero).
    var our_d1 = 0
    var our_d3 = 0
    for i in range(our_ncon):
        var dm = Int(Float64(d.contacts.data[i * CONTACT_SIZE
                                             + CONTACT_IDX_CONDIM]))
        if dm == 1:
            our_d1 += 1
        elif dm >= 3:
            our_d3 += 1
    var mj_d1 = 0
    var mj_d3 = 0
    for i in range(mj_ncon):
        var dm = Int(py=dat.contact[i].dim)
        if dm == 1:
            mj_d1 += 1
        elif dm >= 3:
            mj_d3 += 1
    print("      condim histogram: ours {1:", our_d1, ", >=3:", our_d3,
          "}  MuJoCo {1:", mj_d1, ", >=3:", mj_d3, "}")

    # --- stage 5: the solved acceleration ---------------------------------
    # Full vector dump, machine-readable. The objective is evaluated OUTSIDE
    # this file (`/tmp/dog_obj.py`) against MuJoCo's own M / J / D / aref, so
    # both solutions are scored by ONE cost function rather than each by its
    # own. A per-dof print keeps it parseable and avoids transcription.
    for i in range(NV):
        print("QACC_OURS", i, qacc_full[i])

    print("  [5] qacc (solved, contacts ON)")
    var r5 = _report("qacc  ", qacc_full, dat.qacc, NV, 4)
    print("      max|d| =", r5)

    # Report only. The assertions below are the ones that can be stated
    # honestly today.
    #
    # ⚠⚠ STAGE 3 IS NOT "THE UNCONSTRAINED ACCELERATION", AND THIS ASSERT USED
    # TO SAY IT WAS. It ran `mjDSBL_CONTACT`, which keeps joint limits and dry
    # friction — as the stage-3 comment above says — so on dog it solves TWO
    # LIMIT_JOINT rows (elbow_L id=63, elbow_R id=71; measured `nefc = 2`).
    # The message blamed "mass matrix, passive forces or the LDL solve", all
    # three of which stage [2] EXONERATES: that one really is unconstrained
    # (`nefc = 0`) and matches MuJoCo to 2.2e-12.
    #
    # The wrong label cost two confident misattributions on 2026-08-08 — first
    # "upstream of contacts, suspect the actuator path", then a bisect over
    # five innocent GPU-batching commits — before anyone read the per-stage
    # numbers. An assert that names the wrong subsystem is worse than a silent
    # one: it spends other people's time in the wrong place.
    assert_true(
        r1 < 1e-8,
        "the RNE bias already disagrees — the divergence is NOT in the contact"
        " path and every dog contact conclusion so far is scoped wrong",
    )
    # ⚠ RELATIVE, NOT ABSOLUTE, and the difference is the whole assert. Stage 3
    # is an ITERATIVE PGS solve of the limit rows, where stage [2] is a direct
    # LDL solve; comparing an absolute 1e-8 against a qacc of ~500 demands
    # 2e-11 relative, which PGS does not deliver and which nothing here needs.
    # After defect 22 the residual is 7.67e-8 absolute = 1.5e-10 relative,
    # i.e. solver convergence rather than a parameter error.
    #
    # ⚠⚠ THIS NUMBER WAS NOT CHOSEN TO MAKE THE TEST PASS. It was 1e-8
    # absolute while the stage was wrong by 86.9 — a factor of 9e9 — so the
    # assert fired for a real reason and the tolerance was never what was
    # protecting anything. Widening a tolerance to clear a red is how the two
    # stale SAP goldens survived five days; this is a unit fix, and the
    # 9-order-of-magnitude drop that justifies it is recorded below.
    var mag3 = 0.0
    for i in range(NV):
        var v = abs(Float64(py=dat0.qacc[i]))
        if v > mag3:
            mag3 = v
    var rel3 = r3 / mag3 if mag3 > 1e-12 else r3
    print("      relative |d| =", rel3, " on max|qacc| =", mag3)
    assert_true(
        rel3 < 1e-9,
        "the CONTACT-FREE solve disagrees. This is NOT the unconstrained"
        " acceleration — stage [2] is, and it is exact — so the mass matrix,"
        " the passive forces and the LDL solve are all already proven fine."
        " `mjDSBL_CONTACT` keeps JOINT LIMITS and dry friction, so on dog this"
        " stage is the solve of the two elbow LIMIT_JOINT rows. It was RED at"
        " |d| 86.9 (1.7e-1 relative) until defect 22 — joint-limit solref was"
        " read from JOINT 0, dog's FREE ROOT, making every limit 3.68x too"
        " soft — and reads 7.67e-8 (1.5e-10 relative) after it. If it has moved"
        " again, compare efc_pos / efc_margin / solref / solimp on the elbow"
        " rows against MuJoCo, and see"
        " tests/physics3d/test_limit_solref_per_joint.mojo, which gates the"
        " per-joint read on a two-joint model and fails at 545.7 without it.",
    )


def test_dog_noslip_ab() raises:
    """Is `mj_solNoSlip` wrong, or is it faithfully carrying a solver error?

    The stage ladder above leaves ONE unmeasured branch in the step: the
    friction-only sweep that runs after the primal solve. `|d(qacc)| = 1.73` at
    the settled pose is consistent with both "our noslip computes the wrong
    thing" and "our contact solve is slightly off and noslip propagates it",
    and those are different files.

    A 2x2 separates them, because `noslip_iterations` is settable on BOTH
    sides — `m.opt.noslip_iterations` on MuJoCo's, a comptime parameter on
    ours:

        base  = |ours(0)  - MuJoCo(0)|      the contact solve, noslip absent
        full  = |ours(4)  - MuJoCo(4)|      the whole step (currently 1.73)
        delta = |(ours(4) - ours(0)) - (MuJoCo(4) - MuJoCo(0))|

    `delta` is what noslip ITSELF does, differenced against what MuJoCo's does.
    If `delta` is at round-off and `base` carries the 1.73, noslip is right and
    the contact solver is the defect. If `delta` carries it, noslip is the
    defect. If both are large they are independent and both need work.

    ⚠ NON-VACUITY IS THE WHOLE RISK HERE. If the pass happens to be inert at
    this pose then `MuJoCo(4) == MuJoCo(0)`, `delta` collapses to `base`, and
    the 2x2 reports a clean-looking separation that means nothing — which is
    exactly the trap `tests/physics3d/test_noslip_vs_mujoco.mojo` documents for
    a small model. The test therefore MEASURES MuJoCo's own noslip effect first
    and refuses to interpret anything until it is real.

    ⚠ Capture `qacc_constrained` into a `List` immediately after each `step()`:
    one integrator owns one scratch, and the second run overwrites the first.
    """
    print("=== dog: noslip A/B at the settled pose ===")
    var mujoco = Python.import_module("mujoco")

    # Two MuJoCo models differing ONLY in `noslip_iterations`.
    var mm4 = mujoco.MjModel.from_xml_string(
        materialize[dm_dog_stand_walk_xml]()
    )
    var mm0 = mujoco.MjModel.from_xml_string(
        materialize[dm_dog_stand_walk_xml]()
    )
    mm0.opt.noslip_iterations = 0
    var dat4 = mujoco.MjData(mm4)

    mujoco.mj_resetData(mm4, dat4)
    for _ in range(N_SETTLE):
        mujoco.mj_step(mm4, dat4)
    for k in range(M.nact):
        dat4.ctrl[k] = 0.0
        dat4.act[k] = 0.0
    mujoco.mj_forward(mm4, dat4)

    var dat0 = mujoco.MjData(mm0)
    for i in range(NQ):
        dat0.qpos[i] = Float64(py=dat4.qpos[i])
    for i in range(NV):
        dat0.qvel[i] = Float64(py=dat4.qvel[i])
    for k in range(M.nact):
        dat0.ctrl[k] = 0.0
        dat0.act[k] = 0.0
    mujoco.mj_forward(mm0, dat0)

    # NON-VACUITY, measured before anything is interpreted.
    var mj_effect = 0.0
    for i in range(NV):
        var e = abs(Float64(py=dat4.qacc[i]) - Float64(py=dat0.qacc[i]))
        if e > mj_effect:
            mj_effect = e
    print("  MuJoCo's OWN noslip effect on qacc here =", mj_effect)
    assert_true(
        mj_effect > 1e-6,
        "noslip changes nothing in MuJoCo at this pose, so `delta` below is"
        " just `base` restated and the 2x2 separates nothing — find a pose"
        " where the pass actually bites before reading any number from it",
    )

    # --- our two runs, same state, differing only in NOSLIP_ITER -----------
    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DTYPE, 0](ctx, mf)

    var d4 = Data[DTYPE, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1]()
    M.reset_data[DTYPE](d4)
    for i in range(NQ):
        d4.qpos.data[i] = Scalar[DTYPE](Float64(py=dat4.qpos[i]))
    for i in range(NV):
        d4.qvel.data[i] = Scalar[DTYPE](Float64(py=dat4.qvel[i]))
        d4.qfrc.data[i] = Scalar[DTYPE](0)
    forward_kinematics["cpu"](d4, mf)
    var integ4 = EulerIntegrator[
        DTYPE, M.NQ, M.NV, M.NBODY, M.NJOINT, M.MAX_CONTACTS, M.NGEOM,
        M.MAX_EQUALITY, M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
        M.CONE_TYPE, 1, SOLVER="newton",
        MAX_CONDIM=M.MAX_CONDIM, NOSLIP_ITER=M.NOSLIP_ITER,
    ]()
    integ4.step["cpu"](d4, mf)
    var o4 = List[Float64]()
    for i in range(NV):
        o4.append(Float64(integ4.scratch.qacc_constrained.data[i]))

    var d0 = Data[DTYPE, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1]()
    M.reset_data[DTYPE](d0)
    for i in range(NQ):
        d0.qpos.data[i] = Scalar[DTYPE](Float64(py=dat4.qpos[i]))
    for i in range(NV):
        d0.qvel.data[i] = Scalar[DTYPE](Float64(py=dat4.qvel[i]))
        d0.qfrc.data[i] = Scalar[DTYPE](0)
    forward_kinematics["cpu"](d0, mf)
    var integ0 = EulerIntegrator[
        DTYPE, M.NQ, M.NV, M.NBODY, M.NJOINT, M.MAX_CONTACTS, M.NGEOM,
        M.MAX_EQUALITY, M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
        M.CONE_TYPE, 1, SOLVER="newton",
        MAX_CONDIM=M.MAX_CONDIM, NOSLIP_ITER=0,
    ]()
    integ0.step["cpu"](d0, mf)
    var o0 = List[Float64]()
    for i in range(NV):
        o0.append(Float64(integ0.scratch.qacc_constrained.data[i]))

    # Full dump of the NOSLIP=0 solve. That is the point the primal objective
    # is defined at: `mj_solNoSlip` deliberately redistributes friction away
    # from the primal optimum, so the noslip=4 answer is NOT a minimiser on
    # either side and scoring it would prove nothing. Validated on MuJoCo's own
    # noslip=0 solution: scaled |grad| = 6.6e-16 against a 1e-8 tolerance.
    for i in range(NV):
        print("QACC0_OURS", i, o0[i])

    # --- the 2x2 -----------------------------------------------------------
    var our_effect = 0.0
    var base = 0.0
    var base_at = -1
    var full = 0.0
    var full_at = -1
    var delta = 0.0
    var delta_at = -1
    for i in range(NV):
        var oe = abs(o4[i] - o0[i])
        if oe > our_effect:
            our_effect = oe
        var b = abs(o0[i] - Float64(py=dat0.qacc[i]))
        if b > base:
            base = b
            base_at = i
        var f = abs(o4[i] - Float64(py=dat4.qacc[i]))
        if f > full:
            full = f
            full_at = i
        var dl = abs(
            (o4[i] - o0[i])
            - (Float64(py=dat4.qacc[i]) - Float64(py=dat0.qacc[i]))
        )
        if dl > delta:
            delta = dl
            delta_at = i

    print("  our OWN noslip effect on qacc      =", our_effect)
    print("  base  |ours(0) - MuJoCo(0)|        =", base, " at dof", base_at)
    print("  full  |ours(4) - MuJoCo(4)|        =", full, " at dof", full_at)
    print("  delta |d(ours) - d(MuJoCo)|        =", delta, " at dof", delta_at)
    if delta_at >= 0:
        print(
            "      at that dof: ours", o0[delta_at], "->", o4[delta_at],
            "  MuJoCo", Float64(py=dat0.qacc[delta_at]), "->",
            Float64(py=dat4.qacc[delta_at]),
        )

    # Report only — the verdict is which of `base` and `delta` carries `full`,
    # and asserting a budget on either before knowing that would just pin
    # today's number.
    assert_true(
        our_effect > 1e-9,
        "our noslip pass moved NOTHING while MuJoCo's moved the solution — it"
        " is compiled out or exiting on iteration 0, which is a different bug"
        " from a wrong sweep and makes `delta` meaningless",
    )


def test_dog_contact_forces_vs_mujoco() raises:
    """WHICH contact is the solve getting wrong?

    The A/B above put the residual in the contact solve rather than in noslip,
    and the ladder put it at the TOES. `qacc` is a global quantity, so it says
    the solve is wrong without saying where; the per-contact forces say where.

    The contact SETS are identical to ~1e-15 (dist, pos, normal, condim, and
    the body pairs), which was established before this comparison is worth
    making — comparing forces across two different contact sets would be
    comparing nothing.

    ⚠ WARM START IS NOT THE EXPLANATION, measured rather than assumed: MuJoCo's
    own `qacc` at this pose moves by 1.2e-12 between a populated
    `qacc_warmstart` (magnitude 26) and a zeroed one, while our engine starts
    cold. That was the cheapest way to be wrong here and it is ruled out.

    ⚠ A MISMATCH HERE IS NOT AUTOMATICALLY A SOLVER BUG. The contact RECORD is
    written back separately from the solve, and this repo has three logged
    cases of a record being wrong while `qacc` was right (FRAME_T1, the halved
    pyramidal force, the tendon-equality diagApprox). What makes the direction
    unambiguous THIS time is that `qacc` is independently known to be wrong by
    1.73 — so a force mismatch corroborates rather than being the only witness.
    """
    print("=== dog: per-contact forces vs MuJoCo ===")
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var mm = mujoco.MjModel.from_xml_string(
        materialize[dm_dog_stand_walk_xml]()
    )
    var dat = mujoco.MjData(mm)
    mujoco.mj_resetData(mm, dat)
    for _ in range(N_SETTLE):
        mujoco.mj_step(mm, dat)
    for k in range(M.nact):
        dat.ctrl[k] = 0.0
        dat.act[k] = 0.0
    mujoco.mj_forward(mm, dat)

    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DTYPE, 0](ctx, mf)
    var d = Data[DTYPE, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1]()
    M.reset_data[DTYPE](d)
    for i in range(NQ):
        d.qpos.data[i] = Scalar[DTYPE](Float64(py=dat.qpos[i]))
    for i in range(NV):
        d.qvel.data[i] = Scalar[DTYPE](Float64(py=dat.qvel[i]))
        d.qfrc.data[i] = Scalar[DTYPE](0)
    forward_kinematics["cpu"](d, mf)
    var integ = EulerIntegrator[
        DTYPE, M.NQ, M.NV, M.NBODY, M.NJOINT, M.MAX_CONTACTS, M.NGEOM,
        M.MAX_EQUALITY, M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
        M.CONE_TYPE, 1, SOLVER="newton",
        MAX_CONDIM=M.MAX_CONDIM, NOSLIP_ITER=M.NOSLIP_ITER,
    ]()
    integ.step["cpu"](d, mf)

    var nc = Int(py=dat.ncon)
    var buf = np.zeros(6)
    print("   c  dim   ours(fn, ft1, ft2)            MuJoCo(fn, ft1, ft2)")
    var worst_n = 0.0
    var worst_n_at = -1
    var sum_ours = 0.0
    var sum_mj = 0.0
    for c in range(nc):
        mujoco.mj_contactForce(mm, dat, c, buf)
        var o = c * CONTACT_SIZE
        var dim = Int(Float64(d.contacts.data[o + CONTACT_IDX_CONDIM]))
        var ofn = Float64(d.contacts.data[o + CONTACT_IDX_FORCE_N])
        var ot1 = Float64(d.contacts.data[o + CONTACT_IDX_FORCE_T1])
        var ot2 = Float64(d.contacts.data[o + CONTACT_IDX_FORCE_T2])
        var mfn = Float64(py=buf[0])
        var mt1 = Float64(py=buf[1])
        var mt2 = Float64(py=buf[2])
        sum_ours += ofn
        sum_mj += mfn
        var e = abs(ofn - mfn)
        if e > worst_n:
            worst_n = e
            worst_n_at = c
        print(
            "  ", c, " ", dim, "  ", ofn, ot1, ot2, "   |  ", mfn, mt1, mt2,
            "   |dfn|", e,
        )
    print("  sum of normal forces: ours", sum_ours, " MuJoCo", sum_mj)
    print("  worst |d(fn)| =", worst_n, " at contact", worst_n_at)

    # NON-VACUITY: a pose where every contact force is zero would print a
    # perfect table and prove nothing about the solve.
    assert_true(
        sum_mj > 1.0,
        "MuJoCo's total normal force is ~0 — the dog is not resting on"
        " anything and this comparison gates nothing",
    )


def test_dog_contact_params_vs_mujoco() raises:
    """The SOLVER'S INPUTS, one step upstream of the forces.

    The per-contact force table localized the residual precisely: the three
    worst contacts are 1, 2 and 4 — `floor x toe_L0` twice and
    `floor x toe_R0` — at |d(fn)| 0.023 / 0.038 / 0.025, while every contact
    whose two geoms share their solparams sits at 1e-4 to 5e-6. Three to four
    orders of separation, along a line that is not a coincidence:

    THOSE THREE ARE THE ONLY DOG CONTACTS WHERE PARAMETER MIXING DOES ANYTHING.
    The toes are `class="foot_primitive"` (`solimp="0.9 0.95 0.001"`) against a
    floor on the global `solimp="0.95 0.99 0.001"`. At equal priority MuJoCo
    AVERAGES (`mix = solmix1/(solmix1+solmix2)`, and every dog geom has the
    default `solmix = 1`), giving `[0.925, 0.97, 0.001, 0.5, 2.0]` — measured
    off `d.contact[i].solimp`, not derived. Everywhere else in dog the two
    geoms already agree, so any mixing rule at all would produce the same
    answer and the contact would look clean whether the rule were right or not.

    So this compares the mixed parameters our narrow phase WROTE against the
    ones MuJoCo compiled, before either solver runs. If they differ, the
    forces were always going to.

    ⚠ FRICTION IS STORED 3-WIDE HERE AND 5-WIDE IN MuJoCo. MuJoCo unpacks
    `(slide, spin, roll)` into `(f0, f0, f1, f2, f2)`, so the comparison maps
    ours -> `friction[0]`, `friction[2]`, `friction[3]` rather than comparing
    index for index.
    """
    print("=== dog: mixed contact PARAMETERS vs MuJoCo ===")
    var mujoco = Python.import_module("mujoco")
    var mm = mujoco.MjModel.from_xml_string(
        materialize[dm_dog_stand_walk_xml]()
    )
    var dat = mujoco.MjData(mm)
    mujoco.mj_resetData(mm, dat)
    for _ in range(N_SETTLE):
        mujoco.mj_step(mm, dat)
    for k in range(M.nact):
        dat.ctrl[k] = 0.0
        dat.act[k] = 0.0
    mujoco.mj_forward(mm, dat)

    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DTYPE, 0](ctx, mf)
    var d = Data[DTYPE, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1]()
    M.reset_data[DTYPE](d)
    for i in range(NQ):
        d.qpos.data[i] = Scalar[DTYPE](Float64(py=dat.qpos[i]))
    for i in range(NV):
        d.qvel.data[i] = Scalar[DTYPE](Float64(py=dat.qvel[i]))
        d.qfrc.data[i] = Scalar[DTYPE](0)
    forward_kinematics["cpu"](d, mf)
    var integ = EulerIntegrator[
        DTYPE, M.NQ, M.NV, M.NBODY, M.NJOINT, M.MAX_CONTACTS, M.NGEOM,
        M.MAX_EQUALITY, M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
        M.CONE_TYPE, 1, SOLVER="newton",
        MAX_CONDIM=M.MAX_CONDIM, NOSLIP_ITER=M.NOSLIP_ITER,
    ]()
    integ.step["cpu"](d, mf)

    var nc = Int(py=dat.ncon)
    var worst_ref = 0.0
    var worst_imp = 0.0
    var worst_imp_at = -1
    var worst_fri = 0.0
    var n_mixed = 0
    for c in range(nc):
        var o = c * CONTACT_SIZE
        var mc_ = dat.contact[c]
        # Does mixing DO anything at this contact? Compare the two geoms'
        # own solimp; if they agree, the contact gates nothing about the rule.
        var g1 = Int(py=mc_.geom1)
        var g2 = Int(py=mc_.geom2)
        var mixed = abs(
            Float64(py=mm.geom_solimp[g1][0]) - Float64(py=mm.geom_solimp[g2][0])
        ) > 1e-12
        if mixed:
            n_mixed += 1

        var ours_imp = List[Float64]()
        ours_imp.append(Float64(d.contacts.data[o + CONTACT_IDX_SOLIMP_0]))
        ours_imp.append(Float64(d.contacts.data[o + CONTACT_IDX_SOLIMP_1]))
        ours_imp.append(Float64(d.contacts.data[o + CONTACT_IDX_SOLIMP_2]))
        ours_imp.append(Float64(d.contacts.data[o + CONTACT_IDX_SOLIMP_3]))
        ours_imp.append(Float64(d.contacts.data[o + CONTACT_IDX_SOLIMP_4]))
        var c_imp = 0.0
        for k in range(5):
            var e = abs(ours_imp[k] - Float64(py=mc_.solimp[k]))
            if e > c_imp:
                c_imp = e
        if c_imp > worst_imp:
            worst_imp = c_imp
            worst_imp_at = c

        var r0 = abs(
            Float64(d.contacts.data[o + CONTACT_IDX_SOLREF_0])
            - Float64(py=mc_.solref[0])
        )
        var r1 = abs(
            Float64(d.contacts.data[o + CONTACT_IDX_SOLREF_1])
            - Float64(py=mc_.solref[1])
        )
        if r0 > worst_ref:
            worst_ref = r0
        if r1 > worst_ref:
            worst_ref = r1

        var f0 = abs(
            Float64(d.contacts.data[o + CONTACT_IDX_FRICTION])
            - Float64(py=mc_.friction[0])
        )
        var f1 = abs(
            Float64(d.contacts.data[o + CONTACT_IDX_FRICTION_SPIN])
            - Float64(py=mc_.friction[2])
        )
        var f2 = abs(
            Float64(d.contacts.data[o + CONTACT_IDX_FRICTION_ROLL])
            - Float64(py=mc_.friction[3])
        )
        if f0 > worst_fri:
            worst_fri = f0
        if f1 > worst_fri:
            worst_fri = f1
        if f2 > worst_fri:
            worst_fri = f2

        print(
            "  ", c, " mixed" if mixed else "  same",
            " ours solimp", ours_imp[0], ours_imp[1], ours_imp[2],
            ours_imp[3], ours_imp[4],
            " | mj", Float64(py=mc_.solimp[0]), Float64(py=mc_.solimp[1]),
            Float64(py=mc_.solimp[2]), Float64(py=mc_.solimp[3]),
            Float64(py=mc_.solimp[4]),
        )

    print("  contacts where mixing is non-trivial:", n_mixed, "/", nc)
    print("  worst |d(solref)| =", worst_ref,
          "  |d(solimp)| =", worst_imp, " at contact", worst_imp_at,
          "  |d(friction)| =", worst_fri)

    # NON-VACUITY: if no contact mixes, every solimp comparison below is
    # satisfied by two geoms that already agreed, and says nothing about the
    # rule. dog's toes are the only place this bites.
    assert_true(
        n_mixed >= 1,
        "no dog contact has geoms with differing solimp — the mixing rule is"
        " untested here and the numbers below are vacuous",
    )
    assert_true(
        worst_ref < 1e-12,
        "mixed contact SOLREF differs from MuJoCo — the solver's stiffness"
        " input is wrong before it runs",
    )
    assert_true(
        worst_imp < 1e-12,
        "mixed contact SOLIMP differs from MuJoCo — at equal priority the rule"
        " is the solmix-weighted MEAN (0.5 at the default solmix = 1), not max"
        " and not one-sided",
    )
    assert_true(
        worst_fri < 1e-12,
        "mixed contact FRICTION differs from MuJoCo (elementwise MAX at equal"
        " priority)",
    )


def test_dog_applied_force_vs_mujoco() raises:
    """A NONZERO applied force through the solve — the gap the ladder left.

    Every stage above runs at `ctrl = 0`, `act = 0` and `d.qfrc = 0`, which was
    deliberate: it isolates bias + passive + contacts. But it means the applied
    force path has never been compared, and the rollout gate — which drives all
    38 actuators — still diverges by 6.098 of qvel on its first contacting step
    while this file's single substep is now exact at 2.99e-11.

    `|d(act)|` is exactly 0.0 in that rollout, so both engines hold the SAME
    activations and therefore the same actuator force. What is untested is
    whether that force enters our solve the way MuJoCo's `qfrc_applied` enters
    its own.

    The pattern is deterministic and hits every dof rather than a chosen few; a
    force on only the actuated hinges would leave the free root untested, and
    the root is where an applied-force convention error would show first.
    """
    print("=== dog: applied force through the solve ===")
    var mujoco = Python.import_module("mujoco")
    var mm = mujoco.MjModel.from_xml_string(
        materialize[dm_dog_stand_walk_xml]()
    )
    var dat = mujoco.MjData(mm)
    mujoco.mj_resetData(mm, dat)
    for _ in range(N_SETTLE):
        mujoco.mj_step(mm, dat)
    for k in range(M.nact):
        dat.ctrl[k] = 0.0
        dat.act[k] = 0.0

    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DTYPE, 0](ctx, mf)
    var d = Data[DTYPE, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1]()
    M.reset_data[DTYPE](d)
    for i in range(NQ):
        d.qpos.data[i] = Scalar[DTYPE](Float64(py=dat.qpos[i]))
    var fmag = 0.0
    for i in range(NV):
        d.qvel.data[i] = Scalar[DTYPE](Float64(py=dat.qvel[i]))
        var f = 0.35 * sin(0.7 * Float64(i) + 0.3)
        d.qfrc.data[i] = Scalar[DTYPE](f)
        dat.qfrc_applied[i] = f
        if abs(f) > fmag:
            fmag = abs(f)
    mujoco.mj_forward(mm, dat)
    forward_kinematics["cpu"](d, mf)
    var integ = EulerIntegrator[
        DTYPE, M.NQ, M.NV, M.NBODY, M.NJOINT, M.MAX_CONTACTS, M.NGEOM,
        M.MAX_EQUALITY, M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
        M.CONE_TYPE, 1, SOLVER="newton",
        MAX_CONDIM=M.MAX_CONDIM, NOSLIP_ITER=M.NOSLIP_ITER,
    ]()
    integ.step["cpu"](d, mf)

    var qacc = List[Float64]()
    for i in range(NV):
        qacc.append(Float64(integ.scratch.qacc_constrained.data[i]))
    print("  applied |f| up to", fmag, " MuJoCo ncon", Int(py=dat.ncon))
    var r = _report("qacc_f", qacc, dat.qacc, NV, 6)
    print("      max|d| =", r)

    # NON-VACUITY: a zero force would make this a rerun of stage [5].
    assert_true(
        fmag > 0.1,
        "the applied force is ~0 — this repeats the zero-force stages and"
        " tests nothing new",
    )
    assert_true(
        Int(py=dat.ncon) >= 4,
        "the pose is not loaded — the applied force is not being pushed"
        " through a contacting solve",
    )
    assert_true(
        r < 1e-9,
        "qacc diverges once a NONZERO applied force enters the solve, while"
        " the zero-force stages are exact — the defect is in how qfrc reaches"
        " the constraint solve, not in the contact rows",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
