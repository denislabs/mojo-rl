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

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from std.gpu.host import DeviceContext

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
    print("  [3] qacc with contacts DISABLED on both sides (ncon =",
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
    print("  [5] qacc (solved, contacts ON)")
    var r5 = _report("qacc  ", qacc_full, dat.qacc, NV, 4)
    print("      max|d| =", r5)

    # Report only. The assertions below are the ones that can be stated
    # honestly today: stages 1-3 are contact-free smooth dynamics and are
    # gated elsewhere, so a miss there would relocate the bug entirely.
    assert_true(
        r1 < 1e-8,
        "the RNE bias already disagrees — the divergence is NOT in the contact"
        " path and every dog contact conclusion so far is scoped wrong",
    )
    assert_true(
        r3 < 1e-8,
        "the UNCONSTRAINED acceleration disagrees — mass matrix, passive"
        " forces or the LDL solve, not the contact solver",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
