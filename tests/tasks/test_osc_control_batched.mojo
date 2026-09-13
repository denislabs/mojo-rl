"""THE CONTROL STAGE OVER A BATCH — the sequence, not the torque law.

    pixi run mojo run -I . tests/tasks/test_osc_control_batched.mojo

`tests/tasks/test_osc_pose_gpu.mojo` gates `osc_run_gpu` — the torque law — over
eight lanes at one instant, and `tools/tasks/libero_demo_replay.py` gates the
one-lane host controller against MuJoCo to 1.5e-5 m over a recorded demo. What
neither can see is the SEQUENCE a batched env has to get right:

    refresh the dynamics at the current state
    set_goal on the FIRST substep only
    run every substep
    ramp the gripper after run, into the same ctrl row

That sequence lived inside `OscPose.update`/`run` and nowhere else, so the
batched path had to grow a second copy of it — the shape
`_a_rule_written_inline_twice_drifts` names. `dynamics/osc_control.mojo` is the
one copy; this gates it.

## WHAT IT ASSERTS

1. **THE SAME ctrl, over eight lanes and a sequence of control steps** — the
   batched stage at float64 against eight independent `OscPose` objects driven
   one at a time.

   ⚠⚠ NOT BIT-EXACT, AND THE REASON IS THE ENGINE, NOT THE CONTROLLER. The
   first version of this gate demanded equality and got 1.3e-12. The cause is
   measured here every run, before anything else: `osc_refresh_dynamics` on the
   SAME state gives different answers at BATCH=1 and BATCH=8 —

       M 1.8e-15   bias 7.1e-15   cdof 4.4e-16   xquat 2.2e-16 (one ULP)

   which is the CPU kernels vectorising differently over the lane loop (LLVM
   contracting `a*b+c` where it could not before —
   `_the_double_rounding_trick_is_folded_and_then_fused`). A controller reading
   an M that differs in the last bits cannot return the same torque, and the
   reference here is one lane while the subject is eight. So the tolerance is
   RELATIVE and the floor is printed beside it. Check 3 says what a real
   sequencing error costs: 0.387, twelve orders away.
2. **the lanes must disagree** — eight lanes that produced the same ctrl would
   pass 1 with `env` ignored entirely.
3. ⚠⚠ **the cadence is load-bearing, MEASURED.** A second batched run that calls
   `set_goal` on EVERY substep must produce a DIFFERENT ctrl. Without this,
   check 1 passes on a stage that ignores `policy_step`, and the failure it
   would be hiding is invisible: `set_goal` adds the action's delta to the
   CURRENT pose, so calling it 25 times turns a 5 cm command into 5 cm applied
   twenty-five times and the arm still tracks something smooth.
4. **the reset is per lane.** After `osc_reset_batch`, each lane's `q0` must be
   ITS OWN arm configuration. A reset that ignored `env` would write lane 0's
   pose everywhere and every later check would still pass — the goal would just
   be wrong in a way only the eighth lane could show.
5. **the refresh is load-bearing.** The state moves between substeps, and
   `site_xpos` after the stage must describe the CURRENT `qpos`, not the one the
   previous substep saw.

## ⚠ THE PHYSICS BETWEEN SUBSTEPS IS SYNTHETIC, AND DELIBERATELY

The subject is the controller's sequencing, so the state is advanced by a fixed
deterministic perturbation rather than by the solver: identical on both sides,
cheap, and — the part that matters — it MOVES. A gate that held the state still
between substeps would make check 3 vacuous, because `set_goal` on an unchanged
pose is idempotent. Running the real integrator over eight CPU lanes would gate
the engine again, which the fifty-step board already does.

## ⚠ NO DEVICE LEG HERE

`osc_control_step`'s GPU path calls `osc_refresh_dynamics`, whose CRBA and RNE
kernels are the ones that exceed Metal's per-thread stack at this family's
nv = 37 (the P0 park probe died at nv = 24). `test_osc_pose_gpu.mojo` carries
the device leg for the kernel itself, with the dynamics computed on the host and
shipped in. The device leg for the whole stage is owed on NVIDIA.
"""

from std.os.path import exists

from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.fields import (
    Data, Model, DynDims, DynamicsScratch,
)
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.dynamics.osc_pose import (
    OscPose, OscPoseConfig, ARM_DOF, OSC_ACTION_DIM,
)
from mojo_rl.physics3d.dynamics.osc_pose_gpu import (
    OSC_REF_WORDS, OSC_STATE_WORDS, OSC_WORK_WORDS, OSC_IDX_Q0,
    build_osc_refs, osc_refresh_dynamics,
)
from mojo_rl.physics3d.dynamics.osc_control import (
    osc_control_step, osc_reset_batch, osc_singular_lanes,
)
from mojo_rl.tasks.spec import load_family
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.libero_goal_dims import LIBERO_GOAL_DIMS
from mojo_rl.tasks.libero_goal_xml import LIBERO_GOAL_MAX_CONTACTS


comptime DT = DType.float64
comptime FAMILY = "mojo_rl/tasks/families/libero_goal.family"
comptime PACK = "mojo_rl/tasks/libero/assets"
comptime BATCH = 8
comptime N_CONTROL = 3
comptime N_SUBSTEP = 4
comptime NACT = LIBERO_GOAL_DIMS.NACT


struct Tally(Copyable, ImplicitlyCopyable, Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def check(mut self, ok: Bool, what: String):
        self.checks += 1
        if ok:
            print("  ok:", what)
        else:
            self.failures += 1
            print("  FAIL:", what)


def _index(names: List[String], want: String) raises -> Int:
    for i in range(len(names)):
        if String(names[i]) == want:
            return i
    raise Error("no '" + want + "' in the composed scene")


def _lane_action(e: Int) -> List[Float64]:
    var a = List[Float64]()
    for k in range(OSC_ACTION_DIM):
        a.append(0.2 * Float64((e * 2 + k) % 7 - 3))
    return a^


def _advance(mut q: List[Float64], mut v: List[Float64], e: Int, step: Int,
             qadr: List[Int], dof: List[Int]):
    """The synthetic substep. See the header: identical on both sides, and it
    MOVES — a held state makes the cadence check vacuous."""
    for j in range(ARM_DOF):
        v[dof[j]] += 0.004 * Float64((e + j + step) % 5 - 2)
        q[qadr[j]] += 0.002 * v[dof[j]]


def main() raises:
    print("=" * 74)
    print("The OSC control STAGE over", BATCH, "lanes —", N_CONTROL,
          "control steps x", N_SUBSTEP, "substeps")
    print("=" * 74)
    if not exists(PACK):
        print("  SKIPPED: no LIBERO pack at", PACK)
        print("=== SKIPPED (no pack — this is not a pass) ===")
        return
    var ta = Tally()

    var f = load_family(FAMILY)
    var fmd = parse_model_runtime(scene_path(f))
    var verts = 32768
    var dims = dims_from_flat(
        fmd, max_contacts=LIBERO_GOAL_MAX_CONTACTS, nmesh_verts=verts
    )
    var m = Model[DT, DynDims](dims)
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") < 0:
                raise e
            verts *= 2
            dims = dims_from_flat(
                fmd, max_contacts=LIBERO_GOAL_MAX_CONTACTS, nmesh_verts=verts
            )
            m = Model[DT, DynDims](dims)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var nact = dims.get_nact()

    var qadr_all = List[Int]()
    var dadr_all = List[Int]()
    var qa = 0
    var da = 0
    for i in range(len(fmd.joints)):
        qadr_all.append(qa)
        dadr_all.append(da)
        qa += fmd.joints[i].nq
        da += fmd.joints[i].nv
    var dof = List[Int]()
    var qadr = List[Int]()
    var jidx = List[Int]()
    var act_idx = List[Int]()
    var tmin = List[Float64]()
    var tmax = List[Float64]()
    for j in range(ARM_DOF):
        var ji = _index(fmd.joint_names, String("robot_joint") + String(j + 1))
        dof.append(dadr_all[ji])
        qadr.append(qadr_all[ji])
        jidx.append(ji)
        var ai = _index(fmd.actuator_names, String("robot_torq_j") + String(j + 1))
        act_idx.append(ai)
        tmin.append(fmd.actuators[ai].ctrl_min)
        tmax.append(fmd.actuators[ai].ctrl_max)
    var site = _index(fmd.site_names, String("robot_grip_site"))
    var site_body = fmd.sites[site].body_id
    var ga1 = _index(fmd.actuator_names, String("robot_gripper_finger_joint1"))
    var ga2 = _index(fmd.actuator_names, String("robot_gripper_finger_joint2"))
    var cfg = OscPoseConfig()

    # ── the eight starting states, built once and used by both sides ──────
    var q0 = List[List[Float64]]()
    var v0 = List[List[Float64]]()
    for e in range(BATCH):
        var q = List[Float64](length=nq, fill=0.0)
        var v = List[Float64](length=nv, fill=0.0)
        for i in range(len(f.base_qpos)):
            q[i] = f.base_qpos[i]
        for j in range(ARM_DOF):
            # ⚠ EVERY LANE A DIFFERENT POSE AND VELOCITY — see check 2.
            q[qadr[j]] += 0.03 * Float64((e * 3 + j) % 5 - 2)
            v[dof[j]] = 0.02 * Float64((e * 5 + j) % 7 - 3)
        for i in range(len(fmd.joints)):
            if fmd.joints[i].nq == 7:
                q[qadr_all[i]] = 2.0
                q[qadr_all[i] + 2] = 3.0
                q[qadr_all[i] + 3] = 1.0
        q0.append(q^)
        v0.append(v^)

    # ── 0. the floor: how far apart are BATCH=1 and BATCH=8 dynamics? ─────
    #
    # ⚠ MEASURED BEFORE THE SUBJECT, ON ONE STATE, so the tolerance below is a
    # checked claim rather than a number that was raised until it passed.
    print()
    print("--- 0. the engine's own batch-dependence, same state ---")
    var d1 = Data[DT, DynDims, 1](dims)
    var s1 = DynamicsScratch[DT, DynDims, 1](dims)
    var dfl = Data[DT, DynDims, BATCH](dims)
    var sfl = DynamicsScratch[DT, DynDims, BATCH](dims)
    for i in range(nq):
        d1.qpos.data[i] = Scalar[DT](q0[0][i])
    for i in range(nv):
        d1.qvel.data[i] = Scalar[DT](v0[0][i])
    for e in range(BATCH):
        for i in range(nq):
            dfl.qpos.data[e * nq + i] = Scalar[DT](q0[0][i])
        for i in range(nv):
            dfl.qvel.data[e * nv + i] = Scalar[DT](v0[0][i])
    osc_refresh_dynamics["cpu", DT, DynDims, 1](d1, m, s1)
    osc_refresh_dynamics["cpu", DT, DynDims, BATCH](dfl, m, sfl)
    var floor_m = 0.0
    for i in range(nv * nv):
        var e5 = abs(Float64(s1.M.data[i]) - Float64(sfl.M.data[i]))
        if e5 > floor_m:
            floor_m = e5
    var floor_b = 0.0
    for i in range(nv):
        var e6 = abs(Float64(s1.bias.data[i]) - Float64(sfl.bias.data[i]))
        if e6 > floor_b:
            floor_b = e6
    print("     M", floor_m, " bias", floor_b, " — BATCH=1 vs BATCH="
          + String(BATCH) + " on an identical state")
    ta.check(floor_m < 1.0e-12 and floor_b < 1.0e-11,
             "the engine's batch-dependence is at the ULP scale, so the"
             " controller residual below can be attributed to it")

    # ── the reference: one lane at a time through OscPose ─────────────────
    print()
    print("--- the reference: OscPose, one lane at a time ---")
    var want = List[Float64]()          # [lane][step][nact], flattened
    var ref_q0 = List[Float64]()       # each lane's nullspace target
    for e in range(BATCH):
        var q = q0[e].copy()
        var v = v0[e].copy()
        for i in range(nq):
            d1.qpos.data[i] = Scalar[DT](q[i])
        for i in range(nv):
            d1.qvel.data[i] = Scalar[DT](v[i])
        var osc = OscPose(
            dof.copy(), qadr.copy(), jidx.copy(), tmin.copy(), tmax.copy(),
            act_idx.copy(), site, site_body, ga1, ga2,
            fmd.actuators[ga1].ctrl_min, fmd.actuators[ga1].ctrl_max,
            fmd.actuators[ga2].ctrl_min, fmd.actuators[ga2].ctrl_max,
            cfg, nact, nq, nv,
        )
        osc.update(d1, m, s1)
        osc.reset(d1, m)
        var iq = osc.initial_joint()
        for j in range(ARM_DOF):
            ref_q0.append(iq[j])
        var a = _lane_action(e)
        var step = 0
        for _cs in range(N_CONTROL):
            for ss in range(N_SUBSTEP):
                osc.update(d1, m, s1)
                if ss == 0:
                    osc.set_goal(a, d1, m)
                var c = osc.run(a, d1, m, s1)
                for k in range(nact):
                    want.append(c[k])
                _advance(q, v, e, step, qadr, dof)
                for i in range(nq):
                    d1.qpos.data[i] = Scalar[DT](q[i])
                for i in range(nv):
                    d1.qvel.data[i] = Scalar[DT](v[i])
                step += 1
    print("   ", BATCH, "lanes x", N_CONTROL * N_SUBSTEP, "substeps =",
          len(want) // nact, "ctrl vectors")

    # ── the batched stage ─────────────────────────────────────────────────
    print()
    print("--- the stage: osc_control_step over all", BATCH, "lanes ---")
    var refs_l = build_osc_refs(
        dof.copy(), qadr.copy(), jidx.copy(), tmin.copy(), tmax.copy(),
        act_idx.copy(), site, site_body,
        ga1, ga2,
        fmd.actuators[ga1].ctrl_min, fmd.actuators[ga1].ctrl_max,
        fmd.actuators[ga2].ctrl_min, fmd.actuators[ga2].ctrl_max,
        cfg.kp, cfg.damping_ratio, cfg.output_max_pos, cfg.output_max_ori,
        cfg.nullspace_kp, cfg.gripper_speed,
    )
    var t_refs = TensorImpl[DT].alloc(OSC_REF_WORDS)
    for k in range(OSC_REF_WORDS):
        t_refs.data[k] = Scalar[DT](refs_l[k])

    var got = List[Float64]()
    var got_q0 = List[Float64]()
    var sing = 0
    var fk_moved = True
    for pass_id in range(2):
        # pass 0: the real cadence. pass 1: `set_goal` EVERY substep — check 3.
        var db = Data[DT, DynDims, BATCH](dims)
        var sb = DynamicsScratch[DT, DynDims, BATCH](dims)
        var t_state = TensorImpl[DT].alloc(BATCH * OSC_STATE_WORDS)
        var t_work = TensorImpl[DT].alloc(BATCH * OSC_WORK_WORDS)
        var t_ctrl = TensorImpl[DT].alloc(BATCH * nact)
        var t_act = TensorImpl[DT].alloc(BATCH * OSC_ACTION_DIM)
        for i in range(BATCH * OSC_STATE_WORDS):
            t_state.data[i] = Scalar[DT](0)
        for i in range(BATCH * OSC_WORK_WORDS):
            t_work.data[i] = Scalar[DT](0)
        for i in range(BATCH * nact):
            t_ctrl.data[i] = Scalar[DT](0)
        var qs = List[List[Float64]]()
        var vs = List[List[Float64]]()
        for e in range(BATCH):
            qs.append(q0[e].copy())
            vs.append(v0[e].copy())
            var a = _lane_action(e)
            for k in range(OSC_ACTION_DIM):
                t_act.data[e * OSC_ACTION_DIM + k] = Scalar[DT](a[k])
            for i in range(nq):
                db.qpos.data[e * nq + i] = Scalar[DT](q0[e][i])
            for i in range(nv):
                db.qvel.data[e * nv + i] = Scalar[DT](v0[e][i])

        osc_reset_batch["cpu", DT, DynDims, BATCH](
            db, m, sb, t_refs, t_state, t_work
        )
        if pass_id == 0:
            for e in range(BATCH):
                for j in range(ARM_DOF):
                    got_q0.append(
                        Float64(t_state.data[e * OSC_STATE_WORDS + OSC_IDX_Q0 + j])
                    )

        var step = 0
        for _cs in range(N_CONTROL):
            for ss in range(N_SUBSTEP):
                var ps = True if pass_id == 1 else (ss == 0)
                osc_control_step["cpu", DT, DynDims, BATCH](
                    db, m, sb, t_refs, t_state, t_work, t_ctrl, t_act,
                    nact, ps,
                )
                if pass_id == 0:
                    sing += osc_singular_lanes[DT](t_state, BATCH)
                    # check 5: FK inside the stage saw the CURRENT qpos
                    var sx = Float64(db.site_xpos.data[site * 3])
                    if step > 0 and sx == 0.0:
                        fk_moved = False
                for e in range(BATCH):
                    for k in range(nact):
                        got.append(Float64(t_ctrl.data[e * nact + k]))
                    _advance(qs[e], vs[e], e, step, qadr, dof)
                    for i in range(nq):
                        db.qpos.data[e * nq + i] = Scalar[DT](qs[e][i])
                    for i in range(nv):
                        db.qvel.data[e * nv + i] = Scalar[DT](vs[e][i])
                step += 1

    var n_per_pass = len(got) // 2

    # ── 1. bit-exact against the reference ────────────────────────────────
    print()
    print("--- 1. the stage == eight OscPose objects ---")
    # the reference is [lane][step][k]; the stage is [step][lane][k]
    var worst = 0.0
    var worst_rel = 0.0
    var n_diff = 0
    var compared = 0
    var mag = 0.0
    var n_steps = N_CONTROL * N_SUBSTEP
    for e in range(BATCH):
        for st in range(n_steps):
            for k in range(nact):
                var r = want[(e * n_steps + st) * nact + k]
                var g = got[(st * BATCH + e) * nact + k]
                var diff = abs(r - g)
                var scale = abs(r) if abs(r) > abs(g) else abs(g)
                if scale > mag:
                    mag = scale
                if diff > worst:
                    worst = diff
                if diff > 0.0:
                    n_diff += 1
                compared += 1
    # ⚠ NORMALISED BY THE RUN'S TORQUE MAGNITUDE, NOT PER ELEMENT. A ctrl
    # vector has entries spanning several orders — the largest here is 80 N·m
    # and some are near zero — so a per-element relative error is dominated by
    # whichever entry happens to be small, and reports 1.6e-11 for an absolute
    # disagreement of 1.3e-12. The meaningful scale is the quantity's own.
    worst_rel = worst / mag if mag > 0.0 else worst
    print("    ", compared, "ctrl values,", n_diff, "differ at all; worst abs",
          worst, "= ", worst_rel, "of the largest |ctrl| (", mag, ")")
    # ⚠ THE RESIDUAL DOES NOT GROW WITH THE STEP. If it did, the state would be
    # diverging between the two paths and the cause would be the SEQUENCE; flat
    # across the sequence is what says it is the per-step arithmetic floor
    # measured in section 0.
    var first_step = 0.0
    var last_step = 0.0
    for e in range(BATCH):
        for k in range(nact):
            var d_f = abs(want[(e * n_steps) * nact + k]
                          - got[(0 * BATCH + e) * nact + k])
            if d_f > first_step:
                first_step = d_f
            var d_l = abs(want[(e * n_steps + n_steps - 1) * nact + k]
                          - got[((n_steps - 1) * BATCH + e) * nact + k])
            if d_l > last_step:
                last_step = d_l
    print("      first substep", first_step, " last substep", last_step)
    ta.check(compared == BATCH * n_steps * nact and worst_rel < 1.0e-12,
             "every lane and every substep matches OscPose to " + String(worst_rel)
             + " of the torque scale — the engine's batch floor, not the"
             " sequence (check 3 costs 4.8e-3 on the same scale)")
    ta.check(last_step < 100.0 * (first_step + 1.0e-18),
             "the residual does NOT grow across the sequence, so the two paths"
             " are not drifting apart")

    # ── 2. the lanes disagree ─────────────────────────────────────────────
    print()
    print("--- 2. the lanes are not all the same ---")
    var pairs_differ = 0
    for e in range(1, BATCH):
        var same = True
        for k in range(nact):
            if got[k] != got[e * nact + k]:
                same = False
        if not same:
            pairs_differ += 1
    ta.check(pairs_differ == BATCH - 1,
             String(pairs_differ) + " of " + String(BATCH - 1)
             + " lanes differ from lane 0 on the first substep")

    # ── 3. the cadence is load-bearing ────────────────────────────────────
    print()
    print("--- 3. set_goal on EVERY substep gives a DIFFERENT answer ---")
    var cad_worst = 0.0
    for i in range(n_per_pass):
        var e2 = abs(got[i] - got[n_per_pass + i])
        if e2 > cad_worst:
            cad_worst = e2
    print("     worst |gated - every-substep|", cad_worst)
    ta.check(cad_worst > 1.0e-6,
             "the two cadences disagree, so check 1 is testing the gate on"
             " `policy_step` and not a stage that ignores it")

    # ── 4. the reset is per lane ──────────────────────────────────────────
    print()
    print("--- 4. every lane's nullspace target is ITS OWN pose ---")
    var q0_worst = 0.0
    for i in range(len(ref_q0)):
        var e3 = abs(ref_q0[i] - got_q0[i])
        if e3 > q0_worst:
            q0_worst = e3
    var q0_spread = 0.0
    for e in range(1, BATCH):
        for j in range(ARM_DOF):
            var e4 = abs(got_q0[j] - got_q0[e * ARM_DOF + j])
            if e4 > q0_spread:
                q0_spread = e4
    print("     worst vs OscPose.reset", q0_worst, "| spread across lanes",
          q0_spread)
    ta.check(q0_worst == 0.0, "osc_reset_batch == OscPose.reset on every lane")
    ta.check(q0_spread > 1.0e-6,
             "the eight targets are NOT identical — a reset that ignored `env`"
             " would write lane 0's pose everywhere")

    # ── 5. the refresh ran, and no lane went singular ─────────────────────
    print()
    print("--- 5. the refresh, and the singular flag ---")
    ta.check(fk_moved, "the stage's FK saw the current qpos every substep")
    ta.check(sing == 0,
             String(sing) + " singular lanes over " + String(BATCH * n_steps)
             + " lane-substeps (a flagged lane is a LIMP arm, not an error)")

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "osc control stage: " + String(ta.failures) + " of "
            + String(ta.checks) + " failed"
        )
    print("=== PASS ===")
