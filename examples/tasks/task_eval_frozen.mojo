"""TWO RUNS, ONE FROZEN TABLE, THE SAME NUMBER — P4's gate.

    pixi run -e nvidia mojo run -I . examples/tasks/task_eval_frozen.mojo

⚠⚠ RUN THIS ON NVIDIA. The family is `nv = 24`, which is where the P0 park
probe died on Metal — the physics kernels stack-allocate per-thread arrays
sized by `nv` and pipeline creation fails with "Compute function exceeds
available stack space". It COMPILES on Apple and cannot launch.

`TASK_LAYER_PLAN.md` §6.2: a success rate over states the run sampled for
itself is not comparable with anything. This is the claim that it is.

## THE THREE RUNS, AND WHY THE THIRD IS THE POINT

    A   frozen table, reset seed 1     the reference
    B   frozen table, reset seed 2     must equal A
    C   NO table,     reset seed 2     must DIFFER from A

⚠⚠ **B AND C DIFFER IN ONE THING: WHETHER THE TABLE IS APPLIED.** Both pass a
different reset seed from A. If the eval quietly resampled instead of reading
the table, B would drift from A exactly as C does — so A == B is evidence the
table was used, not merely that physics is deterministic.

⚠ **C IS THE CONTROL AND IT IS NOT DECORATION.** Without it, "two runs agree"
is also what you get from an eval that ignores its seed entirely, and from one
where every lane is the same episode. C is that same eval with the one axis
flipped: `feedback_the_gates_name_named_the_wrong_axis` — add the fixed-axis
leg first.

## ⚠ THE POLICY IS ZERO ACTIONS, AND THAT IS A CHOICE THIS FILE OWNS

Nothing is trained on this family yet. A do-nothing baseline keeps the loop
cheap and deterministic, and the claim being gated is about the HARNESS — that
an init table makes two runs comparable — not about a policy's quality.

⚠ THE STATE COMPARISON IS THE PRIMARY CLAIM — final `qpos`, bit for bit —
because it holds whatever the policy is. `settle` holds from its first step
under zero actions and `reach` never does, so a family whose goals ignored
placement would report identical rates from completely different states.

⚠⚠ BUT THE OUTCOMES DO MOVE HERE, AND I EXPECTED THEY WOULD NOT. Measured on
a 5090: A vs C differ on **128 of 256 lanes** — every `settle` lane. Without
the table the props sit at the composed scene's XML defaults rather than on
the table, so `On(brick, table_top)` is False. The A-vs-C outcome difference
is therefore ASSERTED, not printed: it is a second control, independent of
the `qpos` one.

## ⚠⚠ THE SUCCESS SIGNAL IS `_reward`, NOT `_done`

`Phyics3dBatchedEnv` takes `TERMINATE_ON_UNHEALTHY` as a comptime parameter
**defaulting to False**, and then discards the config's `done` return
outright (`phyics3d_batched_env.mojo:1161`). So `_done` carries only
truncation at `MAX_STEPS`, and a 40-step loop over a 300-step horizon reads a
constant zero. The first version of this file read it and reported 0/128 on a
task that holds at reset.

⚠ AND THE GATE STILL PASSED, which is the part worth remembering: an all-False
outcome vector compares EQUAL to another all-False one, so `same_as` and every
rate check agreed perfectly about nothing. The `any_solved` check below exists
because of that run.

## WHAT IT PRINTS

Per-task success from `tasks/eval_report.SuccessReport`, which is what a run
sends to `mojo-rl-monitor`. Its arithmetic, the lane-wise comparison and the
metric keys are gated on the CPU by `tests/tasks/test_eval_report.mojo`; what
only this can say is that the numbers came out of a real batch.
"""

from std.random import seed as seed_rng
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.core.logger import CsvLogger
from mojo_rl.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_TASK_PARAM_0, META_IDX_TASK_ACTIVE,
    MODEL_CURRICULUM_SIZE,
)
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.physics3d.parser.runtime_load import parse_model_runtime

from mojo_rl.tasks.spec import (
    load_family, load_task, validate_task_against_family,
)
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.family_config import So101TabletopConfig
from mojo_rl.tasks.so101_tabletop_xml import So101TabletopModel
from mojo_rl.tasks.predicates import parse_goal, bind_goal, require_tier_a
from mojo_rl.tasks.eval import (
    region_sites, region_rects, region_half_heights,
)
from mojo_rl.tasks.tape import encode_goal, TAPE_WORDS
from mojo_rl.tasks.gpu_eval import region_table_words, require_gpu_regions
from mojo_rl.tasks.sampler import RegionFrame, SampleReport
from mojo_rl.tasks.reset import free_slot_addresses
from mojo_rl.tasks.init_table import (
    InitTable, append_init_rows, write_init_table, load_init_table,
)
from mojo_rl.tasks.eval_report import SuccessReport


comptime N_ENVS = 256
comptime N_PER_TASK = N_ENVS // 2
comptime EVAL_STEPS = 40

comptime TABLE_A = "/tmp/mojo_rl_eval_frozen_a.h5"
comptime CSV = "/tmp/mojo_rl_eval_frozen.csv"

comptime EnvT = Phyics3dBatchedEnv[
    So101TabletopModel, So101TabletopConfig, N_ENVS
]
comptime NQ = So101TabletopModel.NQ
comptime NV = So101TabletopModel.NV


def run_eval(
    mut env: EnvT,
    ctx: DeviceContext,
    tbl: InitTable,
    tape_g: List[Float64],
    tape_r: List[Float64],
    reset_seed: UInt64,
    use_table: Bool,
    mut solved: List[Bool],
    mut final_qpos: List[Float64],
) raises:
    """One eval run. `solved[e]` is "lane e met its goal at ANY step".

    ⚠ ONE BODY FOR ALL THREE RUNS, at module scope because a nested `def`
    cannot capture `env` mutably. Two copies of an eval loop is how the
    control ends up differing from the reference in a second, unnoticed way —
    and then it proves nothing about the table.
    """
    env.reset_batch[N_ENVS](ctx, reset_seed)
    env.d.qpos.download(ctx)
    ctx.synchronize()

    var qp = List[Float64]()
    for _ in range(NQ):
        qp.append(0.0)
    var qv = List[Float64]()
    for _ in range(NV):
        qv.append(0.0)
    for e in range(N_ENVS):
        var is_g = Int(tbl.task_index[e]) == 0
        for k in range(TAPE_WORDS):
            env.d.meta.data[
                e * METADATA_SIZE + META_IDX_TASK_PARAM_0 + k
            ] = Scalar[DT](tape_g[k]) if is_g else Scalar[DT](tape_r[k])
        env.d.meta.data[e * METADATA_SIZE + META_IDX_TASK_ACTIVE] = (
            Scalar[DT](tbl.mask[e])
        )
        if use_table:
            tbl.apply(e, qp, qv)
            for i in range(NQ):
                env.d.qpos.data[e * NQ + i] = Scalar[DT](qp[i])
            for i in range(NV):
                env.d.qvel.data[e * NV + i] = Scalar[DT](qv[i])
    env.d.meta.upload(ctx)
    if use_table:
        env.d.qpos.upload(ctx)
        env.d.qvel.upload(ctx)
    ctx.synchronize()

    # ⚠ ZERO ACTIONS, WRITTEN EXPLICITLY. `_action` is a device buffer whose
    # contents at this point are whatever the previous run left; relying on
    # "it starts zeroed" makes run B depend on run A.
    ctx.enqueue_memset(env._action, 0)

    solved = List[Bool]()
    for _ in range(N_ENVS):
        solved.append(False)
    # ⚠⚠ `_reward`, **NOT** `_done`. This read `_done` first and every lane
    # came back False — including `settle`, whose goal `test_task_reset_steps`
    # proves holds AT RESET. `Phyics3dBatchedEnv` takes
    # `TERMINATE_ON_UNHEALTHY` as a comptime parameter DEFAULTING TO FALSE,
    # and `phyics3d_batched_env.mojo:1161` then does
    #
    #     comptime if not Self.TERMINATE_ON_UNHEALTHY:
    #         is_terminated = False
    #
    # — so the CONFIG's `done` return is DISCARDED unless the env was
    # instantiated with that flag, and `_done` carries only truncation at
    # `MAX_STEPS`. Over 40 steps of a 300-step horizon it is a constant zero.
    #
    # ⚠ THE REWARD IS THE GOAL HERE, so `_reward > 0.5` is the same signal
    # P3's gate compares against the CPU evaluator, and it does not depend on
    # a flag this file does not set.
    var rew_h = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    for _ in range(EVAL_STEPS):
        env.step_batch[N_ENVS](ctx, UInt64(0))
        # ⚠ NO `selective_reset_batch`. This loop deliberately never resets a
        # done lane: the frozen init must survive the whole episode, and an
        # auto-reset would replace it with a fresh sample the moment a lane
        # succeeded.
        ctx.enqueue_copy(rew_h, env._reward)
        ctx.synchronize()
        var rp = rew_h.unsafe_ptr()
        for e in range(N_ENVS):
            if Float64(rp[e]) > 0.5:
                solved[e] = True

    env.d.qpos.download(ctx)
    ctx.synchronize()
    final_qpos = List[Float64]()
    for i in range(N_ENVS * NQ):
        final_qpos.append(Float64(env.d.qpos.data[i]))


def main() raises:
    seed_rng(0)
    print("=" * 68)
    print("P4 — two runs, one frozen init table, the same number")
    print("=" * 68)

    var f = load_family("mojo_rl/tasks/families/so101_tabletop.family")
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)
    var rheights = region_half_heights(f)

    # ⚠⚠ `settle`, NOT `gather`. Both controls below need a lane whose goal is
    # TRUE under this file's zero-action policy: `any_solved` would otherwise
    # be 0 (the failure this file already recorded once) and the A-vs-C
    # OUTCOME control has nothing to flip. `gather` used to fill that role by
    # accident — its goal held at reset — and is now a real task that scores 0
    # here. `so101_settle_brick.task` is the probe, on purpose.
    var tg = load_task("mojo_rl/tasks/tasks/so101_settle_brick.task")
    var tr = load_task("mojo_rl/tasks/tasks/so101_reach_brick.task")
    validate_task_against_family(tg, f)
    validate_task_against_family(tr, f)
    var gg = bind_goal(parse_goal(tg.goal), f, fmd.body_names, fmd.site_names)
    var gr = bind_goal(parse_goal(tr.goal), f, fmd.body_names, fmd.site_names)
    require_tier_a(gg, tg.name)
    require_tier_a(gr, tr.name)
    # ⚠ ONE region table on device, and a term's region index is ignored
    # there — see `gpu_eval.require_gpu_regions`. This family declares three
    # regions and two of them are for `init=` only.
    require_gpu_regions(gg, tg.name)
    require_gpu_regions(gr, tr.name)
    var tape_g = encode_goal(gg)
    var tape_r = encode_goal(gr)
    print("  task 0:", tg.name, "|", tg.goal)
    print("  task 1:", tr.name, "|", tr.goal)

    var jt = List[Int]()
    var jq = List[Int]()
    var jv = List[Int]()
    for i in range(len(fmd.joints)):
        jt.append(fmd.joints[i].jnt_type)
        jq.append(fmd.joints[i].nq)
        jv.append(fmd.joints[i].nv)
    var addrs = free_slot_addresses(f, fmd.joint_names, jt, jq, jv)

    with DeviceContext() as ctx:
        var env = EnvT(ctx)

        # ⚠ THE HALF-HEIGHT IS THE FIFTH NUMBER AND IT IS REQUIRED. Without
        # it the device would use `IN_HALF_HEIGHT` while `eval.eval_goal` used
        # the region's own band — a CPU/GPU disagreement inside the reward.
        var cw = region_table_words(
            rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3],
            rheights[0],
        )
        for i in range(MODEL_CURRICULUM_SIZE):
            env.mf.curriculum.data[i] = Scalar[DT](cw[i])
        env.mf.curriculum.upload(ctx)

        # ── the region frames, from the model's own sites ─────────────────
        env.reset_batch[N_ENVS](ctx, UInt64(1))
        env.d.qpos.download(ctx)
        env.d.site_xpos.download(ctx)
        ctx.synchronize()
        var frames = List[RegionFrame]()
        for i in range(len(f.regions)):
            var s = rsites[i]
            frames.append(RegionFrame(
                Float64(env.d.site_xpos.data[s * 3]),
                Float64(env.d.site_xpos.data[s * 3 + 1]),
                Float64(env.d.site_xpos.data[s * 3 + 2]),
            ))
        var radii = List[Float64]()
        for _ in range(len(f.slots)):
            radii.append(0.02)
        # ⚠ THE SCENE'S OWN REST POSE, read back from a real reset — not
        # zeros. `append_init_rows` overwrites only the free slots, so this is
        # the arm posture every frozen episode starts from.
        var base = List[Float64]()
        for i in range(NQ):
            base.append(Float64(env.d.qpos.data[i]))

        # ── freeze the table ──────────────────────────────────────────────
        var st = List[Float64]()
        var tix = List[Int32]()
        var mk = List[Float64]()
        var rep0 = SampleReport()
        append_init_rows(
            tg, f, 0, frames, radii, addrs, base, NQ, NV,
            N_PER_TASK, UInt64(7), 0, st, tix, mk, rep0,
        )
        append_init_rows(
            tr, f, 1, frames, radii, addrs, base, NQ, NV,
            N_PER_TASK, UInt64(7), N_PER_TASK, st, tix, mk, rep0,
        )
        var names = List[String]()
        names.append(String(tg.language))
        names.append(String(tr.language))
        write_init_table(
            String(TABLE_A), f.name, NQ, NV, st, tix, mk, names,
            seed=7, source_commit=String("p4-gate"),
        )
        var tbl = load_init_table(String(TABLE_A), f.name, NQ, NV)
        print("  froze", tbl.n_rows(), "episodes to", TABLE_A,
              "(", rep0.accepted, "placements in", rep0.attempts, "draws )")
        if tbl.n_rows() != N_ENVS:
            raise Error(
                "P4: the table has " + String(tbl.n_rows()) + " rows but the"
                " batch is " + String(N_ENVS) + ". Lane i must BE init row i"
                " or the lane-wise comparison compares different episodes."
            )

        var sa = List[Bool]()
        var sb = List[Bool]()
        var sc = List[Bool]()
        var fa = List[Float64]()
        var fb = List[Float64]()
        var fc = List[Float64]()

        print()
        print("--- A: frozen table, reset seed 1 ---")
        run_eval(env, ctx, tbl, tape_g, tape_r, UInt64(1), True, sa, fa)
        var ra = SuccessReport(tbl)
        for e in range(N_ENVS):
            ra.record(e, sa[e])
        ra.show(String("A"))

        print()
        print("--- B: the SAME table, reset seed 2 ---")
        run_eval(env, ctx, tbl, tape_g, tape_r, UInt64(2), True, sb, fb)
        var rb = SuccessReport(tbl)
        for e in range(N_ENVS):
            rb.record(e, sb[e])
        rb.show(String("B"))

        print()
        print("--- C: NO table, reset seed 2 — the control ---")
        run_eval(env, ctx, tbl, tape_g, tape_r, UInt64(2), False, sc, fc)
        var rc = SuccessReport(tbl)
        for e in range(N_ENVS):
            rc.record(e, sc[e])
        rc.show(String("C"))

        # ── the claim ─────────────────────────────────────────────────────
        print()
        var a_vs_b = 0
        var a_vs_c = 0
        for i in range(N_ENVS * NQ):
            if fa[i] != fb[i]:
                a_vs_b += 1
            if fa[i] != fc[i]:
                a_vs_c += 1
        print("  final qpos words differing:  A vs B", a_vs_b,
              " |  A vs C", a_vs_c, " of", N_ENVS * NQ)

        if a_vs_b != 0:
            raise Error(
                "P4: " + String(a_vs_b) + " final qpos words differ between two"
                " runs on the SAME frozen table. The runs differ only in the"
                " reset seed, which the table is supposed to overwrite — so"
                " either the table is not being applied to every lane, or"
                " something in the step path is reading the reset's RNG."
            )
        print("  ok: the same table gives BIT-IDENTICAL final state,"
              " under two different reset seeds")

        # ⚠⚠ THE CONTROL. Without it, everything above passes on a harness
        # that ignores the table AND ignores its seed.
        if a_vs_c == 0:
            raise Error(
                "P4: run C — which never applied the table — reached the SAME"
                " final state as A. Then the table is not what determined the"
                " episode and the agreement above means nothing. Check that"
                " `reset_batch` actually varies with its seed."
            )
        print("  ok: WITHOUT the table the same seed reaches a DIFFERENT state"
              " — so it is the table that fixes the episode")

        # ⚠⚠ THE CHECK THAT WOULD HAVE CAUGHT THE `_done` BUG, AND DID NOT
        # EXIST WHEN IT BIT. An outcome vector that is entirely False compares
        # EQUAL to another entirely False one, so every agreement check below
        # passed while the success metric was structurally a constant zero.
        # "Two runs agree" is worth nothing until something has varied.
        var any_solved = 0
        for e in range(N_ENVS):
            if sa[e]:
                any_solved += 1
        if any_solved == 0:
            raise Error(
                "P4: NO lane met its goal in run A, so every agreement below"
                " is an agreement between two constant-False vectors."
                " `settle` holds AT RESET (tests/tasks/test_task_reset_steps"
                " asserts it), so zero successes means the success SIGNAL is"
                " not being read — check that the loop reads `_reward` and"
                " not `_done`: `TERMINATE_ON_UNHEALTHY` defaults to False and"
                " discards the config's `done`."
            )
        print("  ok:", any_solved, "of", N_ENVS, "lanes met their goal —"
              " the outcome vector is not a constant")

        if not ra.same_as(rb):
            raise Error(
                "P4: the two runs on one table disagree on "
                + String(ra.n_differing_lanes(rb)) + " lanes, though"
                " their final states are identical. That is a bug in the"
                " SUCCESS bookkeeping, not in the physics."
            )
        print("  ok: and their per-lane outcomes agree, lane for lane")

        # ⚠ ASSERTED SINCE THE 2026-09-04 RUN. This was a printed note while I
        # expected it might legitimately be zero — under zero actions I
        # reasoned the goals would not depend on placement. MEASURED: 128 of
        # 256, i.e. every `settle` lane. Without the table the props sit at
        # the composed scene's XML defaults rather than on the table, so
        # `On(brick, table_top)` is False; with it, True. So the table changes
        # the OUTCOME and not merely the state, and this is a second control
        # independent of the qpos comparison.
        var lane_diff = ra.n_differing_lanes(rc)
        print("  A vs C differ on", lane_diff, "of", N_ENVS, "lanes' OUTCOMES")
        if lane_diff == 0:
            raise Error(
                "P4: A and C scored identically on every lane, though they"
                " start from different states. Either the goals of this"
                " family do not depend on placement at all — legitimate, and"
                " then this check should be relaxed FOR THAT FAMILY with the"
                " reason written down — or the table is not reaching the"
                " lanes it is supposed to."
            )

        # ── what the monitor gets ─────────────────────────────────────────
        var logger = CsvLogger(String(CSV))
        ra.log_to(logger, 0)
        logger.close()
        print("  metrics written to", CSV)

        print()
        print("=== PASS ===")
