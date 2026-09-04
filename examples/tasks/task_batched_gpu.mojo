"""1024 lanes, TWO DIFFERENT TASKS, one binary — P3's gate.

    pixi run -e nvidia mojo run -I . examples/tasks/task_batched_gpu.mojo

⚠⚠ RUN THIS ON NVIDIA. The family is `nv = 24`, and that is exactly where the
P0 park probe died on Metal — `k=3` is nv 24 and fails at pipeline creation
with "Compute function exceeds available stack space", because the physics
kernels stack-allocate per-thread arrays sized by `nv`. It COMPILES on Apple
and cannot launch, so an Apple run proves only that it builds.

## WHAT IT ASSERTS, AND THE SECOND ONE IS THE POINT

1. **PARITY** — every lane's GPU reward equals the CPU evaluation of that
   lane's downloaded state, through `tasks/eval.eval_goal`. Two evaluators,
   two containers, one set of predicates.
2. ⚠⚠ **THE NEGATIVE LEG** — lanes running task A and lanes running task B do
   NOT all agree. A tape misindexed by lane hands every lane the SAME goal,
   and check 1 still passes: the CPU leg reads the same misindexed tape and
   agrees with itself. Only a difference BETWEEN the task groups catches it.
   `tests/tasks/test_tape_gpu_parity.mojo` caught exactly that at BATCH=2 by
   ablation (50 agreements -> 35); this is the same shape at scale.

## HOW A LANE GETS ITS TASK

Even lanes run `gather`, odd lanes run `lift`. The tape is written into
`meta[env, META_IDX_TASK_PARAM_0 .. _11]` once, before the loop — reset
preserves it (`gpu/constants.mojo:164`) — and the region table into
`curriculum[0, 0..4]`, which is shared because a region belongs to the FAMILY.

⚠ NEITHER NEEDED A NEW KERNEL OPERAND. Gap E scoped this as two; both channels
already existed and were unused.
"""

from std.random import seed as seed_rng
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_TASK_PARAM_0, MODEL_CURRICULUM_SIZE,
)
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.physics3d.parser.runtime_load import parse_model_runtime

from mojo_rl.tasks.spec import load_family, load_task, validate_task_against_family
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.family_config import So101TabletopConfig
from mojo_rl.tasks.so101_tabletop_xml import So101TabletopModel
from mojo_rl.tasks.predicates import parse_goal, bind_goal, require_tier_a
from mojo_rl.tasks.eval import eval_goal, region_sites, region_rects
from mojo_rl.tasks.tape import encode_goal, TAPE_WORDS
from mojo_rl.tasks.gpu_eval import region_table_words
from mojo_rl.tasks.sampler import sample_placements, RegionFrame, SampleReport
from mojo_rl.tasks.reset import free_slot_addresses, reset_slots


comptime N_ENVS = 1024
comptime STEPS = 8
comptime SEED = UInt64(11)

comptime EnvT = Phyics3dBatchedEnv[
    So101TabletopModel, So101TabletopConfig, N_ENVS
]
comptime NB = So101TabletopModel.NBODY
comptime NS = So101TabletopModel.NSITE
comptime NQ = So101TabletopModel.NQ
comptime NV = So101TabletopModel.NV


def main() raises:
    seed_rng(0)
    print("=" * 68)
    print("P3 — 1024 lanes, two tasks, one monomorphisation")
    print("=" * 68)

    var f = load_family("mojo_rl/tasks/families/so101_tabletop.family")
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)

    var ta = load_task("mojo_rl/tasks/tasks/so101_gather_bricks.task")
    var tb = load_task("mojo_rl/tasks/tasks/so101_lift_brick.task")
    validate_task_against_family(ta, f)
    validate_task_against_family(tb, f)
    var ga = bind_goal(parse_goal(ta.goal), f, fmd.body_names, fmd.site_names)
    var gb = bind_goal(parse_goal(tb.goal), f, fmd.body_names, fmd.site_names)
    require_tier_a(ga, ta.name)
    require_tier_a(gb, tb.name)
    var tpa = encode_goal(ga)
    var tpb = encode_goal(gb)
    print("  even lanes:", ta.name, "|", ta.goal)
    print("  odd  lanes:", tb.name, "|", tb.goal)

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

        # ── the region table, once ────────────────────────────────────────
        var cw = region_table_words(
            rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3]
        )
        for i in range(MODEL_CURRICULUM_SIZE):
            env.mf.curriculum.data[i] = Scalar[DT](cw[i])
        env.mf.curriculum.upload(ctx)

        env.reset_batch[N_ENVS](ctx, SEED)
        env.d.qpos.download(ctx)
        env.d.site_xpos.download(ctx)

        # ── per lane: its task's tape, and its sampled placements ─────────
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

        var rep = SampleReport()
        for e in range(N_ENVS):
            var is_a = (e % 2) == 0
            for k in range(TAPE_WORDS):
                env.d.meta.data[e * METADATA_SIZE + META_IDX_TASK_PARAM_0 + k] = (
                    Scalar[DT](tpa[k]) if is_a else Scalar[DT](tpb[k])
                )
            var qpos = List[Float64]()
            for i in range(NQ):
                qpos.append(Float64(env.d.qpos.data[e * NQ + i]))
            var qvel = List[Float64]()
            for _ in range(NV):
                qvel.append(0.0)
            # ⚠ BRANCHED, NOT `ta if is_a else tb`. `TaskSpec` owns Lists and
            # is Movable but NOT ImplicitlyCopyable, so a ternary would try to
            # copy one per lane — 1024 copies of two Lists, and a compile
            # error rather than a slow loop, which is the better outcome.
            if is_a:
                var pl = sample_placements(ta, f, frames, radii, SEED, e, rep)
                reset_slots(ta, f, pl, addrs, qpos, qvel)
            else:
                var pl = sample_placements(tb, f, frames, radii, SEED, e, rep)
                reset_slots(tb, f, pl, addrs, qpos, qvel)
            for i in range(NQ):
                env.d.qpos.data[e * NQ + i] = Scalar[DT](qpos[i])
            for i in range(NV):
                env.d.qvel.data[e * NV + i] = Scalar[DT](qvel[i])
        env.d.meta.upload(ctx)
        env.d.qpos.upload(ctx)
        env.d.qvel.upload(ctx)
        print("  sampler:", rep.accepted, "placements in", rep.attempts,
              "draws across", N_ENVS, "lanes")

        # ── step, then compare ────────────────────────────────────────────
        for _ in range(STEPS):
            env.step_batch[N_ENVS](ctx, UInt64(0))
        ctx.synchronize()

        env.d.xpos.download(ctx)
        env.d.xquat.download(ctx)
        env.d.site_xpos.download(ctx)
        var rew = env.reward_ptr()

        var mismatch = 0
        var a_true = 0
        var b_true = 0
        var first_bad = -1
        for e in range(N_ENVS):
            # ⚠ THE CPU LEG READS THE STATE THE GPU PRODUCED, at the same
            # float32 values — widened, not recomputed. Recomputing FK on the
            # host would compare two physics runs, not two evaluators.
            var xb = List[Float64]()
            for i in range(NB * 3):
                xb.append(Float64(env.d.xpos.data[e * NB * 3 + i]))
            var xq = List[Float64]()
            for i in range(NB * 4):
                xq.append(Float64(env.d.xquat.data[e * NB * 4 + i]))
            var sp = List[Float64]()
            for i in range(NS * 3):
                sp.append(Float64(env.d.site_xpos.data[e * NS * 3 + i]))
            var is_a = (e % 2) == 0
            var host = eval_goal(ga if is_a else gb, f, xb, xq, sp, rsites)
            var gpu = Float64(rew[e]) > 0.5
            if host != gpu:
                mismatch += 1
                if first_bad < 0:
                    first_bad = e
            if is_a and gpu:
                a_true += 1
            if (not is_a) and gpu:
                b_true += 1

        print()
        print("  lanes on", ta.name, "meeting their goal:", a_true, "/", N_ENVS // 2)
        print("  lanes on", tb.name, "meeting their goal:", b_true, "/", N_ENVS // 2)
        print("  GPU-vs-CPU mismatches:", mismatch, "of", N_ENVS)

        if mismatch != 0:
            raise Error(
                "P3: " + String(mismatch) + " lanes disagree, first lane "
                + String(first_bad) + ". The device evaluator and the host one"
                + " read the same state and must agree."
            )
        print("  ok: every lane's GPU reward equals the CPU evaluation")

        # ⚠⚠ THE NEGATIVE LEG. Without it a tape misindexed by lane passes
        # everything above — the CPU leg reads the same misindexed tape and
        # agrees with itself. Only a DIFFERENCE between the task groups shows
        # that each lane ran ITS OWN goal.
        if a_true == b_true:
            raise Error(
                "P3: both task groups scored identically (" + String(a_true)
                + "). Either the two goals happen to coincide on this state —"
                + " change the seed — or EVERY LANE IS RUNNING THE SAME TAPE,"
                + " which the parity check above cannot see."
            )
        print("  ok: the two task groups scored DIFFERENTLY —",
              a_true, "vs", b_true, "— so each lane ran its own goal")
        print()
        print("=== PASS ===")
