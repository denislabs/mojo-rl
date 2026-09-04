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

## AND GAP D — A PARKED SLOT THAT STAYS PARKED

`So101TabletopConfig.pre_step_full_gpu` pins every inactive free slot at its
park pose, pose and velocity, before every step. `tests/tasks/
test_active_mask.mojo` gates the arithmetic of one call; what only a stepped
batch can say is whether it holds after real physics, and that is the check at
the end of this file. Gravity is a `Model` field shared by the batch, so
before Gap D a parked body fell 7.06 m over a full horizon.

## AND THE ACTIVE MASK, IN THE OBSERVATION

`meta[env, META_IDX_TASK_ACTIVE]` carries which slots this lane is running
(§3.4), and `So101TabletopConfig.custom_extract_obs_gpu` turns it into the
observation's last `N_FREE_SLOTS` words. The two tasks here have DIFFERENT
active sets — `gather` runs `cube_a`, `lift` does too, so a third task would
be needed to separate them on that axis; what separates them here is the mask
WORD, which is read per lane from `meta`.

⚠ `tests/tasks/test_active_mask.mojo` gates the hook's arithmetic on the CPU
at float64. What only THIS can say is that the word survives the round trip
through a real device buffer, a real reset and eight real steps — the mask is
written once, before the loop, and `_reset_env_lane` is what has to preserve
it.
"""

from std.random import seed as seed_rng
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_TASK_PARAM_0, META_IDX_TASK_ACTIVE,
    MODEL_CURRICULUM_SIZE,
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
from mojo_rl.tasks.active import active_mask


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
    var mka = active_mask(ta, f)
    var mkb = active_mask(tb, f)
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
        # ⚠ SYNC BEFORE READING `.data`. `download` ENQUEUES a copy; the host
        # array is not valid until the stream drains. Reading it straight away
        # is a stale read, not a crash — the region frames would come from
        # whatever the buffer held before the reset, and every lane would be
        # sampled around the wrong site.
        ctx.synchronize()

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
            # ⚠ THE WHOLE WORD, EVERY EPISODE. `meta` is not zeroed between
            # episodes — that is what lets this be written once and read every
            # step — so a lane keeps the PREVIOUS episode's mask unless it is
            # rewritten. Same trap `OP_NONE` exists for in the tape above.
            env.d.meta.data[e * METADATA_SIZE + META_IDX_TASK_ACTIVE] = (
                Scalar[DT](mka) if is_a else Scalar[DT](mkb)
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
        ctx.synchronize()

        # ⚠⚠ `reward_ptr()` IS A **DEVICE** POINTER ON THIS ENV.
        # `Phyics3dBatchedEnv._reward` is a `DeviceBuffer[DT]`
        # (`phyics3d_batched_env.mojo:266`) and `reward_ptr` hands back
        # `mptr(self._reward.unsafe_ptr())`. Dereferencing it on the host is a
        # segfault, not a wrong number — which is exactly what the first 5090
        # run did, crashing after the sampler with a bare libc backtrace.
        #
        # ⚠ THE ABI IS THE TRAP. `BatchedEnv.reward_ptr` is documented as
        # something a driver reads "in place", and the CPU-backed
        # implementations in `training/batched_env.mojo` return a pointer into
        # a host `List`. Same signature, two residencies; only this one needs
        # a copy. A host caller must never assume which it has.
        var rew_h = ctx.enqueue_create_host_buffer[DT](N_ENVS)
        ctx.enqueue_copy(rew_h, env._reward)
        ctx.synchronize()
        var rew = rew_h.unsafe_ptr()

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

        # ── the active mask, through the real device path ─────────────────
        #
        # ⚠ WHAT THIS ADDS OVER `tests/tasks/test_active_mask.mojo`, WHICH
        # ALREADY GATES THE HOOK'S ARITHMETIC ON THE CPU: the word was written
        # into a DEVICE buffer, survived `reset_batch`'s `_reset_env_lane` and
        # eight steps, and came back out of the observation kernel. That chain
        # is what the CPU gate cannot exercise, and it is the half that would
        # break if the reset ever started zeroing `meta`.
        #
        # ⚠⚠ THESE TWO TASKS SHARE AN ACTIVE SET — `gather` and `lift` both
        # run table/brick/cube_a — so the mask does NOT vary by lane here and
        # this cannot catch a hook that reads lane 0's word for everyone.
        # `test_active_mask` runs `reach` beside `gather` for exactly that, at
        # BATCH=2. Stated rather than left for a reader to discover, because a
        # per-lane check that cannot vary is the shape of a vacuous gate.
        var obs_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * EnvT.OBS_DIM)
        ctx.enqueue_copy(obs_h, env._obs)
        ctx.synchronize()
        var ob = obs_h.unsafe_ptr()

        comptime MB = So101TabletopConfig.OBS_MASK_BASE
        comptime NF = So101TabletopConfig.N_FREE_SLOTS
        var mask_bad = 0
        var lit = 0
        var off = 0
        for e in range(N_ENVS):
            var want = mka if (e % 2) == 0 else mkb
            for j in range(NF):
                var si = (
                    So101TabletopConfig.FREE_SLOT_IDX_0 if j == 0
                    else (So101TabletopConfig.FREE_SLOT_IDX_1 if j == 1
                          else So101TabletopConfig.FREE_SLOT_IDX_2)
                )
                var on = ((Int(want) >> si) & 1) == 1
                var got = Float64(ob[e * EnvT.OBS_DIM + MB + j])
                if got != (1.0 if on else 0.0):
                    mask_bad += 1
                if on:
                    lit += 1
                else:
                    off += 1

        print()
        print("  active words:", lit, "on,", off, "off, across",
              N_ENVS * NF, "lane-slots")
        if mask_bad != 0:
            raise Error(
                "P3: " + String(mask_bad) + " active words in the observation"
                " disagree with the mask the host wrote. The word is at"
                " `meta[env, META_IDX_TASK_ACTIVE]`; if it reads 0 everywhere,"
                " something in the reset path now ZEROES `meta` and the tape"
                " beside it is next."
            )
        # ⚠ ANTI-VACUITY. "Every active word matches" is also true when every
        # slot is active and every word is 1 — and `cube_b` is parked in all
        # three shipped tasks, so a 0 must appear or the comparison above was
        # against a constant.
        if lit == 0 or off == 0:
            raise Error(
                "P3: the observation's active words are all "
                + ("0" if lit == 0 else "1") + ". A mask that never varies"
                " cannot be told from a hook that writes a constant."
            )
        print("  ok: every lane's active words are the mask the host wrote,"
              " and BOTH values occur")

        # ── Gap D: a parked slot did not move ─────────────────────────────
        #
        # ⚠ THE ONE THING ONLY A STEPPED BATCH CAN SAY. `test_active_mask`
        # gates the repark's arithmetic on one call; this is whether it holds
        # after real physics — gravity is a `Model` field shared by the batch,
        # so before Gap D a parked body FELL, 7.06 m over a full horizon.
        env.d.qpos.download(ctx)
        env.d.qvel.download(ctx)
        ctx.synchronize()
        comptime CB_SLOT = So101TabletopConfig.FREE_SLOT_IDX_2
        comptime CB_QADR = So101TabletopConfig.FREE_QADR_2
        comptime CB_DADR = So101TabletopConfig.FREE_DADR_2
        var pk_z = f.park_z
        var moved = 0
        var spun = 0
        var max_dz = 0.0
        for e in range(N_ENVS):
            # ⚠ `cube_b` IS PARKED IN ALL THREE SHIPPED TASKS, which is why it
            # is the one checked. `brick` and `cube_a` are active here and
            # MUST have moved — that is the anti-vacuity leg below.
            var z = Float64(env.d.qpos.data[e * NQ + CB_QADR + 2])
            var dz = z - pk_z
            if dz < 0.0:
                dz = -dz
            if dz > max_dz:
                max_dz = dz
            if dz != 0.0:
                moved += 1
            for k in range(6):
                if Float64(env.d.qvel.data[e * NV + CB_DADR + k]) != 0.0:
                    spun += 1
                    break
        print()
        print("  parked cube_b after", STEPS, "steps: max |dz| =", max_dz,
              "m,", moved, "lanes moved,", spun, "lanes carry velocity")
        if moved != 0 or spun != 0:
            raise Error(
                "P3/Gap D: " + String(moved) + " lanes' parked cube_b left"
                " z=" + String(pk_z) + " and " + String(spun) + " carry"
                " velocity. `pre_step_full_gpu` pins it EVERY step; if this"
                " fires, either the env stopped calling the wide pre-step"
                " hook or the active mask says cube_b is active."
            )
        # ⚠⚠ ANTI-VACUITY, AND IT IS THE WHOLE CHECK. "Nothing moved" is also
        # true of a batch that never stepped, of a scene with no gravity, and
        # of a repark that pinned EVERY slot. An ACTIVE prop must have moved.
        comptime BR_QADR = So101TabletopConfig.FREE_QADR_0
        var active_moved = 0
        for e in range(N_ENVS):
            if Float64(env.d.qpos.data[e * NQ + BR_QADR + 2]) != Float64(
                env.d.qpos.data[e * NQ + CB_QADR + 2]
            ):
                active_moved += 1
        if active_moved == 0:
            raise Error(
                "P3/Gap D: the ACTIVE brick sits at the parked slot's height"
                " in every lane, so 'the parked one did not move' says"
                " nothing — either the repark pinned everything, or the batch"
                " never stepped."
            )
        print("  ok: the parked slot is EXACTLY at its park pose with zero"
              " velocity, while", active_moved, "lanes' active brick is not")
        print()
        print("=== PASS ===")
