"""SAC ON THE TASK FAMILY — the first policy this layer has ever carried.

    pixi run -e nvidia mojo run -I . examples/tasks/sac_task_reach_gpu.mojo
    ... --steps 200000 --envs 64

⚠⚠ NVIDIA ONLY. The family is `nv = 24`, where the P0 park probe died on
Metal ("Compute function exceeds available stack space" — the physics kernels
stack-allocate per-thread arrays sized by `nv`). It COMPILES on Apple and
cannot launch, exactly like `task_batched_gpu.mojo` beside it.

## ⚠⚠ EVERY NUMBER THIS LAYER HAS REPORTED SO FAR IS A HARNESS NUMBER

P1-P5 gated the spec, the composer, the goal language, the sampler, the tape,
the active mask, the init table and the LIBERO importer. Not one of them put a
POLICY on the family, and the paths a training run exercises are not the paths
a gate does: the observation's scale, a sparse reward's cadence, the reset rate
under success-termination, and whether `TERMINATE_ON_UNHEALTHY` was spelled.
This file is that run.

## ⚠⚠ THE TASK IS `so101_reach_clear`, AND THE REASON IS A REAL GAP

`so101_reach_brick` cannot train today. An ACTIVE free slot must be PLACED at
every episode reset, and on the GPU path nothing places it: `_reset_env_lane`
restores the composed scene's `qpos0` — the PARK pose, 50 m up — and only
INACTIVE slots are pinned there afterwards by `pre_step_full_gpu`. So an
active prop starts every episode in the sky, falls for the whole horizon, and
its qpos and qvel go into the observation. The HOST places props
(`sampler.sample_placements` + `reset.reset_slots`), which is why the eval and
viewer paths are fine and why nothing saw this until something trained.

`so101_reach_clear` has the same goal and NO active free slot, so it needs no
placement. See its header for what that costs.

## ⚠⚠ THREE WIRING FACTS THAT ARE EACH A SILENT FAILURE IF MISSED

1. **`TERMINATE_ON_UNHEALTHY=True`.** The config's reward hook returns
   `(reward, holds)` and asks to terminate on success — and the ask is
   DISCARDED unless the env is instantiated with this flag
   (`phyics3d_batched_env.mojo:1161`). Without it a solved lane keeps running
   and banks +1 per step to the horizon, so a return stops being a success
   indicator and becomes "how early did it succeed". `task_eval_frozen.mojo`
   already reported 0/128 on a task that holds at reset for the same reason,
   from the other side.

2. **`ACTION_SCALE = 1.0`.** `So101TabletopConfig.NORMALIZED_ACTIONS` is True,
   so the action IS [-1, 1] per joint and the env maps it affinely onto each
   actuator's own ctrlrange. A scale of 2.0 maps [-2, 2] onto the range and
   puts the useful band back inside the tanh rails — undoing the fix while
   still looking configured. The measurement behind that flag is in
   `Phyics3dEnvConfig.NORMALIZED_ACTIONS`, on this same robot.

3. **`initial_episode_fill = 0.0`.** SAC seeds its return window with
   `-1250.0`, a HalfCheetah-flavoured value. Here a return is in {0, 1}, so
   the default makes every reading before the window fills a blend of real
   returns and sentinels and reads as a broken reward.

## ⚠⚠ THE MEASURED BASELINES — READ THESE BEFORE ANY CURVE

Per task, 20k env-steps at N_ENVS=64 with `--warmup >= --steps` (uniform
random) and the driver's greedy eval on an UNTRAINED actor:

    task                  random   untrained greedy   constant action
    so101_lift_brick        0.00        0.00           never met
    so101_gather_bricks     0.00        0.00           never met
    so101_reach_clear       0.25        1.00           SWEPT THROUGH
    so101_settle_brick      1.00        1.00           met at every step

⚠ `reach` IS NOT A REACHING TASK AND THAT IS WHY THE DEFAULT IS `lift`.
`examples/tasks/task_null_action.mojo` measures it: a CONSTANT action of +0.3
meets `AtRegion(robot_gripperframe, table_top)` on 77 consecutive steps, and
the run ENDS at step 97 of 300 — the arm sweeps the gripper across the region
on its way somewhere else. An instantaneous predicate over a CONTROLLED end
effector, with a per-step reward and first-hit termination, asks "did the
gripper ever pass through here". A predicate over an OBJECT's pose does not
have that failure: a sweep does not lift a brick.

⚠ `settle` IS THE PROBE and scores 1.00 by construction — its goal holds at
reset. Two GPU gates need a true lane; see its own header. Training it is
meaningless and its 20032 episodes in 20k steps (one per step, because
success terminates) is what a correctly wired success-termination looks like.

⚠⚠ SO A FLOOR OF 0.00 IS WHAT `lift` AND `gather` ACTUALLY HAVE, and any
sustained rate above it is learning. They are also SPARSE and HARD: `lift`
needs a grasp before it pays anything. A flat curve on them is an honest RL
result, not a broken harness — which is a claim this file can now make,
because the harness is measured.

## ⚠ THE RETURN *IS* THE SUCCESS RATE, WHICH IS WHY THIS IS READABLE AT ALL

The reward is sparse — +1 on the step the goal holds — and the episode
TERMINATES on that step. So an episode return is exactly 1 if solved and 0 if
not, and `agent.mean_return()` over the last 100 episodes is the success rate
directly. No band table, no shaped-reward calibration: the criterion is
"does it move off zero".

⚠ AND THE FLOOR IS ZERO HERE, unlike the shaped SO-ARM101 reach whose
untrained baseline is 46 of 500. `examples/tasks/task_reachability.mojo`
measures 3.2% of uniform arm poses inside the goal region, so a random policy
scores somewhere near but not at zero; anything sustained above ~0.1 is
learning.
"""

from std.random import seed as seed_rng
from std.sys import argv
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.linear_relu import LinearReLU
from mojo_rl.core.logger import CsvLogger
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.sac import SACAgent
from mojo_rl.deep_agents.training.blocks import UniformSampleGpuStep
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_TASK_PARAM_0, META_IDX_TASK_ACTIVE,
    META_IDX_INIT_REGION_0, MODEL_CURRICULUM_SIZE,
)
from mojo_rl.physics3d.parser.runtime_load import parse_model_runtime

from mojo_rl.tasks.spec import (
    load_family, load_task, validate_task_against_family, SLOT_FREE,
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
from mojo_rl.tasks.active import active_mask, init_region_words


comptime N_ENVS = 64
# ⚠⚠ `lift`, NOT `reach`. `reach` and `reach_clear` are both
# `AtRegion(robot_gripperframe, table_top)` and
# `examples/tasks/task_null_action.mojo` measured what that is worth: a
# CONSTANT action of +0.3 meets it on 76 consecutive steps, and the longest
# run ends at step 96 of 300 — the arm SWEEPS the gripper across the region on
# its way somewhere else. An instantaneous predicate over a controlled end
# effector plus first-hit termination asks "did the gripper ever pass through
# here", which most large joint motions satisfy without aiming.
#
# A predicate over an OBJECT's pose is not that shape: a sweep does not lift a
# brick. `--task` takes any of them.
comptime DEFAULT_TASK = "so101_lift_brick"
comptime FAMILY = "mojo_rl/tasks/families/so101_tabletop.family"

# ⚠⚠ `TERMINATE_ON_UNHEALTHY=True` — see the header. This is the flag, and it
# is the only place in this file where success termination is expressible.
comptime EnvT = Phyics3dBatchedEnv[
    So101TabletopModel, So101TabletopConfig, N_ENVS,
    TERMINATE_ON_UNHEALTHY=True,
]

comptime OBS_DIM = EnvT.OBS_DIM      # 54 = NQ(27) + NV(24) + N_FREE(3)
comptime ACT_DIM = 6
comptime HIDDEN = 256
comptime BATCH = 256
comptime REPLAY_CAPACITY = 1_000_000

comptime NUM_STEPS = 300_000
comptime WARMUP_STEPS = 10_000
comptime PRINT_EVERY = 10_000
comptime DIAG_EVERY = 2_000
comptime CHECKPOINT_EVERY = 50_000
comptime EVAL_EVERY = 25_000

# ⚠ MEASURED, NOT ASSUMED — the header carries the whole table. These two are
# the DEFAULT task's, because a verdict that has to be looked up to be read is
# a verdict nobody reads.
comptime RANDOM_BASELINE = 0.00
comptime UNTRAINED_GREEDY = 0.00
comptime CHECKPOINT_PATH = "sac_task_reach.ckpt"
comptime CSV = "/tmp/mojo_rl_sac_task_reach.csv"

# See wiring fact 2 in the header. NORMALIZED_ACTIONS is True on this config.
comptime ACTION_SCALE = Scalar[DT](1.0)

comptime ActorNet = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    LinearReLU[OBS_DIM, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
]
comptime CriticNet = Sequential[
    LinearReLU[OBS_DIM + ACT_DIM, HIDDEN],
    LinearReLU[HIDDEN, HIDDEN],
    Linear[HIDDEN, 1],
]


def main() raises:
    seed_rng(42)

    # ⚠ `--warmup` EXISTS FOR THE BASELINE, not for tuning. Setting it at or
    # above `--steps` runs the whole loop on UNIFORM RANDOM actions with no
    # gradient step, which is the only way to measure what this task pays a
    # policy that has learned nothing — and the number every later rate has to
    # beat. `task_reachability.mojo` predicts it from geometry (3.2% of
    # uniform arm POSES meet the goal); this measures it through the dynamics.
    var num_steps = NUM_STEPS
    var warmup = WARMUP_STEPS
    var eval_every = EVAL_EVERY
    var task_name = String(DEFAULT_TASK)
    var args = argv()
    for i in range(1, len(args)):
        var a = String(args[i])
        if a == "--steps" and i + 1 < len(args):
            num_steps = Int(String(args[i + 1]))
        elif a == "--warmup" and i + 1 < len(args):
            warmup = Int(String(args[i + 1]))
        elif a == "--eval-every" and i + 1 < len(args):
            eval_every = Int(String(args[i + 1]))
        elif a == "--task" and i + 1 < len(args):
            task_name = String(args[i + 1])

    print("=" * 72)
    print("SAC on the task family —", task_name, "(GPU)")
    print("=" * 72)

    # ── the task, on the host ─────────────────────────────────────────────
    var f = load_family(FAMILY)
    var t = load_task("mojo_rl/tasks/tasks/" + task_name + ".task")
    validate_task_against_family(t, f)
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)
    var rheights = region_half_heights(f)

    var g = bind_goal(parse_goal(t.goal), f, fmd.body_names, fmd.site_names)
    require_tier_a(g, t.name)
    # ⚠ ONE region table on device and a term's region index is ignored there.
    require_gpu_regions(g, t.name)
    var tape = encode_goal(g)
    var mask = active_mask(t, f)

    print("  task     :", t.name)
    print("  language :", t.language)
    print("  goal     :", t.goal)
    print("  OBS_DIM  :", OBS_DIM, " ACT_DIM:", ACT_DIM, " N_ENVS:", N_ENVS)
    print("  steps    :", num_steps, " warmup:", warmup,
          "(baseline run)" if warmup >= num_steps else "")
    print("  action_scale:", ACTION_SCALE, "(NORMALIZED_ACTIONS is True)")

    # ⚠⚠ AN ACTIVE FREE SLOT WOULD FALL FOR THE WHOLE EPISODE. Refused here
    # rather than trained around — see the header. The failure is not a crash:
    # it is a policy learning from an observation with a prop falling through
    # it, and the curve looks like a hard task.
    # ⚠ THE REFUSAL THAT USED TO BE HERE IS GONE, AND THE INIT WORDS ARE WHY.
    # This file refused any task activating a FREE slot, because nothing
    # placed one at a GPU reset and a prop would start every episode 50 m up
    # and fall. `So101TabletopConfig.init_qpos_gpu` now samples them per lane
    # from `META_IDX_INIT_REGION_*`, gated against the host sampler coordinate
    # for coordinate by `tests/tasks/test_device_placement.mojo`.
    var iw = init_region_words(t, f)
    var n_active_free = 0
    for j in range(len(iw)):
        if iw[j] >= 0.0:
            n_active_free += 1
    print("  free slots placed at reset:", n_active_free, "of", len(iw))

    with DeviceContext() as ctx:
        var logger = CsvLogger(CSV)
        var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

        var agent = SACAgent[
            "gpu",
            UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
            ActorNet,
            CriticNet,
        ](
            ctx=ctx,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            gamma=0.99,
            tau=0.005,
            action_scale=ACTION_SCALE,
            init_alpha=Scalar[DT](0.2),
            target_entropy=-Scalar[DT](ACT_DIM),
            learning_starts=warmup,
            window_size=100,
            # See wiring fact 3 in the header: a return here is 0 or 1.
            initial_episode_fill=0.0,
        )
        var env = EnvT(ctx)

        # ── the region table, once; the tape and mask, once per lane ──────
        #
        # ⚠ WRITTEN BEFORE THE LOOP AND NEVER AGAIN, which is only safe
        # because `_reset_env_lane` writes just `META_IDX_STEP_COUNT` and
        # leaves the rest of `meta` alone (`gpu/constants.mojo`). Every lane
        # runs the SAME task here, so there is no per-lane variation to
        # maintain — a multi-task run would still write these once, with
        # different words per lane.
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

        for e in range(N_ENVS):
            for w in range(TAPE_WORDS):
                env.d.meta.data[e * METADATA_SIZE + META_IDX_TASK_PARAM_0 + w] \
                    = Scalar[DT](tape[w])
            env.d.meta.data[e * METADATA_SIZE + META_IDX_TASK_ACTIVE] = \
                Scalar[DT](mask)
            # ⚠⚠ THE INIT WORDS SURVIVE EVERY RESET, which is the whole point:
            # `_reset_env_lane` writes only META_IDX_STEP_COUNT, so writing
            # these once before the loop makes EVERY later reset place the
            # props. A driver that wrote them per episode would be doing the
            # host round-trip this exists to remove.
            for j in range(len(iw)):
                env.d.meta.data[
                    e * METADATA_SIZE + META_IDX_INIT_REGION_0 + j
                ] = Scalar[DT](iw[j])
        env.d.meta.upload(ctx)

        # ── a SECOND env, for greedy eval ─────────────────────────────────
        #
        # ⚠⚠ IT NEEDS THE SAME THREE UPLOADS, AND A MISSING ONE IS SILENT.
        # `curriculum` and `meta` are per-ENV-INSTANCE device buffers, so an
        # eval env constructed without them evaluates a tape of zeros — op 0
        # is `OP_IN`, with body 0 against region 0 — which is a real, wrong
        # predicate that returns a plausible rate rather than an error. The
        # driver's `eval_env` is an ISOLATED BatchedEnv by design (it must
        # never share the training env's state), and isolation is exactly what
        # makes this easy to forget.
        var eval_env = EnvT(ctx)
        for i in range(MODEL_CURRICULUM_SIZE):
            eval_env.mf.curriculum.data[i] = Scalar[DT](cw[i])
        eval_env.mf.curriculum.upload(ctx)
        for e in range(N_ENVS):
            for w in range(TAPE_WORDS):
                eval_env.d.meta.data[
                    e * METADATA_SIZE + META_IDX_TASK_PARAM_0 + w
                ] = Scalar[DT](tape[w])
            eval_env.d.meta.data[e * METADATA_SIZE + META_IDX_TASK_ACTIVE] = \
                Scalar[DT](mask)
            for j in range(len(iw)):
                eval_env.d.meta.data[
                    e * METADATA_SIZE + META_IDX_INIT_REGION_0 + j
                ] = Scalar[DT](iw[j])
        eval_env.d.meta.upload(ctx)
        print("  ok: region table, tape and active mask uploaded to BOTH envs")

        print("-" * 72)
        var t0 = perf_counter_ns()
        _ = agent.train[
            EnvT,
            N_ENVS=N_ENVS,
            USE_TRAIN_CUDA_GRAPH=True,
            USE_ENV_CUDA_GRAPH=False,
            L=CsvLogger,
        ](
            env,
            num_steps,
            rng_seed=UInt64(42),
            updates_per_step=N_ENVS,
            print_every=PRINT_EVERY,
            verbose=True,
            logger=logger_ptr,
            diag_every=DIAG_EVERY,
            episode_sync_every=32,
            checkpoint_every=CHECKPOINT_EVERY,
            checkpoint_path=String(CHECKPOINT_PATH),
            # ⚠ GREEDY, on a SEPARATE env, at a FIXED eval seed — the
            # criterion number. `mean_return()` below is measured under SAC's
            # stochastic policy and understates what the actor has learned;
            # this is the actor mean, with no sampling. Both are printed
            # because a large gap between them is itself a reading (an actor
            # that is good and an alpha that is still high).
            eval_env=Pointer(to=eval_env).as_unsafe_any_origin(),
            eval_every=eval_every,
            eval_episodes=N_ENVS,
            eval_max_steps=So101TabletopConfig.MAX_STEPS + 1,
        )
        var secs = Float64(perf_counter_ns() - t0) / 1e9
        logger.close()

        var rate = Float64(agent.mean_return())
        print("-" * 72)
        print("  env steps          :", num_steps)
        print("  elapsed            :", secs, "s")
        print("  episodes           :", agent.ep_count())
        print("  SUCCESS RATE       :", rate, "(last 100 episodes)")
        print("  csv                :", CSV)
        print("  checkpoint         :", CHECKPOINT_PATH)

        # ⚠⚠ THE ANTI-VACUITY CHECK, AND IT IS NOT THE SUCCESS CRITERION.
        # Zero completed episodes reports `mean_return` as the fill value and
        # prints a plausible rate — which is what a run whose episodes never
        # terminate looks like, i.e. exactly the `TERMINATE_ON_UNHEALTHY`
        # failure this file's header is about. The horizon is 300 steps and
        # truncation ends every episode regardless, so a run of this length
        # with no episodes is a broken loop, not a hard task.
        if agent.ep_count() == 0:
            raise Error(
                "sac task reach: NOT ONE episode completed in "
                + String(num_steps) + " env-steps against a 300-step horizon."
                " `mean_return` is then the fill value and the rate above is"
                " meaningless. Check that the driver is stepping and that"
                " truncation is reaching the tracker."
            )
        print()
        # ⚠⚠ THE VERDICT IS AGAINST A BASELINE, NOT AGAINST ZERO. "The rate
        # moved off zero" was the criterion this run was built to answer and
        # it is the WRONG one for this task: uniform random already scores
        # 0.27 and the UNTRAINED greedy actor scores 1.00 (see the header).
        # Printing "moved off zero" here would have reported a trivial task as
        # a trained one.
        print("  baselines for", DEFAULT_TASK, "— random", RANDOM_BASELINE,
              " untrained greedy", UNTRAINED_GREEDY)
        if task_name != DEFAULT_TASK:
            print("  ⚠ RUNNING", task_name, "— the baselines above are the")
            print("  DEFAULT task's. See this file's header for the table.")
        if rate <= RANDOM_BASELINE:
            print("  FLAT — the rate did not beat the random baseline.")
        else:
            print("  the rate BEAT the random baseline:", rate)
        print("=" * 72)
