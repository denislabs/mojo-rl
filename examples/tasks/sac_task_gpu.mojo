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
    so101_gather_bricks    <0.02        0.00           never met
    so101_reach_clear       0.25        1.00           SWEPT THROUGH
    so101_settle_brick      1.00        1.00           met at every step

⚠ `gather` IS NOT EXACTLY ZERO under random actions — two 20k warmup-only runs
gave 0.000 and 0.0156 (one lane of 64). Random does occasionally push the
blocks together, so 0.02 is the number a rate has to beat, and at a 100-
episode window its standard error is 0.014. The verdict prints that band
because the first long `gather` run oscillated 0.00 .. 0.05 for 125k steps and
every one of those values sits inside it.

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
sustained rate above it is learning.

## ⚠⚠ THE REWARD IS SHAPED NOW, AND THE RETURN IS NO LONGER THE SUCCESS RATE

The first 125k-step `gather` run was flat, and the reason was arithmetic:
at ~1.5% success that is about SIX rewarding transitions in 125,000 — 5e-05
of the replay buffer, which a batch of 256 contains 1.4% of the time. The
critic almost never saw a success. `So101TabletopConfig` now subtracts two
dense terms, `SHAPE_W_GOAL` on the goal's own distance (generic, from the
tape) and `SHAPE_W_REACH` on the gripper's distance to the body the goal
names — the second because goal distance alone has NO gradient until the arm
touches something, which on `gather` is the whole difficulty.

    tolerance reward, random actions, gather, 32 envs   +70.7
    success rate, greedy, untrained                       0.00

⚠⚠ THE REWARD IS POSITIVE AND BOUNDED NOW — two `tolerance` terms in [0, 1]
weighted 1.0 and 0.5, so an episode return lives in [0, 450] and random sits
at 70.7: 16% of the ceiling with the rest reachable. The old clipped linear
PENALTY put random at -31.5 against a ceiling of 0 — the same information with
no room above it and no saturation below.

⚠ EARLIER FIGURES ARE NOT COMPARABLE, and are recorded because the reasoning
around them is: -31.5 (32 envs) and -19.98 (64 envs) were the linear form at
weights 0.50/0.25, and -3.996 was 0.10/0.05. A return is comparable only
within one reward SHAPE, one weight pair and one lane count. All three now
travel with the run as `cfg/*` fields.

⚠ THE SUCCESS-RATE baselines are lane-count independent — a rate is per
episode either way — so only the RETURN figures moved.

⚠⚠ CRITIC HEALTH TRACKS WHICH TERM DOMINATES, not the magnitude and not the
shape. Five runs, all at 32 env / 32 upd / tau 0.0025 — the same 7.7%:

    run   shape      weights    goal    reach   reach/goal   critic
    7/8   linear     0.5/0.25   0.0575  0.0478        0.83   healthy
     10   linear     0.1/0.70   0.0115  0.1337       11.63   DIVERGED
     11   tolerance  1.0/0.50   0.0476  0.0929        1.95   DIVERGED
     12   tolerance  0.43/0.21  0.0205  0.0390        1.91   DIVERGED

The two columns are each term's contribution at the MEASURED random distances
(goal 0.115 m, reach 0.191 m — `task_shaping_probe.mojo`). Every run whose
reach term outweighs its goal term diverged; the one where it does not is the
only configuration that has ever held.

⚠ MAGNITUDE AND SHAPE ARE BOTH RULED OUT BY RUN 12, which matched the stable
|r| of 0.105 exactly (0.10 measured) with the tolerance shape and diverged
anyway — `mean_q` to 2008 against a fixed point of 10.2.

⚠ THE MECHANISM IS A HYPOTHESIS OVER FOUR POINTS. `|gripper - subject|` moves
as fast as the arm; the goal term is a separation between two props that only
changes on CONTACT. A bootstrap target dominated by the fast-varying term has
the variance of the fast one, and a noisy target is what a critic chases.

⚠ AND THE SHAPE IS EARNING ITS KEEP REGARDLESS. Run 11's eval peaked at 130
against a random 69 — 1.9x — where the linear form's best was 13% over its own
baseline in 290k steps. Both tolerance runs swung wildly with no trend, which
is what a policy riding a diverging value function looks like.

⚠ AND A HEALTHY RUN HAS A SHAPE. What to read FIRST is `mean_q`: it should
converge toward `mean_reward / (1 - gamma)` with `mean_next_q - mean_q` under
a tenth. Run 3's gap was +5 and its `mean_q` ran to 508; run 10's was +35 at
5743. That is a critic chasing itself, and no return moves under one.

⚠ SO THIS FILE PRINTS TWO NUMBERS. `mean_return` is what SAC optimises and
moves smoothly; the SUCCESS RATE is measured separately by
`greedy_success_rate` — one greedy episode per lane, counting `reward > 0.5`
— reading `META_IDX_GOAL_HELD`, the word the reward hook writes. `tests/tasks/test_goal_distance.mojo` asserts that bound.

⚠ SET `SHAPE_W_GOAL` AND `SHAPE_W_REACH` TO 0.0 to get the sparse reward
back. Every success-rate baseline above was measured there, and a shaped run
is not comparable with a sparse one on RETURN — only on the rate.

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

from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.ptr import mptr
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.linear_relu import LinearReLU
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import CsvLogger, RemoteLogger, CompositeLogger
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.sac import SACAgent
from mojo_rl.deep_agents.training.blocks import UniformSampleGpuStep
from mojo_rl.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from mojo_rl.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_TASK_PARAM_0, META_IDX_TASK_ACTIVE,
    META_IDX_INIT_REGION_0, META_IDX_GOAL_HELD, MODEL_CURRICULUM_SIZE,
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


# ⚠⚠ 32, MATCHING THE TWO EXAMPLES THAT TRAIN ON THIS STACK.
# `sac_so_arm101_reach_training_gpu.mojo` and
# `sac_half_cheetah_training_gpu.mojo` both run 32 with
# `updates_per_step = N_ENVS`, which is UTD 1 at a target tracking rate of
# 14.8%. This file ran 64 with 64 — also UTD 1, but 27.4% tracking — and the
# critic diverged in four consecutive runs before the tracking rate was the
# thing anybody looked at.
#
# ⚠ COMPTIME, so this is an edit and not a flag: `N_ENVS` sizes `Layout`
# parameters and the greedy eval's lane count. `--updates-per-step` is the
# flag, and it is what varies the tracking rate WITHOUT changing how much
# data an iteration collects.
comptime N_ENVS = 32
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

def baselines_for(task: String) -> Tuple[Float64, Float64, Bool]:
    """`(random, untrained_greedy, measured)` for a task — see the header.

    ⚠⚠ PER TASK, AND IT WAS A PAIR OF CONSTANTS. This file trained one task
    when those were written; `--task` made them a lie, and the first `gather`
    run on a 5090 printed `lift`'s floors under `gather`'s rate. A verdict
    that names the wrong baseline is worse than none, because it reads as
    though somebody checked.

    ⚠ THE THIRD FIELD IS "HAS THIS BEEN MEASURED". A task nobody has run a
    baseline for gets `False` and the verdict says so, rather than defaulting
    to 0.0 — which is a real claim, and the flattering one.
    """
    if task == "so101_lift_brick":
        return (0.00, 0.00, True)
    if task == "so101_gather_bricks":
        # ⚠ NOT EXACTLY ZERO. Two 20k warmup-only runs gave 0.000 and 0.0156
        # (one lane of 64), so random DOES occasionally push the blocks
        # together. 0.02 is the ceiling of what was seen and is what a rate
        # has to beat before it means anything.
        #
        # ⚠ THESE ARE SUCCESS RATES AND ARE LANE-COUNT INDEPENDENT, unlike the
        # shaped RETURN — a rate is per episode either way. The return
        # baselines in the header are not, and mixing the two cost two rounds.
        return (0.02, 0.00, True)
    if task == "so101_reach_clear" or task == "so101_reach_brick":
        return (0.25, 1.00, True)
    if task == "so101_settle_brick":
        return (1.00, 1.00, True)
    return (0.0, 0.0, False)
# ⚠⚠ PER TASK, AND IT WAS NOT. Both of these were fixed strings from when
# this file trained one task, so the first `gather` run on a 5090 wrote
# `sac_task_reach.ckpt` — and a `lift` run after it would have OVERWRITTEN
# that checkpoint with weights for a different task, silently, under a name
# naming a third. `--task` made the name a lie; these make it the task's.
comptime CKPT_PREFIX = "sac_task_"
comptime CSV_PREFIX = "/tmp/mojo_rl_sac_"

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

comptime AgentT = SACAgent[
    "gpu",
    UniformSampleGpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
    ActorNet,
    CriticNet,
]


def greedy_success_rate(
    mut agent: AgentT, mut env: EnvT, ctx: DeviceContext
) raises -> Float64:
    """Fraction of lanes whose goal is met at ANY step of one greedy episode.

    ## ⚠⚠ WHY THIS EXISTS: THE RETURN STOPPED BEING THE SUCCESS RATE

    Sparse, with termination on success, an episode return was exactly 0 or 1
    and `agent.mean_return()` WAS the rate — no band table, no decoding. The
    shaped terms in `So101TabletopConfig.custom_reward_gpu` subtract a
    per-step penalty from that same scalar, so a return is now dominated by
    integrated distance and says nothing directly about success. The driver's
    own greedy eval returns that shaped mean too.

    ⚠⚠ SUCCESS COMES FROM `META_IDX_GOAL_HELD`, NOT FROM THE REWARD. It used
    to be `reward > 0.5`, which held only while the reward was `+1 if holds`
    minus a penalty capped below 0.5. The reward is two `tolerance` terms in
    [0, 1] now and carries no success bonus, so that test would count a lane
    hovering near the blocks as solved. The hook writes the bit; every reader
    reads the bit.

    ⚠ NO `selective_reset_batch` IN THE LOOP, deliberately. A lane that
    succeeds TERMINATES and then keeps stepping with its done flag set; what
    is being counted is "did this lane ever meet its goal in one episode", so
    resetting mid-window would let one lane contribute twice and inflate the
    rate above what an episode is worth.
    """
    comptime AO = 2 * ACT_DIM
    var ao = ctx.enqueue_create_buffer[DT](N_ENVS * AO)
    env.reset_batch[N_ENVS](ctx, UInt64(20260907))

    var solved = List[Bool](length=N_ENVS, fill=False)
    var rew_h = List[Scalar[DT]](length=N_ENVS, fill=Scalar[DT](0))

    for step in range(So101TabletopConfig.MAX_STEPS):
        agent.trainer.select_greedy_action_batched[N_ENVS](
            ctx,
            LayoutTensor[DT, Layout.row_major(N_ENVS, OBS_DIM), MutAnyOrigin](
                env.obs_ptr()
            ),
            LayoutTensor[DT, Layout.row_major(N_ENVS, ACT_DIM), MutAnyOrigin](
                env.action_ptr()
            ),
            LayoutTensor[DT, Layout.row_major(N_ENVS, AO), MutAnyOrigin](
                mptr(ao.unsafe_ptr())
            ),
        )
        env.step_batch[N_ENVS](ctx, UInt64(step + 1))
        # ⚠⚠ THE GOAL BIT, NOT THE REWARD. This read `reward > 0.5`, which
        # was the same signal only while the reward was `+1 if holds` minus a
        # bounded penalty. The reward is two `tolerance` terms now and carries
        # no success bonus at all, so that test would count a lane hovering
        # near the blocks as a success.
        env.d.meta.download(ctx)
        ctx.synchronize()
        for e in range(N_ENVS):
            if Float64(
                env.d.meta.data[e * METADATA_SIZE + META_IDX_GOAL_HELD]
            ) > 0.5:
                solved[e] = True

    var n = 0
    for e in range(N_ENVS):
        if solved[e]:
            n += 1
    return Float64(n) / Float64(N_ENVS)


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
    # ⚠⚠ FLAGS BECAUSE THESE TWO ARE WHAT A FLAT RUN ACTUALLY NEEDS SWEPT.
    # Measured on a 230k-step `gather` run: `mean_q` reached **1151** while the
    # true episode return was **-7.5** — wrong sign, and 11x beyond the most
    # the task can pay even if every one of 300 steps scored the +1 success
    # bonus. The critic had diverged, and it was tracking the ENTROPY term,
    # not the task: corr(alpha, mean_q) = 0.88 over 111 points, with
    # mean_q ~ 2500 * alpha throughout.
    #
    # The task's own contribution to Q is r/(1-gamma) = -0.025 * 95 = -2.4, so
    # at alpha = 0.26 the task was about 0.2% of the value function the actor
    # was maximising. That is a REWARD SCALE problem — SAC's per-step reward
    # here is 0.025 where HalfCheetah's is O(1..10) — and the two levers that
    # reach it without changing what a reward MEANS are the entropy target and
    # the initial temperature.
    #
    # ⚠ `-ACT_DIM` IS THE STANDARD HEURISTIC AND IT IS WHAT FORCES ALPHA UP.
    # Making it more negative (-12, -24) drives alpha down and shrinks the
    # entropy contribution to Q. Falsifiable in a short run: if the mechanism
    # is right, `mean_q` should fall roughly in proportion.
    var target_entropy = -Scalar[DT](ACT_DIM)
    var init_alpha = Scalar[DT](0.2)
    # ⚠⚠ 0.50/0.25 IS THE ONLY PAIR THE CRITIC HAS SURVIVED. Measured, all at
    # 32 envs / 32 updates / tau 0.0025 — the SAME 7.7% tracking rate:
    #
    #     0.50 / 0.25   mean_q -> -6.2 converging, alpha -> 0.0014, 290k steps
    #     0.10 / 0.70   mean_q -> 5743, alpha -> 2.2, critic_loss 17640
    #
    # So the tracking rate is NECESSARY AND NOT SUFFICIENT, and the weights
    # are the other axis. Hypothesis fitted to two points, recorded as such:
    # the reach term is `|gripper - subject|` and moves as fast as the arm,
    # while the goal term is a separation between two props that barely moves,
    # so weighting the fast one at 7x the slow one raises the target's
    # step-to-step variance. What is MEASURED is only that 0.10/0.70 diverges
    # where 0.50/0.25 does not, at identical everything else.
    #
    # ⚠⚠ THE REWARD SCALE IS THE OPEN QUESTION ON THIS FAMILY, so it is a flag.
    # Three 190k-step runs at 0.10/0.05 held `mean_reward` at -0.024 from the
    # first diagnostic sample to the last — through an alpha fix and an
    # observation widening — while `mean_q` ran to 508. With `mean_done` at
    # 8e-05 nothing anchors the value function except the reward, and 0.024
    # per step is 200x too small to. `SoArm101ReachConfig`, which DOES train
    # on this robot, pays a `tolerance` in [0, 1] every step.
    var shape_goal = So101TabletopConfig.SHAPE_W_GOAL
    var shape_reach = So101TabletopConfig.SHAPE_W_REACH
    # ⚠⚠ THE TARGET NETWORK'S TRACKING RATE, WHICH `N_ENVS` SETS BY ACCIDENT.
    # `updates_per_step = N_ENVS` keeps UTD at 1 — 64 transitions collected,
    # 64 gradient steps — and that is the number people quote. It is not the
    # number that governs the CRITIC's stability. Polyak runs ONCE PER UPDATE
    # at `tau`, so a driver iteration moves the target by
    #
    #     1 - (1 - tau)^updates_per_step
    #
    # which at tau 0.005 is 27% for 64 updates and 15% for 32. A target that
    # moves a quarter of the way to the online net between env steps is barely
    # a target, and chasing it is what a runaway critic looks like:
    # `mean_next_q` sat about +5 above `mean_q` at EVERY sample of run 3.
    #
    # ⚠ THIS IS WHY 64 ENVS VS 32 IS NOT THE NO-OP I SAID IT WAS. The UTD
    # argument was right and the conclusion was wrong — the two examples that
    # DO train on this stack (`sac_so_arm101_reach_training_gpu.mojo` and
    # `sac_half_cheetah_training_gpu.mojo`) both run 32.
    # ⚠⚠ `N_ENVS` AGAIN, WHICH IS UTD 1 — BUT AT 32 ENVS, NOT 64. The
    # measured history, same task and same everything else:
    #
    #                    64 env / 64 upd     64 env / 16 upd
    #                    (27.4% tracking)    (7.7% tracking)
    #     mean_q          0 -> 14820          0 -> -6.63, converging
    #     next_q - q      about +5            +0.048
    #     critic_loss     190800              0.0035
    #     mean_reward     -0.115, FLAT        -0.112 -> -0.0889, improving on
    #                                         70 of 79 samples
    #     eval return     -25 .. -51          -28.7 -> -20.1
    #
    # The second column trains and costs sample efficiency: UTD 16/64 = 0.25.
    # 32 envs with 32 updates is UTD 1 at 14.8% tracking — between the two,
    # and exactly what the working references use. ⚠ THAT COMBINATION IS NOT
    # YET MEASURED ON THIS FAMILY; the 7.7% column is. If the critic diverges
    # again, `--updates-per-step 16` is the configuration known to hold.
    var updates_per_step = N_ENVS
    # ⚠⚠ 0.0025, NOT SAC'S USUAL 0.005, BECAUSE 32 UPDATES OF 0.005 IS 14.8%
    # AND THIS FAMILY DIVERGES THERE. Lowering `tau` rather than the update
    # count is what buys the safe tracking rate WITHOUT paying UTD:
    #
    #     32 upd, tau 0.005    14.8%   UTD 1.00   mean_q -> 1152, diverged
    #     16 upd, tau 0.005     7.7%   UTD 0.50   mean_q -> -6.63, converged
    #     32 upd, tau 0.0025    7.7%   UTD 1.00   mean_q -> -3.10, converged
    #
    # ⚠ THE THIRD ROW IS WHAT SETTLED IT. Rows one and two differ in TWO
    # things, so neither could say whether the tracking rate or the UTD was
    # the axis; row three holds UTD at 1 and moves only the rate, and the
    # critic came back healthy — `next_q - q` +0.18 against +5.85 at 14.8%,
    # `critic_loss` 0.013 against 1667. The tracking rate is the axis.
    var tau = Scalar[DT](0.0025)
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
        elif a == "--target-entropy" and i + 1 < len(args):
            target_entropy = Scalar[DT](Float64(String(args[i + 1])))
        elif a == "--alpha" and i + 1 < len(args):
            init_alpha = Scalar[DT](Float64(String(args[i + 1])))
        elif a == "--shape-goal" and i + 1 < len(args):
            shape_goal = Float64(String(args[i + 1]))
        elif a == "--shape-reach" and i + 1 < len(args):
            shape_reach = Float64(String(args[i + 1]))
        elif a == "--updates-per-step" and i + 1 < len(args):
            updates_per_step = Int(String(args[i + 1]))
        elif a == "--tau" and i + 1 < len(args):
            tau = Scalar[DT](Float64(String(args[i + 1])))

    print("=" * 72)
    print("SAC on the task family —", task_name, "(GPU)")
    var ckpt_path = String(CKPT_PREFIX) + task_name + ".ckpt"
    var csv_path = String(CSV_PREFIX) + task_name + ".csv"
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
    print("  target_entropy:", target_entropy, " init_alpha:", init_alpha)
    print("  shape weights: goal", shape_goal, " reach", shape_reach,
          " (tolerance margins", So101TabletopConfig.GOAL_MARGIN, "/",
          So101TabletopConfig.REACH_MARGIN, "m)")
    # ⚠ THE NUMBER THAT ACTUALLY GOVERNS CRITIC STABILITY, printed because it
    # is derived and nobody sets it directly.
    var track = 1.0 - (1.0 - Float64(tau)) ** Float64(updates_per_step)
    print("  updates/step:", updates_per_step, " tau:", tau,
          " -> the target moves", track, "per iteration")
    print("  UTD:", Float64(updates_per_step) / Float64(N_ENVS))
    # ⚠⚠ THE KNOWN-BAD BAND, MEASURED ON THIS FAMILY. Three runs, same task:
    #
    #     27.4%  (64 env / 64 upd)   mean_q -> 14820, mean_reward FLAT
    #     14.8%  (32 env / 32 upd)   mean_q -> 1152,  next_q-q +5.9, no trend
    #      7.7%  (64 env / 16 upd)   mean_q -> -6.63 converging, reward moving
    #
    # So the threshold is somewhere in (7.7%, 14.8%] and it is BELOW what the
    # working references use — `sac_so_arm101_reach_training_gpu.mojo` trains
    # at 14.8%, this family does not. Warned rather than refused, because the
    # rate is a legitimate sweep axis and the band is three points wide, not a
    # law.
    #
    # ⚠ TO KEEP UTD AT 1 AND STILL LAND AT 7.7%, LOWER `tau`, NOT THE UPDATE
    # COUNT: 32 updates at tau 0.0025 is 7.70%, which is the tracking rate
    # that held and the sample efficiency that 16-updates gave away.
    if track > 0.10:
        print()
        print("  ⚠⚠ THE TARGET MOVES", track, "PER ITERATION, AND THIS FAMILY")
        print("  HAS DIVERGED ABOVE 0.10 IN EVERY RUN SO FAR (14.8% and")
        print("  27.4%; 7.7% converged). Watch `mean_q` against")
        print("  `mean_reward / (1 - gamma)` — if it passes zero the run is")
        print("  already lost. `--tau 0.0025` at", updates_per_step,
              "updates gives 7.7% with UTD unchanged.")
        print()

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
        if iw[j] > 0.0:      # the word is region_index + 1; 0 = not placed
            n_active_free += 1
    print("  free slots placed at reset:", n_active_free, "of", len(iw))

    with DeviceContext() as ctx:
        # ── the logger: CSV always, the dashboard when it is configured ──
        #
        # ⚠⚠ A SPARSE 0/1 RETURN TELLS YOU ALMOST NOTHING WHILE IT IS ZERO,
        # which is exactly the regime this task sits in. `diag_every` flushes
        # the SAC bundle — `mean_q`, `critic_loss`, `actor_loss`, `alpha`,
        # `mean_reward`, `train_steps` — and those move long before the return
        # does: a critic whose `mean_q` is drifting up has found SOMETHING to
        # predict, and an `alpha` pinned at its ceiling says the actor is
        # still being paid to be random. Without them a flat return is
        # indistinguishable from a broken reward.
        #
        # ⚠ `RemoteLogger` WITH NO URL IS INERT — its POST sink is built
        # lazily on the first payload — so this is safe with no `.env` and
        # costs nothing. The CSV is the local artefact that survives the
        # dashboard being down and is what a later run gets diffed against.
        var env_vars = load_dotenv()
        var remote = RemoteLogger(
            server_url=env_vars.get("RL_MONITOR_URL", ""),
            run_name=String("SAC task ") + task_name,
            buffer_size=64,
            api_key=env_vars.get("RL_MONITOR_API_KEY", ""),
        )
        remote.set_config("algorithm", "SAC")
        remote.set_config("family", f.name)
        remote.set_config("task", task_name)
        remote.set_config("goal", t.goal)
        remote.set_config("language", t.language)
        remote.set_config("target", "gpu")
        remote.set_config("n_envs", String(N_ENVS))
        remote.set_config("hidden", String(HIDDEN))
        remote.set_config("batch", String(BATCH))
        remote.set_config("warmup", String(warmup))
        remote.set_config("horizon", String(So101TabletopConfig.MAX_STEPS))
        remote.set_config("action_scale", String(ACTION_SCALE))
        remote.set_config("target_entropy", String(target_entropy))
        remote.set_config("init_alpha", String(init_alpha))
        remote.set_config("shape_w_goal", String(shape_goal))
        remote.set_config("shape_w_reach", String(shape_reach))
        remote.set_config("updates_per_step", String(updates_per_step))
        remote.set_config("tau", String(tau))
        remote.set_config("target_track_per_iter", String(track))
        # ⚠ THE MEASURED FLOOR TRAVELS WITH THE RUN. A rate on a dashboard is
        # unreadable without it — 0.05 is nothing on `reach` and would be real
        # on `lift` — and a config field is the only part of a run that is
        # still there when somebody opens the chart a week later.
        var bl0 = baselines_for(task_name)
        remote.set_config("baseline_random", String(bl0[0]))
        remote.set_config("baseline_untrained_greedy", String(bl0[1]))
        remote.set_config("baseline_measured", String(bl0[2]))
        var logger = CompositeLogger(CsvLogger(csv_path), remote)

        # ⚠⚠ THE CONFIG ALSO GOES OUT AS SCALARS AT STEP 0, SO THE CSV IS
        # SELF-DESCRIBING. `set_config` reaches the dashboard and NOT the CSV
        # — `CsvLogger` writes `step,wall_time_ms,name,value` and has nowhere
        # to put a config field — so a CSV read later carries the curves and
        # none of the settings that produced them.
        #
        # That cost a real conclusion. Two runs' shaped costs were decomposed
        # into a goal distance and a reach distance under ASSUMED weights, and
        # the two decompositions were mutually inconsistent: solving the pair
        # gives a goal distance of -0.030 m, which is impossible. The
        # arithmetic was fine; one of the weights I assumed was not what ran,
        # and nothing in the file could say so.
        #
        # ⚠ AS `cfg/*` SO THEY SORT TOGETHER and cannot collide with a metric
        # name. Emitted once, at step 0, before anything else is logged.
        logger.log_scalar(String("cfg/shape_w_goal"), shape_goal, 0)
        logger.log_scalar(String("cfg/shape_w_reach"), shape_reach, 0)
        logger.log_scalar(
            String("cfg/goal_margin"), So101TabletopConfig.GOAL_MARGIN, 0
        )
        logger.log_scalar(
            String("cfg/reach_margin"), So101TabletopConfig.REACH_MARGIN, 0
        )
        logger.log_scalar(
            String("cfg/target_entropy"), Float64(target_entropy), 0
        )
        logger.log_scalar(String("cfg/init_alpha"), Float64(init_alpha), 0)
        logger.log_scalar(String("cfg/tau"), Float64(tau), 0)
        logger.log_scalar(
            String("cfg/updates_per_step"), Float64(updates_per_step), 0
        )
        logger.log_scalar(String("cfg/n_envs"), Float64(N_ENVS), 0)
        logger.log_scalar(String("cfg/warmup"), Float64(warmup), 0)
        logger.log_scalar(String("cfg/obs_dim"), Float64(OBS_DIM), 0)
        logger.log_scalar(String("cfg/max_steps"),
                          Float64(So101TabletopConfig.MAX_STEPS), 0)
        logger.log_scalar(String("cfg/target_track_per_iter"), track, 0)

        var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

        var agent = AgentT(
            ctx=ctx,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            gamma=0.99,
            tau=tau,
            action_scale=ACTION_SCALE,
            init_alpha=init_alpha,
            target_entropy=target_entropy,
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
            rheights[0], shape_goal, shape_reach,
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
            L=CompositeLogger[CsvLogger, RemoteLogger],
        ](
            env,
            num_steps,
            rng_seed=UInt64(42),
            updates_per_step=updates_per_step,
            print_every=PRINT_EVERY,
            verbose=True,
            logger=logger_ptr,
            diag_every=DIAG_EVERY,
            episode_sync_every=32,
            checkpoint_every=CHECKPOINT_EVERY,
            checkpoint_path=ckpt_path,
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
        # ⚠ `close()` IS NOT OPTIONAL on the remote half — it drains the queue
        # and joins the POST thread, and whatever is still queued at process
        # exit is otherwise lost. It also prints the sink's drop tally.
        logger.close()
        _ = logger        # keeps `logger_ptr` alive to here

        # ⚠⚠ TWO DIFFERENT NUMBERS NOW, AND THEY USED TO BE ONE. `mean_return`
        # is the SHAPED return — dominated by integrated distance, and what
        # SAC actually optimises. The success rate has to be measured, and
        # `greedy_success_rate` is that measurement: one greedy episode per
        # lane, counting `reward > 0.5`.
        var shaped = Float64(agent.mean_return())
        var rate = greedy_success_rate(agent, eval_env, ctx)
        print("-" * 72)
        print("  env steps          :", num_steps)
        print("  elapsed            :", secs, "s")
        print("  episodes           :", agent.ep_count())
        print("  shaped mean return :", shaped, "(last 100 episodes)")
        print("  SUCCESS RATE       :", rate, "(greedy,", N_ENVS, "lanes)")
        print("  csv                :", csv_path)
        print("  checkpoint         :", ckpt_path)

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
        var bl = baselines_for(task_name)
        if not bl[2]:
            print("  ⚠⚠ NO BASELINE MEASURED for", task_name, "— run it with")
            print("  `--warmup >= --steps` first. A rate with nothing to")
            print("  compare it to is not a result.")
            print("=" * 72)
            return
        print("  baselines for", task_name, "— random", bl[0],
              " untrained greedy", bl[1])

        # ⚠⚠ A RATE OVER n EPISODES HAS A STANDARD ERROR, AND AT THESE n IT IS
        # THE SAME SIZE AS THE EFFECT. The window is 100 episodes, so a
        # baseline of 0.02 carries se = sqrt(p(1-p)/n) = 0.014 — and a reading
        # of 0.05 is 2 se above it, which is suggestive and is NOT a result.
        # Printing the interval is what stops a noisy tick being read as a
        # curve; the first `gather` run oscillated 0.00 .. 0.05 for 125k steps
        # and every one of those values sits inside this band.
        # ⚠ THE DENOMINATOR IS THE GREEDY EVAL'S LANE COUNT, not the training
        # window — `rate` above comes from N_ENVS greedy episodes and the band
        # has to be the band for THAT n.
        var n = Float64(N_ENVS)
        var p = bl[0]
        var se = 0.0
        if n > 0.0:
            se = (p * (1.0 - p) / n) ** 0.5
        print("  baseline se over", Int(n), "greedy episodes:", se,
              " -> 2-sigma band ends at", p + 2.0 * se)

        if rate <= p + 2.0 * se:
            print("  FLAT — the rate is inside the random baseline's 2-sigma")
            print("  band. ⚠ READ THE SHAPED RETURN BEFORE CONCLUDING")
            print("  ANYTHING: it is dense, so it moves long before the rate")
            print("  does. Random actions score", -3.996, "on `gather`; a run")
            print("  climbing toward 0 is learning to close the distance even")
            print("  with no successes yet. If the shaped return is ALSO flat,")
            print("  the shaping is not reaching the policy — check `mean_q`")
            print("  and `alpha` in the logger before touching anything else.")
        else:
            print("  the rate is ABOVE the baseline's 2-sigma band:", rate)
        print("=" * 72)
