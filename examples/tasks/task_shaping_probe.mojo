"""WHAT THE SHAPED TERMS ACTUALLY MEASURE, UNDER A RANDOM POLICY — CPU.

    pixi run mojo run -I . examples/tasks/task_shaping_probe.mojo [task]

## ⚠⚠ WHY THIS EXISTS: A CLIP IN THE WRONG PLACE IS AN INVISIBLE FLAT SPOT

`So101TabletopConfig`'s reward subtracts `SHAPE_W_GOAL * min(goal_dist, CLIP)`
and `SHAPE_W_REACH * min(reach_dist, CLIP)`. Beyond `CLIP` each term is
CONSTANT — the gradient is exactly zero — so if the clip sits inside the state
distribution the policy gets no signal over the part of the space it actually
occupies, and the curve plateaus for a reason nothing in the logs names.

Whether that is happening is a question about DISTANCES, and no run's metrics
answer it: `mean_reward` is the weighted sum of the two terms, so one number
carries both and neither can be recovered from it.

⚠⚠ I TRIED TO RECOVER THEM ANYWAY AND IT WAS UNSOUND. Two runs share a seed
and a warmup, so their uniform-random phase is the same trajectories, and two
cost equations in two unknowns should have pinned the pair. Solving them gave
`goal_dist = -0.030 m`, which is impossible. Rather than a fourth round of
algebra on two aggregates, this measures the two distances directly.

## WHAT IT PRINTS

Per-step distributions over one random-action episode, for both shaped terms,
against the clip — and the per-step cost the reward would charge, so it can be
checked against a run's own `mean_reward`.
"""

from std.random import random_float64
from std.sys import argv

from mojo_rl.tasks.shaping import optimal_margin
from mojo_rl.tasks.spec import (
    load_family, load_task, validate_task_against_family,
)
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.family_config import So101TabletopConfig
from mojo_rl.tasks.so101_tabletop_xml import So101TabletopModel
from mojo_rl.tasks.predicates import parse_goal, bind_goal, slot_body_id
from mojo_rl.tasks.eval import region_sites
from mojo_rl.tasks.active import active_mask
from mojo_rl.tasks.tape import encode_goal, TAPE_WORDS
from mojo_rl.tasks.sampler import sample_placements, RegionFrame, SampleReport
from mojo_rl.tasks.reset import free_slot_addresses, reset_slots
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_TASK_PARAM_0, META_IDX_TASK_ACTIVE,
)
from mojo_rl.physics3d.parser.runtime_load import parse_model_runtime
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.core.cont_action import ContAction
from mojo_rl.envs.dm_control.rewards import (
    tolerance, SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN,
)
from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.tasks.gpu_eval import region_table_words, tape_distance_gpu
from mojo_rl.tasks.eval import region_rects, region_half_heights
from mojo_rl.physics3d.gpu.constants import (
    MODEL_CURRICULUM_SIZE, METADATA_SIZE,
)
from layout import Layout


comptime DT = DType.float64
comptime EnvT = Phyics3dEnv[So101TabletopModel, So101TabletopConfig, DT]
comptime NQ = So101TabletopModel.NQ
comptime NV = So101TabletopModel.NV
comptime CFG = So101TabletopConfig
comptime SEED = 3


def margin_report(
    label: String, dist: Float64, radius: Float64, margin: Float64
) raises:
    """What `margin` is worth against a term that starts at `dist`.

    ## ⚠⚠ TWO NUMBERS DECIDE A MARGIN AND NEITHER IS THE DISTANCE

    **The term at reset** is what the policy is handed for doing nothing. A
    `tolerance` band wide against the distance to be closed pays out before
    the episode starts: at margin 0.10 against `lift`'s 0.030 m z-shortfall
    the goal term is 0.81, so 81% of it is free and only 0.19 is left to
    train on.

    **The gradient** is what it can climb. It is NOT monotone in the margin —
    it peaks and falls away on both sides, because a wide band is flat near
    zero and a narrow one is flat everywhere outside itself:

        margin   term@reset   gradient/m     (lift, shortfall 0.030 m)
        0.020        0.006          1.9      too narrow — dead band
        0.046        0.376         24.5      the peak
        0.050        0.437         24.1
        0.100        0.813         11.2      too wide — mostly free
        0.150        0.912          5.6

    The peak sits where `k * dist / margin == sqrt(2)`, with `k =
    sqrt(-2 ln(value_at_margin))` — which is `margin ≈ 1.518 * dist` for the
    default `value_at_margin`, and lands the term at exp(-1) = 0.368. Both
    conditions at once, so there is no trade to make.

    ⚠ IT IS A STARTING POINT, NOT A RULE. The optimum is computed at the
    RESET distance, and the gradient a policy actually needs is the one along
    the path it takes. A term with no headroom cannot train regardless.
    """
    # ⚠ THE BAND IS SUBTRACTED FIRST. `tolerance` is 1 inside
    # `[0, radius]`, so the distance that has to be closed is the SHORTFALL
    # past the radius, not the raw distance — using the raw one would move
    # the recommended margin by the radius.
    var d = dist - radius
    if d <= 0.0:
        print("  ", label, "margin: already inside the radius at reset —"
              " the term is 1.0 and this half of the shaping is off")
        return
    var t = Float64(
        tolerance[SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DT](
            Scalar[DT](dist), Scalar[DT](0), Scalar[DT](radius),
            Scalar[DT](margin),
        )
    )
    var h = 1e-5
    var tp = Float64(
        tolerance[SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DT](
            Scalar[DT](dist + h), Scalar[DT](0), Scalar[DT](radius),
            Scalar[DT](margin),
        )
    )
    var tm = Float64(
        tolerance[SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DT](
            Scalar[DT](dist - h), Scalar[DT](0), Scalar[DT](radius),
            Scalar[DT](margin),
        )
    )
    var grad = (tp - tm) / (2.0 * h)
    var best = optimal_margin(d)
    print("  ", label, "margin", margin, "-> term at reset", t,
          " headroom", 1.0 - t, " gradient", grad, "/m")
    print("        gradient-optimal margin for a", d, "m shortfall is",
          best, "(term at reset 0.368)")
    if t > 0.6:
        print("        ⚠ MORE THAN 0.6 OF THIS TERM IS FREE AT RESET. The"
              " margin is wide against the distance to be closed, so most of"
              " the term is paid before the policy acts.")


def main() raises:
    var task_name = String("so101_gather_bricks")
    var a = argv()
    if len(a) > 1:
        task_name = String(a[1])

    print("=" * 72)
    print("what the shaped terms measure —", task_name, ", random actions")
    print("=" * 72)

    var f = load_family("mojo_rl/tasks/families/so101_tabletop.family")
    var t = load_task("mojo_rl/tasks/tasks/" + task_name + ".task")
    validate_task_against_family(t, f)
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var g = bind_goal(parse_goal(t.goal), f, fmd.body_names, fmd.site_names)
    var tape = encode_goal(g)
    var mask = active_mask(t, f)

    # ⚠ THE SUBJECT IS TERM 0's `a`, WHICH IS WHAT THE REWARD READS. Resolved
    # the same way here — out of the bound goal — so this cannot drift from
    # `custom_reward_gpu`'s `meta[TASK_PARAM_0 + 1]`.
    var subj = g.terms[0].a
    var other = g.terms[0].b
    print("  goal :", t.goal)
    print("  reach term subject: body", subj, "=", fmd.body_names[subj])

    var env = EnvT()
    _ = env.reset()
    for w in range(TAPE_WORDS):
        env.d.meta.data[META_IDX_TASK_PARAM_0 + w] = Scalar[DT](tape[w])
    env.d.meta.data[META_IDX_TASK_ACTIVE] = Scalar[DT](mask)

    var jt = List[Int]()
    var jq = List[Int]()
    var jv = List[Int]()
    for i in range(len(fmd.joints)):
        jt.append(fmd.joints[i].jnt_type)
        jq.append(fmd.joints[i].nq)
        jv.append(fmd.joints[i].nv)
    var addrs = free_slot_addresses(f, fmd.joint_names, jt, jq, jv)
    var sp0 = List[Float64]()
    for i in range(len(fmd.site_names) * 3):
        sp0.append(Float64(env.d.site_xpos.data[i]))
    var frames = List[RegionFrame]()
    for i in range(len(f.regions)):
        var rs = rsites[i]
        frames.append(RegionFrame(sp0[rs * 3], sp0[rs * 3 + 1], sp0[rs * 3 + 2]))
    var radii = List[Float64]()
    for _ in range(len(f.slots)):
        radii.append(CFG.SLOT_RADIUS)
    var rep = SampleReport()
    var placed = sample_placements(t, f, frames, radii, UInt64(SEED), 0, rep)
    var q0 = List[Float64]()
    for i in range(NQ):
        q0.append(Float64(env.d.qpos.data[i]))
    var v0 = List[Float64]()
    for _ in range(NV):
        v0.append(0.0)
    reset_slots(t, f, placed, addrs, q0, v0)
    for i in range(NQ):
        env.d.qpos.data[i] = Scalar[DT](q0[i])
    for i in range(NV):
        env.d.qvel.data[i] = Scalar[DT](v0[i])

    # ⚠⚠ THE GOAL DISTANCE COMES FROM `tape_distance_gpu`, THE FUNCTION THE
    # KERNEL USES. This file computed `|subject - other| - param` inline — a
    # 3D body-to-body distance — which is `Near`'s rule and ONLY `Near`'s.
    # `Above`'s shortfall is z-only, and `On`/`In`/`AtRegion` take a REGION
    # index as their second argument, so the inline form read body 0 (the
    # world) and reported 0.33 m for `so101_settle_brick`, whose goal HOLDS AT
    # RESET and whose distance is therefore zero. The margins in
    # `So101TabletopConfig` were set from those numbers.
    #
    # ⚠ THE SAME DEFECT THIS TREE KEEPS PAYING FOR: a rule written inline
    # beside the one place that already owns it.
    comptime NB = So101TabletopModel.NBODY
    comptime NS = So101TabletopModel.NSITE
    comptime L_META = Layout.row_major(1, METADATA_SIZE)
    comptime L_CUR = Layout.row_major(1, MODEL_CURRICULUM_SIZE)
    comptime L_XP = Layout.row_major(1, NB * 3)
    comptime L_XQ = Layout.row_major(1, NB * 4)
    comptime L_SP = Layout.row_major(1, NS * 3)
    var t_meta = TensorImpl[DT].alloc(METADATA_SIZE)
    var t_cur = TensorImpl[DT].alloc(MODEL_CURRICULUM_SIZE)
    var t_xp = TensorImpl[DT].alloc(NB * 3)
    var t_xq = TensorImpl[DT].alloc(NB * 4)
    var t_sp = TensorImpl[DT].alloc(NS * 3)
    var rects = region_rects(f)
    var rheights = region_half_heights(f)
    var cw = region_table_words(
        rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3],
        rheights[0],
    )
    for i in range(MODEL_CURRICULUM_SIZE):
        t_cur.data[i] = Scalar[DT](cw[i])
    for w in range(TAPE_WORDS):
        t_meta.data[META_IDX_TASK_PARAM_0 + w] = Scalar[DT](tape[w])
    var n = 0
    var sum_goal = 0.0
    var sum_reach = 0.0
    var sum_goal_c = 0.0
    var sum_reach_c = 0.0
    var max_reach = 0.0
    var clipped_reach = 0
    var clipped_goal = 0

    for step in range(CFG.MAX_STEPS):
        var av = List[Float64]()
        for _ in range(6):
            av.append(random_float64() * 2.0 - 1.0)
        _ = env.step(ContAction[6].from_list(av))

        var gs = CFG.GRIPPER_SITE
        var gx = Float64(env.d.site_xpos.data[gs * 3])
        var gy = Float64(env.d.site_xpos.data[gs * 3 + 1])
        var gz = Float64(env.d.site_xpos.data[gs * 3 + 2])
        var sx = Float64(env.d.xpos.data[subj * 3])
        var sy = Float64(env.d.xpos.data[subj * 3 + 1])
        var sz = Float64(env.d.xpos.data[subj * 3 + 2])
        var ox = Float64(env.d.xpos.data[other * 3])
        var oy = Float64(env.d.xpos.data[other * 3 + 1])
        var oz = Float64(env.d.xpos.data[other * 3 + 2])

        var reach = (
            (gx - sx) ** 2 + (gy - sy) ** 2 + (gz - sz) ** 2
        ) ** 0.5
        # `Near`'s distance: |a - b| - param, floored at zero
        for i in range(NB * 3):
            t_xp.data[i] = env.d.xpos.data[i]
        for i in range(NB * 4):
            t_xq.data[i] = env.d.xquat.data[i]
        for i in range(NS * 3):
            t_sp.data[i] = env.d.site_xpos.data[i]
        var gd = Float64(
            tape_distance_gpu[DT, 1, NB, NS * 3](
                t_meta.lt["cpu", L_META](), t_cur.lt["cpu", L_CUR](),
                t_xp.lt["cpu", L_XP](), t_xq.lt["cpu", L_XQ](),
                t_sp.lt["cpu", L_SP](), 0,
            )
        )
        _ = ox
        _ = oy
        _ = oz

        sum_goal += gd
        sum_reach += reach
        if reach > max_reach:
            max_reach = reach
        sum_goal_c += gd
        sum_reach_c += reach
        n += 1
        _ = step

    var mg = sum_goal / Float64(n)
    var mr = sum_reach / Float64(n)
    var mgc = sum_goal_c / Float64(n)
    var mrc = sum_reach_c / Float64(n)
    print()
    print("  steps                :", n)
    print("  goal  distance  mean :", mg)
    print("  reach distance  mean :", mr, "  max:", max_reach)
    print()

    print()
    # ⚠⚠ WHAT THE REWARD ACTUALLY PAYS, THROUGH `tolerance` — because the
    # weights multiply a SATURATING function, not the distance. Two tasks
    # whose natural distance scales differ get very different reward
    # magnitudes from the SAME weights and the SAME margins, and `curriculum`
    # holds one weight pair for the whole batch.
    # ⚠ THE REAL `tolerance`, not a re-derivation of it — same import the
    # reward hook uses, so this cannot drift from what the kernel charges.
    var gt = Float64(
        tolerance[SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DT](
            Scalar[DT](mg), Scalar[DT](0), Scalar[DT](0),
            Scalar[DT](CFG.GOAL_MARGIN),
        )
    )
    var rt = Float64(
        tolerance[SIGMOID_GAUSSIAN, DEFAULT_VALUE_AT_MARGIN, DT](
            Scalar[DT](mr), Scalar[DT](0), Scalar[DT](CFG.REACH_RADIUS),
            Scalar[DT](CFG.REACH_MARGIN),
        )
    )
    print("  tolerance at those distances (margins", CFG.GOAL_MARGIN, "/",
          CFG.REACH_MARGIN, "):")
    print("     goal ", gt, "  reach ", rt)
    print("  reward at weights", CFG.SHAPE_W_GOAL, "/", CFG.SHAPE_W_REACH,
          ":  goal", CFG.SHAPE_W_GOAL * gt, " reach",
          CFG.SHAPE_W_REACH * rt, " total",
          CFG.SHAPE_W_GOAL * gt + CFG.SHAPE_W_REACH * rt)
    print("  reach/goal contribution ratio:",
          (CFG.SHAPE_W_REACH * rt) / (CFG.SHAPE_W_GOAL * gt))

    # ⚠⚠ WHAT THE MARGIN IS WORTH, WHICH IS THE NUMBER THIS PROBE EXISTS FOR.
    # The distances above are the input to a margin choice and the probe used
    # to stop before making it, so the choice stayed a guess — and it has been
    # wrong twice on `lift` in opposite directions. 0.02 against a 0.030 m
    # shortfall left a gradient of 1.9/m; 0.10 left the goal term at 0.81
    # BEFORE the policy moved anything, so a 1M-step run bought +0.034/step
    # and the brick never left the table.
    margin_report("goal ", mg, CFG.GOAL_RADIUS, CFG.GOAL_MARGIN)
    margin_report("reach", mr, CFG.REACH_RADIUS, CFG.REACH_MARGIN)
    print()
    print("=== MEASURED ===")
