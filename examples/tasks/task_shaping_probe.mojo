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


comptime DT = DType.float64
comptime EnvT = Phyics3dEnv[So101TabletopModel, So101TabletopConfig, DT]
comptime NQ = So101TabletopModel.NQ
comptime NV = So101TabletopModel.NV
comptime CFG = So101TabletopConfig
comptime SEED = 3


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

    comptime CLIP = 0.5  # the OLD clip, kept as a reference scale only
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
        var gd = (
            (sx - ox) ** 2 + (sy - oy) ** 2 + (sz - oz) ** 2
        ) ** 0.5 - g.terms[0].param
        if gd < 0.0:
            gd = 0.0

        sum_goal += gd
        sum_reach += reach
        if reach > max_reach:
            max_reach = reach
        var gc = gd
        if gc > CLIP:
            gc = CLIP
            clipped_goal += 1
        var rc = reach
        if rc > CLIP:
            rc = CLIP
            clipped_reach += 1
        sum_goal_c += gc
        sum_reach_c += rc
        n += 1
        _ = step

    var mg = sum_goal / Float64(n)
    var mr = sum_reach / Float64(n)
    var mgc = sum_goal_c / Float64(n)
    var mrc = sum_reach_c / Float64(n)
    print()
    print("  steps                :", n, " clip =", CLIP, "m")
    print("  goal  distance  mean :", mg, "  clipped mean:", mgc)
    print("  reach distance  mean :", mr, "  clipped mean:", mrc,
          "  max:", max_reach)
    print("  steps at the clip    : goal", clipped_goal, " reach",
          clipped_reach, "of", n)
    print()
    # ⚠⚠ THE FLAT SPOT, AS A FRACTION. Beyond the clip the term's gradient is
    # exactly zero, so this is the share of the episode over which that half
    # of the shaping says NOTHING about what to do.
    var frac = Float64(clipped_reach) / Float64(n)
    print("  reach term is FLAT on", frac, "of steps")

    print()
    print("  per-step cost this implies, by weight pair:")
    var wgs = List[Float64]()
    var wrs = List[Float64]()
    wgs.append(0.50); wrs.append(0.25)
    wgs.append(0.10); wrs.append(0.70)
    for i in range(len(wgs)):
        print("     goal", wgs[i], " reach", wrs[i], " -> ",
              wgs[i] * mgc + wrs[i] * mrc,
              "  (episode return", -(wgs[i] * mgc + wrs[i] * mrc)
              * Float64(CFG.MAX_STEPS), ")")
    print()
    print("=== MEASURED ===")
