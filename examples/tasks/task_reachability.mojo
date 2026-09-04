"""IS `so101_reach_brick` REACHABLE, AND HOW BIG IS THE TARGET? — CPU.

    pixi run mojo run -I . examples/tasks/task_reachability.mojo

## ⚠⚠ THE QUESTION THIS ANSWERS IS "CAN A SPARSE POLICY EVER SEE REWARD"

`so101_reach_brick`'s goal is `AtRegion(robot_gripperframe, table_top)` and
its reward is SPARSE: one bit, at the end of a 300-step horizon, or nothing.
A sparse task is trainable only if random exploration stumbles into the goal
often enough to bootstrap, so the number that decides whether to train it is
not "is the goal satisfiable" but **what fraction of the arm's configuration
space satisfies it**. A target set of 1e-6 of the space is satisfiable and
untrainable; the two answers look identical to every gate written so far.

So this measures the FRACTION, uniformly over the arm's joint limits, and
prints the closest approach beside it so a zero is diagnosable rather than
merely disappointing.

## ⚠ CONFIGURATION SPACE, NOT ACTION SPACE, AND THE DIFFERENCE MATTERS

A uniform draw over joint limits is not what a torque-controlled policy
explores: real exploration is a random walk in torque, filtered through the
arm's dynamics and gravity, and it does NOT cover configuration space
uniformly. So this number is an OPTIMISTIC bound — an upper bound on how easy
the task is, useful because a small value here is decisive (if the target set
is tiny in the uniform measure it will be tinier under a gravity-biased walk)
while a large value only says "worth trying".

⚠ IT IS ALSO A LOWER BOUND ON NOTHING. Do not read a healthy fraction here as
"reach will train". That question is answered by putting a policy on it, which
is the next step and not this file.

## ⚠ THE ARM ONLY. The props are parked out of the way and the draw touches
only the arm's own hinges, because the question is about the ARM's kinematics
and a prop's free joint would add six uniform dimensions that mean nothing.

## WHAT IT PRINTS

    the region box            the AtRegion acceptance volume, in world coords
    hits / draws              the fraction of uniform arm poses inside it
    closest approach          min distance from the gripper site to the box
    the reachable envelope    the gripper site's own bounding box, for context
"""

from std.random.philox import Random as PhiloxRandom
from std.sys import argv

from mojo_rl.envs.robots.so_arm101_xml import SO_ARM101_NMESH_VERTS
from mojo_rl.tasks.spec import load_family, load_task, validate_task_against_family
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.predicates import parse_goal, bind_goal
from mojo_rl.tasks.eval import (
    eval_goal, region_sites, region_rects, IN_HALF_HEIGHT,
)
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics


comptime DT = DType.float64
comptime N_DRAWS: Int = 4096
comptime JNT_HINGE: Int = 3
comptime JNT_SLIDE: Int = 2


def main() raises:
    var draws = N_DRAWS
    var a = argv()
    if len(a) > 1:
        draws = Int(String(a[1]))

    print("=" * 72)
    print("reachability of `so101_reach_brick` — uniform over the arm's joints")
    print("=" * 72)

    var f = load_family("mojo_rl/tasks/families/so101_tabletop.family")
    var t = load_task("mojo_rl/tasks/tasks/so101_reach_brick.task")
    validate_task_against_family(t, f)

    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)
    var g = bind_goal(parse_goal(t.goal), f, fmd.body_names, fmd.site_names)

    var dims = dims_from_flat(
        fmd, max_contacts=32, nmesh_verts=SO_ARM101_NMESH_VERTS
    )
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    var nq = dims.get_nq()
    var nb = dims.get_nbody()
    var ns = dims.get_nsite()

    # ── which qpos words belong to the ARM's own limited hinges ───────────
    #
    # ⚠ A LIMITED SINGLE-DOF JOINT, AND NOTHING ELSE. A free joint's seven
    # words are a prop's pose, and an UNLIMITED hinge has no range to draw
    # from — `range_min == range_max == 0` on those records, so including one
    # would pin it at zero and silently narrow the sweep.
    var q_adr = List[Int]()
    var q_lo = List[Float64]()
    var q_hi = List[Float64]()
    var adr = 0
    for j in range(len(fmd.joints)):
        ref jj = fmd.joints[j]
        var single = jj.jnt_type == JNT_HINGE or jj.jnt_type == JNT_SLIDE
        if single and jj.is_limited and jj.range_max > jj.range_min:
            q_adr.append(adr)
            q_lo.append(jj.range_min)
            q_hi.append(jj.range_max)
        adr += jj.nq
    print("  arm dofs swept:", len(q_adr), "of", nq, "qpos words")
    for i in range(len(q_adr)):
        print("     qpos[", q_adr[i], "] in [", q_lo[i], ",", q_hi[i], "]")
    if len(q_adr) == 0:
        raise Error(
            "reachability: found NO limited single-dof joint to sweep, so"
            " every draw below would be the home pose and the hit rate would"
            " report the home pose's answer " + String(draws) + " times."
        )

    # ── the acceptance box, in world coordinates ──────────────────────────
    #
    # ⚠ RESTATED FROM THE REGION, NOT FROM `eval_goal`. The hit COUNT below
    # comes from `eval_goal` — the real evaluator — and this box is only for
    # printing and for the closest-approach number. Two spellings of one
    # volume is the shape of a gate that agrees with itself, so the count
    # never reads this.
    forward_kinematics["cpu", DT, DynDims, 1](d, m)
    var ts = rsites[0]
    var sx = Float64(d.site_xpos.data[ts * 3])
    var sy = Float64(d.site_xpos.data[ts * 3 + 1])
    var sz = Float64(d.site_xpos.data[ts * 3 + 2])
    var bx0 = sx + rects[0][0]
    var by0 = sy + rects[0][1]
    var bx1 = sx + rects[0][2]
    var by1 = sy + rects[0][3]
    var bz0 = sz - IN_HALF_HEIGHT
    var bz1 = sz + IN_HALF_HEIGHT
    print()
    print("  goal:", t.goal)
    print("  AtRegion box: x [", bx0, ",", bx1, "]  y [", by0, ",", by1,
          "]  z [", bz0, ",", bz1, "]")
    print("  box volume:", (bx1 - bx0) * (by1 - by0) * (bz1 - bz0), "m^3")

    # ── the sweep ─────────────────────────────────────────────────────────
    var hits = 0
    var closest = 1.0e9
    var ex0 = 1.0e9
    var ey0 = 1.0e9
    var ez0 = 1.0e9
    var ex1 = -1.0e9
    var ey1 = -1.0e9
    var ez1 = -1.0e9

    var gs = g.terms[g.root()].a      # the gripper SITE id AtRegion reads

    for k in range(draws):
        for i in range(len(q_adr)):
            # ⚠ COUNTER-BASED, like `tasks/sampler.mojo` and for the same
            # reason: a draw is a pure function of (k, i), so a rerun of this
            # file reports the same number and a change in it is a change in
            # the MODEL rather than in the stream.
            var rng = PhiloxRandom(
                seed=UInt64(0x5EED),
                subsequence=(UInt64(k) << 8) | UInt64(i),
                offset=0,
            )
            var u = Float64(rng.step_uniform()[0])
            d.qpos.data[q_adr[i]] = Scalar[DT](
                q_lo[i] + u * (q_hi[i] - q_lo[i])
            )
        forward_kinematics["cpu", DT, DynDims, 1](d, m)

        var px = Float64(d.site_xpos.data[gs * 3])
        var py = Float64(d.site_xpos.data[gs * 3 + 1])
        var pz = Float64(d.site_xpos.data[gs * 3 + 2])
        if px < ex0: ex0 = px
        if py < ey0: ey0 = py
        if pz < ez0: ez0 = pz
        if px > ex1: ex1 = px
        if py > ey1: ey1 = py
        if pz > ez1: ez1 = pz

        # distance to the box — zero inside it
        var dx = 0.0
        if px < bx0: dx = bx0 - px
        elif px > bx1: dx = px - bx1
        var dy = 0.0
        if py < by0: dy = by0 - py
        elif py > by1: dy = py - by1
        var dz = 0.0
        if pz < bz0: dz = bz0 - pz
        elif pz > bz1: dz = pz - bz1
        var dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        if dist < closest:
            closest = dist

        var xb = List[Float64]()
        for i in range(nb * 3):
            xb.append(Float64(d.xpos.data[i]))
        var xq = List[Float64]()
        for i in range(nb * 4):
            xq.append(Float64(d.xquat.data[i]))
        var sp = List[Float64]()
        for i in range(ns * 3):
            sp.append(Float64(d.site_xpos.data[i]))
        if eval_goal(g, f, xb, xq, sp, rsites):
            hits += 1

    print()
    print("  draws           :", draws)
    print("  hits            :", hits)
    print("  hit rate        :", Float64(hits) / Float64(draws))
    print("  closest approach:", closest, "m (0 = inside the box)")
    print("  gripper envelope: x [", ex0, ",", ex1, "]  y [", ey0, ",", ey1,
          "]  z [", ez0, ",", ez1, "]")

    print()
    # ⚠⚠ THE TWO CONTROLS. A hit rate is a number and a number is not a
    # finding until the two ways of getting it trivially are ruled out.
    #
    # (a) THE SWEEP MOVED. If FK were reading a stale `qpos` — or if `q_adr`
    # picked words no body depends on — every draw would land on one point,
    # the envelope would be degenerate and the hit rate would be 0 or 1 with
    # no middle. A zero-extent envelope is that failure, and it is the one
    # that reads most like a real answer.
    var span = (ex1 - ex0) + (ey1 - ey0) + (ez1 - ez0)
    if span < 0.05:
        raise Error(
            "reachability: the gripper site moved less than 5 cm in total"
            " over " + String(draws) + " draws (envelope span "
            + String(span) + " m). The sweep is not sweeping — either the"
            " drawn qpos words are not the arm's, or FK is not seeing them."
        )
    print("  ok: the sweep MOVES the gripper — envelope span", span, "m")

    # (b) THE EVALUATOR AGREES WITH THE BOX. `eval_goal` counted the hits and
    # the box computed the distance, independently; a hit with a nonzero
    # closest approach, or zero hits with a zero closest approach, means the
    # two disagree about what the region IS — which is exactly the class of
    # bug `region_rects` vs the device table produces.
    if (hits > 0) != (closest <= 1e-12):
        raise Error(
            "reachability: `eval_goal` counted " + String(hits) + " hits while"
            " the closest approach to the region box is " + String(closest)
            + " m. The evaluator and the printed box disagree about the"
            " region — check `region_rects` against `eval.pred_in_rect`."
        )
    print("  ok: the evaluator and the printed box agree")

    # (c) ⚠⚠ AND THE TASK MUST STAY TRAINABLE, which is a THRESHOLD and not
    # a nonzero test. A target set of 1e-6 of configuration space is
    # satisfiable and untrainable, and "hits > 0" cannot tell the two apart —
    # it would pass a family whose table had drifted to the edge of the
    # workspace and report the flat training curve as an agent problem.
    #
    # ⚠ THE FLOOR IS 0.5% AND THE MEASURED VALUE IS 3.2%. The floor is set
    # from what sparse RL needs rather than from what this model scores: below
    # roughly one uniform pose in two hundred, a 300-step episode is unlikely
    # to contain a success and there is nothing for SAC to bootstrap from. The
    # gap between 0.5 and 3.2 is deliberate slack — a drop to 0.6% passes and
    # is VISIBLE in the printed rate, which is the point of printing it.
    comptime MIN_HIT_RATE: Float64 = 0.005
    var rate = Float64(hits) / Float64(draws)
    print()
    if rate < MIN_HIT_RATE:
        raise Error(
            "reachability: `so101_reach_brick` is met by " + String(rate)
            + " of uniform arm poses, below the " + String(MIN_HIT_RATE)
            + " floor a sparse reward needs (closest approach "
            + String(closest) + " m). The task is not trainable as written:"
            " a policy on it would give a flat curve, which reads as an AGENT"
            " problem and is a TASK one. Check the region's rect and the"
            " fixture's pose against the arm's envelope printed above."
        )
    print("  ok: hit rate", rate, ">= the", MIN_HIT_RATE,
          "floor a sparse reward needs")
    print()
    print("  ⚠ THIS IS AN UPPER BOUND, NOT A PREDICTION. A torque random walk")
    print("  covers configuration space far less evenly than a uniform draw;")
    print("  see the header. Whether `reach` TRAINS is answered by putting a")
    print("  policy on it, not here.")
    print()
    print("=== MEASURED ===")
