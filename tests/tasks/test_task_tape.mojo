"""The device tape agrees with the host evaluator — P3a's gate.

The reward kernel will read a goal out of twelve `meta` floats and evaluate it
with `eval_tape`; the CPU path reads a `BoundGoal` and evaluates it with
`eval_goal`. **They must agree on every state**, because P3's real gate is
per-lane reward matching the CPU leg bit-for-bit, and that is impossible if the
two evaluators disagree before a GPU is involved.

⚠⚠ THE STATES ARE CHOSEN TO MAKE EACH GOAL FLIP. Agreement on states where
every goal is False is worthless: `return False` passes it. This sweeps a grid
that puts each task's goal on BOTH sides of its threshold, and ASSERTS BOTH
OUTCOMES OCCUR before comparing — a run where some goal never fired would mean
the sweep, not the evaluators, is what agreed.

⚠ It also asserts the tape REFUSES what it cannot hold: a fourth term, and a
Tier B op. Both are silent-wrong-answer shapes — a truncated goal evaluates to
something simpler than the author wrote, and a Tier B goal on the device tape
would read contacts the kernel does not carry.

Run: pixi run mojo run -I . tests/tasks/test_task_tape.mojo
"""

from mojo_rl.tasks.spec import load_family, load_task
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.predicates import parse_goal, bind_goal
from mojo_rl.tasks.eval import (
    eval_goal, region_sites, region_rects, region_half_heights,
)
from mojo_rl.tasks.tape import encode_goal, eval_tape, TAPE_WORDS
from mojo_rl.physics3d.parser.runtime_load import parse_model_runtime


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


def _task_names() -> List[String]:
    var o = List[String]()
    o.append(String("so101_reach_brick"))
    o.append(String("so101_lift_brick"))
    o.append(String("so101_gather_bricks"))
    o.append(String("so101_settle_brick"))
    return o^


def main() raises:
    print("=== the device tape vs the host evaluator — P3a ===")
    var ta = Tally()

    var f = load_family("mojo_rl/tasks/families/so101_tabletop.family")
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)
    var nb = len(fmd.body_names)
    var ns = len(fmd.site_names)

    # region table as the flat arrays a kernel would get
    var r_site = List[Int]()
    var r_x0 = List[Float64]()
    var r_y0 = List[Float64]()
    var r_x1 = List[Float64]()
    var r_y1 = List[Float64]()
    # ⚠ THE PER-REGION Z BAND, which used to be `eval.IN_HALF_HEIGHT` inside
    # both evaluators. Built here so the host tape reader gets exactly what
    # `region_table_words` writes for the device.
    var r_h = List[Float64]()
    var rheights = region_half_heights(f)
    for i in range(len(f.regions)):
        r_h.append(rheights[i])
        r_site.append(rsites[i])
        r_x0.append(rects[i][0])
        r_y0.append(rects[i][1])
        r_x1.append(rects[i][2])
        r_y1.append(rects[i][3])

    var names = _task_names()
    var total = 0
    var agreed = 0
    var trues = 0

    for n in range(len(names)):
        var t = load_task("mojo_rl/tasks/tasks/" + names[n] + ".task")
        var g = bind_goal(parse_goal(t.goal), f, fmd.body_names, fmd.site_names)
        var tape = encode_goal(g)
        ta.check(len(tape) == TAPE_WORDS,
                 String(names[n]) + " encodes to " + String(TAPE_WORDS)
                 + " words")

        var t_true = 0
        var t_n = 0
        # ⚠ THE GRID STRADDLES EVERY THRESHOLD: the table site is at
        # (0.25, 0, 0.02) with a +-0.10 rect, `On`'s band is [-0.005, 0.08]
        # and `lift`'s margin is 0.06 above the table body at z 0.01. These
        # x/z values put props inside and outside both.
        for ix in range(5):
            for iz in range(5):
                var xp = List[Float64]()
                for _ in range(nb * 3):
                    xp.append(0.0)
                var xq = List[Float64]()
                for _ in range(nb):
                    xq.append(0.0)
                    xq.append(0.0)
                    xq.append(0.0)
                    xq.append(1.0)
                var sp = List[Float64]()
                for _ in range(ns * 3):
                    sp.append(0.0)

                var ts = rsites[0]
                sp[ts * 3] = 0.25
                sp[ts * 3 + 1] = 0.0
                sp[ts * 3 + 2] = 0.02
                var px = 0.10 + Float64(ix) * 0.09
                var pz = 0.00 + Float64(iz) * 0.045
                for b in range(nb):
                    xp[b * 3] = px
                    xp[b * 3 + 1] = 0.0
                    xp[b * 3 + 2] = pz
                # the table body sits where the family composes it
                for b in range(nb):
                    if String(fmd.body_names[b]).startswith("table_"):
                        xp[b * 3] = 0.25
                        xp[b * 3 + 1] = 0.0
                        xp[b * 3 + 2] = 0.01
                # ⚠⚠ `cube_a` IS PINNED, AND A RELATIVE PREDICATE IS WHY.
                # The loop above puts EVERY body at the swept point, which is
                # fine for the goals that compare a body to a REGION or to the
                # table — and makes `Near(brick, cube_a, 0.06)` a constant
                # TRUE, distance zero, in all 25 states. `gather` then
                # contributed 25 agreements and tested nothing, which the
                # flip check below caught the moment the task stopped being a
                # region predicate. Pinning the second prop makes the
                # separation vary with the sweep exactly as the other goals'
                # quantities do.
                for b in range(nb):
                    if String(fmd.body_names[b]).startswith("cube_a_"):
                        xp[b * 3] = 0.25
                        xp[b * 3 + 1] = 0.0
                        xp[b * 3 + 2] = 0.04
                # the gripper site follows the sweep too, for AtRegion
                for si in range(ns):
                    if String(fmd.site_names[si]) == "robot_gripperframe":
                        sp[si * 3] = px
                        sp[si * 3 + 1] = 0.0
                        sp[si * 3 + 2] = pz

                var host = eval_goal(g, f, xp, xq, sp, rsites)
                var dev = eval_tape(
                    tape, 0, xp, xq, sp, r_site, r_x0, r_y0, r_x1, r_y1, r_h
                )
                total += 1
                t_n += 1
                if host == dev:
                    agreed += 1
                if host:
                    trues += 1
                    t_true += 1
        print("    ", names[n], ":", t_true, "of", t_n, "states satisfy it")
        # ⚠ EVERY TASK MUST FLIP SOMEWHERE IN THE SWEEP. A goal that is False
        # in all 25 states contributes 25 agreements and tests nothing.
        ta.check(
            t_true > 0 and t_true < t_n,
            String(names[n]) + " flips inside the sweep (not all-True/all-False)",
        )

    print("  ", agreed, "of", total, "states agree;", trues, "were True")
    ta.check(agreed == total, "host and device tape agree on EVERY state")

    # ── refusals ──────────────────────────────────────────────────────────
    print("--- the tape refuses what it cannot hold ---")
    var deep = String("And(On(brick, table_top), And(On(cube_a, table_top),"
                      " On(cube_b, table_top)))")
    var raised = False
    try:
        _ = encode_goal(
            bind_goal(parse_goal(deep), f, fmd.body_names, fmd.site_names)
        )
    except e:
        raised = True
    ta.check(raised, "a FOURTH term is refused, not truncated")

    raised = False
    try:
        _ = encode_goal(
            bind_goal(parse_goal(String("Grasped(brick)")), f,
                      fmd.body_names, fmd.site_names)
        )
    except e:
        raised = True
    ta.check(raised, "a TIER B op is refused (the kernel has no contacts)")

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "tape: " + String(ta.failures) + " of " + String(ta.checks)
            + " check(s) failed"
        )
    print("=== PASS ===")
