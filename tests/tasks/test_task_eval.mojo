"""Three tasks, one family, no `.mojo` between them — P2c's gate.

## WHAT THIS ASSERTS

1. **the data-only claim** — three `.task` files load, validate against the
   family, and bind, with nothing recompiled between them. That sentence is
   the whole point of the task layer and this is where it is checked;
2. **the predicate SEMANTICS**, on geometry this file controls. Each goal is
   evaluated in a state where it MUST hold and a state where it MUST NOT.

⚠⚠ THE NEGATIVE STATE IS NOT OPTIONAL. `eval_goal` returning True is half a
result: a predicate that returns True unconditionally satisfies every positive
assertion, and `In` degenerating to "anything near the table" is exactly the
kind of drift that would train a policy on a task nobody wrote.

⚠ GEOMETRY IS CONSTRUCTED, NOT SIMULATED. The point here is what a predicate
MEANS, and stepping physics to reach a state would test the solver instead —
and could not reach the negative states at all. Whether a POLICY can achieve
these goals is a different question, and needs a controller this layer does
not have; see the note at the end of this file.

Run: pixi run mojo run -I . tests/tasks/test_task_eval.mojo
"""

from mojo_rl.tasks.spec import (
    load_family, load_task, validate_task_against_family,
)
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.predicates import bind_goal, parse_goal, require_tier_a
from mojo_rl.tasks.eval import eval_goal, region_sites
from mojo_rl.physics3d.parser.runtime_load import parse_model_runtime


comptime TASKS = "mojo_rl/tasks/tasks/"


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


def _task_files() -> List[String]:
    var out = List[String]()
    out.append(String("so101_reach_brick"))
    out.append(String("so101_lift_brick"))
    out.append(String("so101_gather_bricks"))
    return out^


def fresh_state(nb: Int, ns: Int) -> List[List[Float64]]:
    """`[xpos, xquat, site_xpos]`, everything at the origin and unrotated.

    ⚠ A TOP-LEVEL FUNCTION, not a nested one: a nested `def` in Mojo cannot
    infer a capture convention for `nb`/`ns` and the error names the loop
    rather than the closure.
    """
    var xp = List[Float64]()
    for _ in range(nb * 3):
        xp.append(0.0)
    var xq = List[Float64]()
    for _ in range(nb):
        xq.append(1.0)
        xq.append(0.0)
        xq.append(0.0)
        xq.append(0.0)
    var sp = List[Float64]()
    for _ in range(ns * 3):
        sp.append(0.0)
    var out = List[List[Float64]]()
    out.append(xp^)
    out.append(xq^)
    out.append(sp^)
    return out^


def main() raises:
    print("=== three tasks, one family — P2c ===")
    var ta = Tally()

    var f = load_family("mojo_rl/tasks/families/so101_tabletop.family")
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    print("  family:", f.name, "| regions:", len(f.regions),
          "| region 0 -> site", rsites[0], "=", fmd.site_names[rsites[0]])

    # ── 1. the data-only claim ────────────────────────────────────────────
    print("--- every task loads, validates and binds ---")
    var names = _task_files()
    for i in range(len(names)):
        var t = load_task(TASKS + names[i] + ".task")
        validate_task_against_family(t, f)
        var g = parse_goal(t.goal)
        var b = bind_goal(g, f, fmd.body_names, fmd.site_names)
        require_tier_a(b, t.name)
        print("    ", t.name, "->", g.describe(), "| active:", len(t.active))
        ta.check(len(b.terms) > 0, String(names[i]) + " binds")
    ta.check(
        len(names) == 3,
        "THREE tasks, one family, no .mojo touched between them",
    )

    # ── 2. semantics, on constructed geometry ─────────────────────────────
    print("--- predicate semantics (positive AND negative) ---")
    var nb = len(fmd.body_names)
    var ns = len(fmd.site_names)

    var brick = 0
    var cube_a = 0
    var table = 0
    for i in range(nb):
        if String(fmd.body_names[i]).startswith("brick_"):
            brick = i
        if String(fmd.body_names[i]).startswith("cube_a_"):
            cube_a = i
        if String(fmd.body_names[i]).startswith("table_"):
            table = i
    var grip = 0
    for i in range(ns):
        if String(fmd.site_names[i]) == "robot_gripperframe":
            grip = i
    var tsite = rsites[0]

    # The table's site sits at (0.25, 0.0, 0.31); its region rect is
    # -0.10..0.10 in x and y, from the .family.
    var st = fresh_state(nb, ns)
    st[2][tsite * 3] = 0.25
    st[2][tsite * 3 + 1] = 0.0
    st[2][tsite * 3 + 2] = 0.31
    st[0][table * 3 + 2] = 0.30

    var g_on = bind_goal(
        parse_goal(String("And(On(brick, table_top), On(cube_a, table_top))")),
        f, fmd.body_names, fmd.site_names,
    )

    # both props resting on the surface, inside the rect
    st[0][brick * 3] = 0.25
    st[0][brick * 3 + 1] = 0.04
    st[0][brick * 3 + 2] = 0.33
    st[0][cube_a * 3] = 0.21
    st[0][cube_a * 3 + 1] = -0.03
    st[0][cube_a * 3 + 2] = 0.33
    ta.check(
        eval_goal(g_on, f, st[0], st[1], st[2], rsites),
        "gather: BOTH props on the table -> True",
    )

    # ⚠ THE NEGATIVE. One prop slid outside the rect in x: the And must fail.
    st[0][cube_a * 3] = 0.60
    ta.check(
        not eval_goal(g_on, f, st[0], st[1], st[2], rsites),
        "gather: one prop OUTSIDE the region -> False",
    )
    st[0][cube_a * 3] = 0.21

    # ⚠ AND THE OTHER NEGATIVE — inside the rect in XY but a metre up. Without
    # this, `On` could be pure XY containment and still pass everything above.
    st[0][brick * 3 + 2] = 1.30
    ta.check(
        not eval_goal(g_on, f, st[0], st[1], st[2], rsites),
        "gather: a prop in XY but a metre ABOVE the surface -> False",
    )

    # lift: Above(brick, table)
    var g_lift = bind_goal(
        parse_goal(String("Above(brick, table)")),
        f, fmd.body_names, fmd.site_names,
    )
    ta.check(
        eval_goal(g_lift, f, st[0], st[1], st[2], rsites),
        "lift: the brick a metre up IS above the table",
    )
    st[0][brick * 3 + 2] = 0.29
    ta.check(
        not eval_goal(g_lift, f, st[0], st[1], st[2], rsites),
        "lift: the brick BELOW the table is not",
    )

    # reach: AtRegion(robot_gripperframe, table_top) — a SITE subject
    var g_reach = bind_goal(
        parse_goal(String("AtRegion(robot_gripperframe, table_top)")),
        f, fmd.body_names, fmd.site_names,
    )
    st[2][grip * 3] = 0.26
    st[2][grip * 3 + 1] = 0.01
    st[2][grip * 3 + 2] = 0.36
    ta.check(
        eval_goal(g_reach, f, st[0], st[1], st[2], rsites),
        "reach: the gripper SITE over the zone -> True",
    )
    st[2][grip * 3] = 0.90
    ta.check(
        not eval_goal(g_reach, f, st[0], st[1], st[2], rsites),
        "reach: the gripper parked away -> False",
    )

    # Upright, and its negative
    var g_up = bind_goal(
        parse_goal(String("Upright(brick, 0.05)")),
        f, fmd.body_names, fmd.site_names,
    )
    ta.check(
        eval_goal(g_up, f, st[0], st[1], st[2], rsites),
        "upright: an identity quaternion IS upright",
    )
    # 90 degrees about x -> local +z points along world -y
    st[1][brick * 4] = 0.7071067811865476
    st[1][brick * 4 + 1] = 0.7071067811865476
    st[1][brick * 4 + 2] = 0.0
    st[1][brick * 4 + 3] = 0.0
    ta.check(
        not eval_goal(g_up, f, st[0], st[1], st[2], rsites),
        "upright: tipped 90 degrees is NOT",
    )

    # ── 3. Tier B is refused, not silently False ──────────────────────────
    print("--- Tier B ---")
    var g_b = bind_goal(
        parse_goal(String("Grasped(brick)")),
        f, fmd.body_names, fmd.site_names,
    )
    var raised = False
    try:
        _ = eval_goal(g_b, f, st[0], st[1], st[2], rsites)
    except e:
        raised = True
    ta.check(
        raised,
        "eval_goal REFUSES Tier B rather than returning a silent False",
    )

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "task eval: " + String(ta.failures) + " of " + String(ta.checks)
            + " check(s) failed"
        )
    print("=== PASS ===")
    print()
    print("⚠ THIS GATES WHAT A GOAL MEANS, NOT WHETHER A POLICY REACHES IT.")
    print("  §9-P2's 'each running to a success' needs a controller or a")
    print("  trained policy; neither is part of the task layer. See the")
    print("  note in docs/TASK_LAYER_IMPLEMENTATION.md.")
