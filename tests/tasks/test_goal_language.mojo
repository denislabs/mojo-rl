"""The goal language — parse, bind, and the Tier A rule. P2a's gate.

⚠ THE BINDING HALF RUNS AGAINST THE REAL COMPOSED FAMILY, not a fixture. A
goal's whole job is to name things in a scene, so a gate that binds against
invented names would pass while every real task failed to resolve. This binds
`In(brick, table_top)` against `mojo_rl/tasks/scenes/so101_tabletop.xml` and
checks it lands on the body `<attach prefix="brick_">` actually produced.

Run: pixi run mojo run -I . tests/tasks/test_goal_language.mojo
"""

from mojo_rl.tasks.spec import load_family
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.predicates import (
    parse_goal, bind_goal, require_tier_a, slot_body_id,
    op_is_composite, op_name,
    OP_IN, OP_AND, OP_NEAR, OP_GRASPED, MAX_GOAL_TERMS,
)
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

    def refuses(mut self, text: String, what: String):
        self.checks += 1
        var raised = False
        try:
            _ = parse_goal(text)
        except e:
            raised = True
        if raised:
            print("  ok: refused —", what)
        else:
            self.failures += 1
            print("  FAIL: ACCEPTED —", what)


def main() raises:
    print("=== the goal language — P2a ===")
    var ta = Tally()

    # ── 1. parse + round-trip ─────────────────────────────────────────────
    print("--- parse ---")
    var g1 = parse_goal(String("In(brick, table_top)"))
    ta.check(len(g1.terms) == 1, "a leaf goal is one term")
    ta.check(g1.terms[0].op == OP_IN, "In parsed")
    ta.check(g1.describe() == "In(brick, table_top)", "round-trips as written")

    var g2 = parse_goal(
        String("And(In(brick, table_top), Near(cube_a, cube_b, 0.05))")
    )
    ta.check(len(g2.terms) == 3, "And + two leaves is three terms")
    ta.check(g2.terms[g2.root()].op == OP_AND, "the ROOT is the composite")
    ta.check(
        g2.describe() == "And(In(brick, table_top), Near(cube_a, cube_b, 0.05))",
        "a nested goal round-trips",
    )
    ta.check(g2.terms[1].param == 0.05, "Near's distance survives")

    # ⚠⚠ THE POST-ORDER INVARIANT IS WHAT MAKES STACKLESS EVALUATION LEGAL.
    # A GPU kernel sweeps this array FORWARDS and reads each child's already
    # -computed result; if a parent could reference a later term, that sweep
    # would read an uninitialised slot and the goal would evaluate to garbage
    # ONLY on device. Asserted here, where it is cheap.
    var ordered = True
    for i in range(len(g2.terms)):
        if op_is_composite(g2.terms[i].op):
            if g2.terms[i].kid0 >= i:
                ordered = False
            if g2.terms[i].kid1 >= i:
                ordered = False
    ta.check(ordered, "every child index is LESS than its parent's")

    var g3 = parse_goal(String("Not(Above(brick, cube_a))"))
    ta.check(g3.describe() == "Not(Above(brick, cube_a))", "Not is unary")

    # ── 2. refusals ───────────────────────────────────────────────────────
    print("--- refusals ---")
    ta.refuses(String("Floats(brick, table_top)"), "an unknown predicate")
    ta.refuses(String("In(brick"), "an unclosed paren")
    ta.refuses(String("In(brick, table_top) On(cube_a, table_top)"),
               "TWO goals with no operator between them")
    ta.refuses(String("In()"), "no arguments")
    ta.refuses(String("Near(a, b)"), "Near with a missing distance")
    ta.refuses(String(""), "an empty goal")
    # ⚠ THE TERM CAP IS A DEVICE-SIDE ONE — P3's tape is comptime-sized, so a
    # goal past it must be refused rather than truncated.
    var deep = String("In(brick, table_top)")
    for _ in range(MAX_GOAL_TERMS):
        deep = String("And(") + deep + ", In(brick, table_top))"
    ta.refuses(deep, "a goal past MAX_GOAL_TERMS")

    # ── 3. binding against the REAL composed family ───────────────────────
    print("--- bind against the composed scene ---")
    var f = load_family("mojo_rl/tasks/families/so101_tabletop.family")
    var fmd = parse_model_runtime(scene_path(f))
    print("    scene has", len(fmd.body_names), "body names,",
          len(fmd.site_names), "sites")

    var brick_id = slot_body_id(String("brick"), fmd.body_names)
    print("    slot 'brick' -> body", brick_id, "=",
          fmd.body_names[brick_id])
    ta.check(
        String(fmd.body_names[brick_id]).startswith("brick_"),
        "a slot resolves to the body its PREFIX produced",
    )
    # ⚠ body_names[0] is the WORLDBODY. A binder that forgot that would
    # resolve one body early and still return an id — the silent kind.
    ta.check(fmd.body_names[0] == "world", "body 0 is the worldbody")

    var b1 = bind_goal(g1, f, fmd.body_names)
    ta.check(len(b1.terms) == 1, "bound goal has the same shape")
    ta.check(b1.terms[0].a == brick_id, "In's first arg is the brick's body")
    ta.check(b1.terms[0].b == 0, "In's second arg is the REGION index")
    ta.check(b1.is_tier_a(), "In is Tier A")
    require_tier_a(b1, String("t"))
    print("  ok: require_tier_a accepts it")
    ta.checks += 1

    var b2 = bind_goal(g2, f, fmd.body_names)
    ta.check(
        b2.terms[b2.root()].a == 0 and b2.terms[b2.root()].b == 1,
        "And's args are TERM indices, not body ids",
    )

    # ── 4. Tier B binds, and is refused where it matters ──────────────────
    print("--- Tier B ---")
    var gb = parse_goal(String("Grasped(brick)"))
    var bb = bind_goal(gb, f, fmd.body_names)
    ta.check(not bb.is_tier_a(), "Grasped is NOT Tier A")
    var raised = False
    try:
        require_tier_a(bb, String("pick"))
    except e:
        raised = True
    ta.check(raised, "require_tier_a REFUSES a Tier B goal (§5.1)")

    # ── 5. binding refuses what does not exist ────────────────────────────
    print("--- binding refusals ---")
    var bad = 0
    try:
        _ = bind_goal(parse_goal(String("In(hammer, table_top)")), f,
                      fmd.body_names)
    except e:
        bad += 1
    try:
        _ = bind_goal(parse_goal(String("In(brick, nowhere)")), f,
                      fmd.body_names)
    except e:
        bad += 1
    ta.check(bad == 2,
             "a goal naming an unknown SLOT or REGION is refused at bind")

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "goal language: " + String(ta.failures) + " of "
            + String(ta.checks) + " check(s) failed"
        )
    print("=== PASS ===")
