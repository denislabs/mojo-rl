"""`.family` / `.task` parsing — P1b's gate.

Three halves, and the second and third are the ones that matter:

1. **round-trip is a FIXED POINT** — parse -> encode -> parse -> encode gives
   the identical text. The studio must WRITE these files, so a reader that
   cannot reproduce what it read would silently rewrite an author's file.
2. **every REFUSAL actually refuses** — each `raise` in `spec.mojo` has an arm
   here. ⚠ Without these the file would pass its round-trip and accept a
   task naming an object that does not exist; a parser gate that only tests
   the happy path is the vacuity default with extra steps.
3. **the cross-file rule holds** — `TASK_LAYER_PLAN.md` §4.4, "a task cannot
   introduce an object the family did not declare".

Run: pixi run mojo run -I . tests/tasks/test_spec_roundtrip.mojo
"""

from mojo_rl.tasks.spec import (
    FamilySpec, TaskSpec,
    parse_family, parse_task, validate_task_against_family,
    SLOT_FREE, SLOT_STATIC,
)


comptime FAMILY = String(
    "schema_version=1\n"
    "family=so101_tabletop\n"
    "base=scenes/so101_tabletop.xml\n"
    "horizon=300\n"
    "control_freq=20\n"
    "\n"
    "# the movable props\n"
    "slot=brick:free:props/brick.xml\n"
    "slot=cube_a:free:props/cube.xml\n"
    "slot=box:free:props/box_small.xml\n"
    "slot=table:static:props/table.xml:0.25,0.0,0.30\n"
    "\n"
    "park=10.0,0.0,50.0\n"
    "\n"
    "region=table_top:site:table_surface:-0.1,-0.15,0.1,0.15\n"
    "region=box_inside:site:box_inner\n"
)

comptime TASK = String(
    "schema_version=1\n"
    "task=so101_pick_brick\n"
    "family=so101_tabletop\n"
    "language=Pick up the brick and put it in the box\n"
    "goal=In(brick, box_inside)\n"
    "active=brick\n"
    "active=box\n"
    "active=table\n"
    "init=brick@table_top\n"
    "init=box@table_top\n"
)


struct Tally(Copyable, ImplicitlyCopyable, Movable):
    """Checks run and checks failed. ⚠ BOTH are printed at the end: a run that
    reports "0 failed" over 0 checks is indistinguishable from a pass."""

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

    def refuses(
        mut self, text: String, what: String, is_family: Bool = False
    ):
        """The parse MUST raise. A refusal that never fires is a dead gate."""
        self.checks += 1
        var raised = False
        try:
            if is_family:
                _ = parse_family(text)
            else:
                _ = parse_task(text)
        except e:
            raised = True
        if raised:
            print("  ok: refused —", what)
        else:
            self.failures += 1
            print("  FAIL: ACCEPTED what it must refuse —", what)

    def refuses_pair(mut self, t: String, f: String, what: String):
        self.checks += 1
        var raised = False
        try:
            var fam = parse_family(f)
            var task = parse_task(t)
            validate_task_against_family(task, fam)
        except e:
            raised = True
        if raised:
            print("  ok: refused —", what)
        else:
            self.failures += 1
            print("  FAIL: ACCEPTED what it must refuse —", what)


def main() raises:
    print("=== .family / .task spec — P1b ===")
    var ta = Tally()

    # ── 1. round-trip is a fixed point ────────────────────────────────────
    print("--- round-trip ---")
    var f1 = parse_family(FAMILY)
    var f_txt = f1.encode()
    var f2 = parse_family(f_txt)
    ta.check(f2.encode() == f_txt, "family parse->encode is a fixed point")
    ta.check(f1.name == "so101_tabletop", "family name survives")
    ta.check(len(f1.slots) == 4, "four slots parsed")
    ta.check(f1.horizon == 300, "horizon survives")
    ta.check(f1.park_z == 50.0, "park z survives")

    # ⚠ THE COST MODEL. Three of the four slots are free; the table is static
    # and must NOT be counted, because it adds no dofs and pays none of the
    # quadratic the budget measurement priced.
    ta.check(f1.n_free_slots() == 3, "n_free_slots counts free only (3 of 4)")
    ta.check(
        f1.slots[3].kind == SLOT_STATIC, "the table parsed as STATIC"
    )
    # ⚠ A STATIC SLOT CARRIES ITS OWN POSE, because it has no joint and so
    # cannot be moved after composition. Parking one welds it wherever the
    # park pose is — 50 m up, in the first version of this family, with the
    # region on its surface and the sampler placing props into the sky.
    ta.check(f1.slots[3].has_pose and f1.slots[3].pz == 0.30,
             "the static slot keeps its composed pose")
    ta.check(f1.regions[0].has_rect, "region with a rectangle keeps it")
    ta.check(not f1.regions[1].has_rect, "region without one is the site extent")
    ta.check(f1.regions[0].x_min == -0.1, "a NEGATIVE rect bound survives")

    var t1 = parse_task(TASK)
    var t_txt = t1.encode()
    var t2 = parse_task(t_txt)
    ta.check(t2.encode() == t_txt, "task parse->encode is a fixed point")
    ta.check(t1.goal == "In(brick, box_inside)", "goal carried as text")
    ta.check(
        t1.language == "Pick up the brick and put it in the box",
        "language survives, spaces and all",
    )
    ta.check(len(t1.active) == 3, "three active slots")

    # ── 2. every refusal fires ────────────────────────────────────────────
    print("--- refusals (each is a `raise` in spec.mojo) ---")
    ta.refuses(
        String("schema_version=1\ntask=t\nfamily=f\ngoal=In(a,b)\nlanguge=x\n"),
        "a TYPO'D KEY ('languge') — the whole reason unknown keys raise",
    )
    ta.refuses(
        String("task=t\nfamily=f\ngoal=In(a,b)\n"),
        "no schema_version",
    )
    ta.refuses(
        String("schema_version=99\ntask=t\nfamily=f\ngoal=In(a,b)\n"),
        "a schema_version newer than this build",
    )
    ta.refuses(
        String("schema_version=1\ntask=t\nfamily=f\n"),
        "NO GOAL — a task that would always succeed",
    )
    ta.refuses(
        String("schema_version=1\ntask=t\nfamily=f\ngoal=g\nactive=a\nactive=a\n"),
        "the same slot listed active twice",
    )
    ta.refuses(
        String("schema_version=1\ntask=t\nfamily=f\ngoal=g\ninit=brick\n"),
        "an init with no '@region'",
    )
    ta.refuses(
        String("schema_version=1\ntask=t\nfamily=f\ngoal=g\nthis line has no equals\n"),
        "a line with no '='",
    )
    ta.refuses(
        String("schema_version=1\nfamily=f\nbase=b\nhorizon=1\n"
               "slot=brick:floaty:a.xml\n"),
        "an unknown slot KIND ('floaty')", is_family=True,
    )
    ta.refuses(
        String("schema_version=1\nfamily=f\nbase=b\nhorizon=1\n"
               "slot=a:free:x.xml\nslot=a:free:y.xml\n"),
        "a DUPLICATE slot name — two objects under one key", is_family=True,
    )
    ta.refuses(
        String("schema_version=1\nfamily=f\nbase=b\nhorizon=1\n"
               "slot=t:static:x.xml\n"),
        "a STATIC slot with NO POSE — it would be welded at the park pose",
        is_family=True,
    )
    ta.refuses(
        String("schema_version=1\nfamily=f\nbase=b\nhorizon=1\n"
               "slot=b:free:x.xml:0.1,0.2,0.3\n"),
        "a FREE slot WITH a pose — it would be silently ignored",
        is_family=True,
    )
    ta.refuses(
        String("schema_version=1\nfamily=f\nbase=b\nhorizon=1\n"
               "region=r:body:torso\n"),
        "a region targeting a BODY (only sites travel)", is_family=True,
    )
    ta.refuses(
        String("schema_version=1\nfamily=f\nbase=b\nhorizon=1\n"
               "region=r:site:s:0.1,0.1,-0.1,-0.1\n"),
        "a REVERSED rectangle — the sampler would just never accept a draw",
        is_family=True,
    )
    ta.refuses(
        String("schema_version=1\nfamily=f\nbase=b\nhorizon=0\n"),
        "horizon=0", is_family=True,
    )
    ta.refuses(
        String("schema_version=1\nfamily=f\nbase=b\nhorizon=1\npark=1.0,2.0\n"),
        "a park pose with two numbers instead of three", is_family=True,
    )

    # ── 3. the cross-file rule — PLAN §4.4 ────────────────────────────────
    print("--- task vs family (§4.4: a task cannot add an object) ---")
    validate_task_against_family(t1, f1)
    print("  ok: the good pair validates")
    ta.checks += 1

    ta.refuses_pair(
        String("schema_version=1\ntask=t\nfamily=so101_tabletop\ngoal=g\n"
               "active=hammer\n"),
        FAMILY,
        "an active slot the family NEVER DECLARED ('hammer')",
    )
    ta.refuses_pair(
        String("schema_version=1\ntask=t\nfamily=so101_tabletop\ngoal=g\n"
               "active=brick\ninit=brick@nowhere\n"),
        FAMILY,
        "an init into an undeclared region",
    )
    ta.refuses_pair(
        String("schema_version=1\ntask=t\nfamily=so101_tabletop\ngoal=g\n"
               "active=brick\ninit=brick@table_top\ninit=cube_a@table_top\n"),
        FAMILY,
        "an init for a slot that is NOT active — it would be silently parked",
    )
    ta.refuses_pair(
        String("schema_version=1\ntask=t\nfamily=so101_tabletop\ngoal=g\n"
               "active=brick\n"),
        FAMILY,
        "a FREE slot active with no init — identical placement every episode",
    )
    ta.refuses_pair(
        String("schema_version=1\ntask=t\nfamily=other\ngoal=g\n"),
        FAMILY,
        "a task validated against the WRONG family",
    )

    # ⚠ THE POSITIVE CONTROL FOR THE RULE ABOVE. A STATIC slot active with no
    # init must be ACCEPTED — a fixture has its pose from the scene. Without
    # this, tightening the free-slot rule into "every active slot needs an
    # init" would pass every refusal arm above and break every fixture.
    var t_static = parse_task(
        String("schema_version=1\ntask=t\nfamily=so101_tabletop\ngoal=g\n"
               "active=table\n")
    )
    validate_task_against_family(t_static, f1)
    print("  ok: a STATIC slot with no init is ACCEPTED (the control)")
    ta.checks += 1

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "spec gate: " + String(ta.failures) + " of "
            + String(ta.checks) + " check(s) failed"
        )
    print("=== PASS ===")
