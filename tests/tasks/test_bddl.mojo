"""LIBERO'S `.bddl`, READ — P5's gate.

⚠⚠ THE FIXTURE BELOW IS ONE UPSTREAM FILE, VERBATIM. It is
`libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl` from LIBERO
(MIT, Lifelong-Robot-Learning/LIBERO), copied byte for byte and NOT
paraphrased. `references/` is gitignored, so a gate that read the corpus from
there would test nothing on a fresh clone; and a fixture I WROTE would gate
the parser against my own idea of the format, which is the shape
`feedback_a_gate_that_shares_its_reference_implementation_is_blind` names.

`examples/tasks/libero_survey.mojo` is the other half: all 130 real files when
the tree is present, skipping loudly when it is not.

## WHY THIS PARTICULAR FILE

It carries every construct that broke the reader while it was being written:

* `moka_pot_1 moka_pot_2 - moka_pot` — a PDDL **typed list**, several names
  per category. 40 of the 130 files use it and the first reader failed all 40.
* `cook_region` with a `(:target flat_stove_1)` and **no `:ranges`** — a
  region whose rectangle lives in the fixture's asset, not in the `.bddl`.
* a **three-term goal**, so unwrapping the outer `And` is exercised on more
  than one child.
* `(Turnon flat_stove_1)` in BOTH `:init` and `:goal` — the articulation
  predicate our language has no equivalent for.
* `:yaw_rotation` blocks, which are parsed and then deliberately unused.

Run: pixi run mojo run -I . tests/tasks/test_bddl.mojo
"""

from mojo_rl.tasks.bddl import parse_bddl, tokenize_bddl
from mojo_rl.tasks.libero_import import (
    classify_goal, translate_family, translate_task, family_todo_count,
    GAP_NONE, GAP_ARTICULATION, GAP_FIXTURE_REGION, GAP_OBJECT_TARGET,
)
from mojo_rl.tasks.spec import SLOT_FREE, SLOT_STATIC


comptime FIXTURE = """(define (problem LIBERO_Kitchen_Tabletop_Manipulation)
  (:domain robosuite)
  (:language put both moka pots on the stove)
    (:regions
      (flat_stove_init_region
          (:target kitchen_table)
          (:ranges (
              (-0.21000000000000002 -0.21000000000000002 -0.19 -0.19)
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (moka_pot_right_init_region
          (:target kitchen_table)
          (:ranges (
              (-0.07500000000000001 0.225 -0.025 0.275)
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (moka_pot_left_init_region
          (:target kitchen_table)
          (:ranges (
              (0.025 0.025 0.07500000000000001 0.07500000000000001)
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )
      (cook_region
          (:target flat_stove_1)
      )
    )

  (:fixtures
    kitchen_table - kitchen_table
    flat_stove_1 - flat_stove
  )

  (:objects
    moka_pot_1 moka_pot_2 - moka_pot
  )

  (:obj_of_interest
    moka_pot_1
    moka_pot_2
    flat_stove_1
  )

  (:init
    (On flat_stove_1 kitchen_table_flat_stove_init_region)
    (On moka_pot_1 kitchen_table_moka_pot_right_init_region)
    (On moka_pot_2 kitchen_table_moka_pot_left_init_region)
    (Turnon flat_stove_1)
  )

  (:goal
    (And (On moka_pot_1 flat_stove_1_cook_region) (On moka_pot_2 flat_stove_1_cook_region) (Turnon flat_stove_1))
  )

)
"""


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


def main() raises:
    print("=== LIBERO .bddl, read ===")
    var ta = Tally()

    var toks = tokenize_bddl(String(FIXTURE))
    print("  tokenised into", len(toks), "tokens")
    # ⚠ AN EXACT COUNT ON A VERBATIM FIXTURE. A range would pass on a
    # tokeniser that dropped every parenthesis; this moves the moment the
    # token stream does. (I first wrote `> 200` from a guess and it failed at
    # 176 — the guess was the defect, which is the argument for measuring.)
    ta.check(len(toks) == 176, "the token stream is exactly 176 tokens")

    var p = parse_bddl(String(FIXTURE))
    print("  problem :", p.problem)
    print("  language:", p.language)
    ta.check(p.problem == "LIBERO_Kitchen_Tabletop_Manipulation",
             "the problem name")
    ta.check(p.domain == "robosuite", "the domain")
    # ⚠ THE FREE TEXT IS JOINED BACK FROM TOKENS, so a lost or doubled space
    # is exactly what this catches.
    ta.check(p.language == "put both moka pots on the stove",
             "the language, word for word")

    # ── the typed list ────────────────────────────────────────────────────
    print("  fixtures:", len(p.fixtures), " objects:", len(p.objects))
    ta.check(len(p.fixtures) == 2, "two fixtures")
    ta.check(len(p.objects) == 2,
             "TWO objects from ONE typed list `moka_pot_1 moka_pot_2 -"
             " moka_pot`")
    var both = (
        len(p.objects) == 2
        and p.objects[0].name == "moka_pot_1"
        and p.objects[1].name == "moka_pot_2"
        and p.objects[0].category == "moka_pot"
        and p.objects[1].category == "moka_pot"
    )
    ta.check(both, "and both carry the SAME category, in order")

    # ── regions, ranged and not ───────────────────────────────────────────
    var ranged = 0
    var unranged = 0
    for i in range(len(p.regions)):
        if p.regions[i].has_ranges:
            ranged += 1
        else:
            unranged += 1
    print("  regions :", len(p.regions), "(", ranged, "ranged,", unranged,
          "unranged )")
    ta.check(ranged == 3 and unranged == 1,
             "three ranged regions and one without :ranges")
    # ⚠ THE COMPOSED NAME IS THE KEY. `cook_region` is declared bare and the
    # goal names it `flat_stove_1_cook_region`.
    var ci = p.region_index(String("flat_stove_1_cook_region"))
    ta.check(ci >= 0, "the unranged region resolves by its COMPOSED name")
    ta.check(
        p.region_index(String("cook_region")) < 0,
        "and NOT by its bare name — the bare name is not unique in the corpus",
    )
    var ki = p.region_index(String("kitchen_table_moka_pot_left_init_region"))
    ta.check(ki >= 0, "a ranged region resolves too")
    if ki >= 0:
        ref r = p.regions[ki]
        ta.check(
            r.x0 == 0.025 and r.y0 == 0.025
            and r.x1 == 0.07500000000000001
            and r.y1 == 0.07500000000000001,
            "its four range values are EXACT, not rounded",
        )
        ta.check(r.has_yaw and r.yaw_lo == 0.0 and r.yaw_hi == 0.0,
                 "and its :yaw_rotation was read")

    # ── init and goal ─────────────────────────────────────────────────────
    print("  init    :", len(p.init), " goal terms:", len(p.goal))
    ta.check(len(p.init) == 4, "four :init atoms")
    # ⚠ THE OUTER `And` IS UNWRAPPED, and this file has THREE children so a
    # reader that returned only the first would score 1 here.
    ta.check(len(p.goal) == 3, "THREE goal terms, with the And unwrapped")
    var g_ok = (
        len(p.goal) == 3
        and p.goal[0].pred == "On" and p.goal[0].args[0] == "moka_pot_1"
        and p.goal[1].pred == "On" and p.goal[1].args[0] == "moka_pot_2"
        and p.goal[2].pred == "Turnon"
    )
    ta.check(g_ok, "in order: On, On, Turnon")
    ta.check(len(p.interest) == 3, "three :obj_of_interest entries")

    # ── the translation, and its refusal ──────────────────────────────────
    print()
    print("--- translation ---")
    var f = translate_family(p)
    print("  family  :", f.name, "|", len(f.slots), "slots,",
          len(f.regions), "regions,", family_todo_count(f), "TODO assets")
    ta.check(len(f.slots) == 4, "2 fixtures + 2 objects = 4 slots")
    var kinds_ok = (
        f.slots[0].kind == SLOT_STATIC and f.slots[1].kind == SLOT_STATIC
        and f.slots[2].kind == SLOT_FREE and f.slots[3].kind == SLOT_FREE
    )
    ta.check(kinds_ok, "fixtures are STATIC slots and objects are FREE ones")
    # ⚠ THE UNRANGED REGION IS DROPPED, and the goal naming it is refused —
    # the two must agree or a task would name a region the family lacks.
    ta.check(len(f.regions) == 3,
             "only the RANGED regions become `region=` entries")
    # ⚠⚠ THE ASSET PATHS ARE `TODO:` ON PURPOSE. A `.bddl` names CATEGORIES;
    # emitting a path would be inventing one.
    ta.check(family_todo_count(f) == 5,
             "every asset is a visible TODO (1 base + 4 slots), not a guess")

    var gap = classify_goal(p)
    # ⚠⚠ THE **FIRST** GAP, NOT THE MOST INTERESTING ONE. This goal is
    # `(On .. cook_region) (On .. cook_region) (Turnon ..)` and I expected
    # ARTICULATION — but term 0 already blocks it, on the UNRANGED region, and
    # reporting that is what keeps the survey's gap counts equal to the number
    # of tasks blocked rather than to the number of bad terms.
    ta.check(gap.kind == GAP_FIXTURE_REGION,
             "the goal is refused at its FIRST blocking term: " + gap.term)
    var refused = False
    try:
        var _t = translate_task(p, f)
    except e:
        refused = True
    ta.check(refused, "and `translate_task` RAISES rather than approximating")

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "bddl: " + String(ta.failures) + " of " + String(ta.checks)
            + " failed"
        )
    print("=== PASS ===")
