"""Placement sampling — determinism, rejection, and exhaustion. P2b's gate.

⚠⚠ THE REJECTION BRANCH IS THE THING BEING TESTED, and it is easy to write a
gate that never reaches it. A sampler with the overlap check DELETED still
returns poses, still lands them in the region, and still reproduces per seed —
it passes every obvious assertion. So this gate constructs a CROWDED region
and asserts the attempt counter MOVES, and an IMPOSSIBLE one and asserts it
RAISES. Those two arms are the file's reason to exist; the rest is scaffolding.

Synthetic families on purpose: the pathological cases need region rectangles
and object radii chosen to make rejection certain, which no real family would
carry. One arm runs against the real `so101_tabletop` for realism.

Run: pixi run mojo run -I . tests/tasks/test_sampler.mojo
"""

from mojo_rl.tasks.spec import parse_family, parse_task
from mojo_rl.tasks.sampler import (
    sample_placements, RegionFrame, SampleReport, MAX_PLACE_ATTEMPTS,
)


comptime R: Float64 = 0.02   # every prop is a 2 cm-radius disc here


def _fam(rect: String) raises -> String:
    return String(
        "schema_version=1\nfamily=synth\nbase=b.xml\nhorizon=10\n"
        "slot=a:free:p.xml\nslot=b:free:p.xml\nslot=c:free:p.xml\n"
        "region=zone:site:s:"
    ) + rect + "\n"


comptime TASK3 = String(
    "schema_version=1\ntask=t\nfamily=synth\ngoal=In(a, zone)\n"
    "active=a\nactive=b\nactive=c\n"
    "init=a@zone\ninit=b@zone\ninit=c@zone\n"
)


def _frames() -> List[RegionFrame]:
    var fr = List[RegionFrame]()
    # A site 1 m up and offset, so a placement that ignored the frame lands
    # somewhere obviously wrong rather than plausibly near the origin.
    fr.append(RegionFrame(0.5, -0.25, 1.0))
    return fr^


def _radii() -> List[Float64]:
    var rr = List[Float64]()
    for _ in range(3):
        rr.append(R)
    return rr^


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
    print("=== placement sampling — P2b ===")
    var ta = Tally()
    var task = parse_task(TASK3)
    var fr = _frames()
    var rr = _radii()

    # ── 1. a roomy region: lands in bounds, on the surface, no overlap ─────
    print("--- a roomy region (0.20 x 0.20) ---")
    var roomy = parse_family(_fam(String("-0.10,-0.10,0.10,0.10")))
    var rep = SampleReport()
    var p = sample_placements(task, roomy, fr, rr, UInt64(7), 0, rep)
    ta.check(len(p) == 3, "all three inits placed")
    print("    attempts", rep.attempts, " accepted", rep.accepted)

    var in_bounds = True
    var on_surface = True
    for i in range(len(p)):
        if p[i].x < 0.5 - 0.10 or p[i].x > 0.5 + 0.10:
            in_bounds = False
        if p[i].y < -0.25 - 0.10 or p[i].y > -0.25 + 0.10:
            in_bounds = False
        # ⚠ RESTING ON the surface: centre one radius above the site, not AT
        # it. Placing at the site starts every episode half inside the table.
        if p[i].z < 1.0 + R - 1e-12 or p[i].z > 1.0 + R + 1e-12:
            on_surface = False
    ta.check(in_bounds, "every placement is inside the region, frame included")
    ta.check(on_surface, "every placement RESTS on the surface (z = site + r)")

    var clear = True
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            var dx = p[i].x - p[j].x
            var dy = p[i].y - p[j].y
            if dx * dx + dy * dy < (2.0 * R) * (2.0 * R) - 1e-15:
                clear = False
    ta.check(clear, "no two accepted placements overlap")

    # ── 2. determinism ────────────────────────────────────────────────────
    print("--- determinism ---")
    var rep2 = SampleReport()
    var p2 = sample_placements(task, roomy, fr, rr, UInt64(7), 0, rep2)
    var same = True
    for i in range(len(p)):
        if p[i].x != p2[i].x or p[i].y != p2[i].y:
            same = False
    ta.check(same, "same (seed, lane) -> IDENTICAL placements")

    var rep3 = SampleReport()
    var p3 = sample_placements(task, roomy, fr, rr, UInt64(7), 1, rep3)
    var differs = False
    for i in range(len(p)):
        if p[i].x != p3[i].x or p[i].y != p3[i].y:
            differs = True
    # ⚠ THE CONTROL FOR DETERMINISM. Without it, a sampler that returned a
    # CONSTANT pose would pass the line above perfectly.
    ta.check(differs, "a different LANE draws different placements")

    var rep4 = SampleReport()
    var p4 = sample_placements(task, roomy, fr, rr, UInt64(8), 0, rep4)
    var seed_differs = False
    for i in range(len(p)):
        if p[i].x != p4[i].x:
            seed_differs = True
    ta.check(seed_differs, "a different SEED draws different placements")

    # ── 3. ⚠ THE REJECTION BRANCH ACTUALLY RUNS ───────────────────────────
    print("--- a crowded region (0.08 x 0.08, three 2 cm discs) ---")
        # ⚠ TIGHT BUT FEASIBLE, and the difference matters. Three 4 cm-diameter
    # discs DO fit in 8 cm (corners are 8 cm apart) but most random draws
    # collide, so this exercises the retry loop heavily and still succeeds.
    # An earlier version used 5 cm, where three discs cannot fit AT ALL — it
    # raised, which is correct behaviour but tests the EXHAUSTION arm twice
    # and leaves the "rejects then succeeds" path uncovered.
    var crowded = parse_family(_fam(String("-0.04,-0.04,0.04,0.04")))
    var repc = SampleReport()
    var pc = sample_placements(task, crowded, fr, rr, UInt64(3), 0, repc)
    print("    attempts", repc.attempts, " accepted", repc.accepted,
          " rejected", repc.rejected())
    ta.check(len(pc) == 3, "a crowded region still places all three")
    # ⚠⚠ THIS IS THE ONE. A sampler with the overlap check deleted reports
    # attempts == accepted, and passes every other assertion in this file.
    ta.check(
        repc.rejected() > 0,
        "the REJECTION branch ran (attempts > accepted)",
    )
    var clear_c = True
    for i in range(len(pc)):
        for j in range(i + 1, len(pc)):
            var dx = pc[i].x - pc[j].x
            var dy = pc[i].y - pc[j].y
            if dx * dx + dy * dy < (2.0 * R) * (2.0 * R) - 1e-15:
                clear_c = False
    ta.check(clear_c, "and the crowded result is STILL overlap-free")

    # ── 4. ⚠ EXHAUSTION RAISES rather than returning an overlap ───────────
    print("--- an impossible region (1 mm, three 2 cm discs) ---")
    var tiny = parse_family(_fam(String("-0.0005,-0.0005,0.0005,0.0005")))
    var raised = False
    var rept = SampleReport()
    try:
        _ = sample_placements(task, tiny, fr, rr, UInt64(3), 0, rept)
    except e:
        raised = True
    ta.check(raised, "an over-constrained region RAISES")
    ta.check(
        rept.attempts >= MAX_PLACE_ATTEMPTS,
        "and it tried the full budget before giving up",
    )

    # ── 5. the real family, for realism ───────────────────────────────────
    print("--- the real so101_tabletop family ---")
    from mojo_rl.tasks.spec import load_family
    var real = load_family("mojo_rl/tasks/families/so101_tabletop.family")
    var rtask = parse_task(
        String("schema_version=1\ntask=pick\nfamily=so101_tabletop\n"
               "goal=In(brick, table_top)\n"
               "active=table\nactive=brick\nactive=cube_a\n"
               "init=brick@table_top\ninit=cube_a@table_top\n")
    )
    # ⚠ ONE FRAME PER REGION, IN FAMILY ORDER — `sample_placements` raises
    # otherwise, because it indexes `frames` by family region index. The
    # family declares three (`table_top` plus the two lateral init strips) and
    # they all sit on the SAME site, so one position serves for all of them
    # here; the rects differ and the rects live on the family, not the frame.
    var rfr = List[RegionFrame]()
    for _ in range(len(real.regions)):
        rfr.append(RegionFrame(0.25, 0.0, 0.31))  # roughly a table site
    var rrad = List[Float64]()
    for _ in range(len(real.slots)):
        rrad.append(R)
    var rrep = SampleReport()
    var rp = sample_placements(rtask, real, rfr, rrad, UInt64(1), 0, rrep)
    ta.check(len(rp) == 2, "the real family places its two free props")
    print("    brick at", rp[0].x, rp[0].y, rp[0].z)

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "sampler: " + String(ta.failures) + " of " + String(ta.checks)
            + " check(s) failed"
        )
    print("=== PASS ===")
