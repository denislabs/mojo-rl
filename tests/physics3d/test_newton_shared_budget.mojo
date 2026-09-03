"""`je_spills` budgets the WHOLE threadgroup footprint. P4's gate.

WHY THIS EXISTS
===============
`je_spills` compared **`Je` alone** against a 64 KB constant. At the k=12 park
scene `Je` is 54 KB — comfortably under — so it declined to spill, while the
kernel's three `NV*NV` matrices put the block at 136,212 B against a 101,376 B
limit and `ptxas` refused to compile it:

    ptxas error : Entry function 'mojo_rl_physics3d_solver_newt...' uses
                  too much shared data (0x21414 bytes, 0x18c00 max)

Budgeting one array out of eleven cannot predict that. The models it WAS tuned
on — humanoid_CMU, dog — hid it, because they are high-nv AND high-contact so
`Je` dominates the total. A fixed scene budget produces the shape it was never
tuned for: high nv, LOW contact count.

⚠⚠ THE FORMULA IS PINNED TO `ptxas`, NOT TO ITSELF. `newton_shared_elems` is a
transcription of the kernel's `stack_allocation()` list, and a transcription
that drifts is the whole failure being fixed here. Arm A checks it against the
four byte counts `ptxas` actually printed for the k=6/9/10/12 park scenes —
48,372 / 86,676 / 101,940 / 136,212 — which is an EXTERNAL oracle, obtained
before any of this code existed.

⚠ THOSE FOUR NUMBERS PREDATE PN2c, which added `seg0_sh`/`seg1_sh` — `2 * NV`
scalars. So the expectation is `recorded + 8*NV` bytes, and the test spells the
delta out rather than folding it in: if someone adds another shared array, this
arm fails and names the size, instead of the model failing to compile later.

Run: pixi run mojo run -I . tests/physics3d/test_newton_shared_budget.mojo
"""

from std.sys.info import size_of
from mojo_rl.physics3d.solver.je_budget import (
    newton_shared_elems, je_spills, je_elems, SOLVER_SHARED_BUDGET,
)

comptime DT = DType.float32          # the park probe's dtype
comptime MC = 16                     # PARK_MAX_CONTACTS
comptime CONDIM = 3


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def _bytes[NV: Int, NJOINT: Int]() -> Int:
    return newton_shared_elems[
        NV, NJOINT, 0, 0, MC, CONDIM, True
    ]() * size_of[Scalar[DT]]()


def _je_bytes[NV: Int, NJOINT: Int]() -> Int:
    return je_elems[NV, NJOINT, 0, 0, MC, CONDIM]() * size_of[Scalar[DT]]()


def main() raises:
    var t = Tally()
    print("=== the Newton kernel's shared footprint (P4) ===")
    print("  budget:", SOLVER_SHARED_BUDGET, "B  (ptxas 0x18c00 on an RTX 5090)")

    # ── A: the formula reproduces ptxas, to the byte ─────────────────────
    # (k, nv, njoint, the bytes ptxas reported BEFORE PN2c's two seg arrays)
    print("--- A: vs the four byte counts ptxas printed ---")
    var seg_delta_42 = 8 * 42
    var seg_delta_60 = 8 * 60
    var seg_delta_66 = 8 * 66
    var seg_delta_78 = 8 * 78
    t.truth(_bytes[42, 12]() == 48372 + seg_delta_42,
            String("k=6  nv=42: ", _bytes[42, 12](), " == 48372 + ",
                   seg_delta_42, " (ptxas + PN2c's 2*NV)"))
    t.truth(_bytes[60, 15]() == 86676 + seg_delta_60,
            String("k=9  nv=60: ", _bytes[60, 15](), " == 86676 + ",
                   seg_delta_60))
    t.truth(_bytes[66, 16]() == 101940 + seg_delta_66,
            String("k=10 nv=66: ", _bytes[66, 16](), " == 101940 + ",
                   seg_delta_66))
    t.truth(_bytes[78, 18]() == 136212 + seg_delta_78,
            String("k=12 nv=78: ", _bytes[78, 18](), " == 136212 + ",
                   seg_delta_78))

    # ── B: the decision flips where ptxas does ───────────────────────────
    print("--- B: k<=9 keeps threadgroup Je, k>=10 spills ---")
    t.truth(not je_spills[DT, 42, 12, 0, 0, MC, CONDIM](),
            "k=6  does NOT spill (48,708 B fits)")
    t.truth(not je_spills[DT, 60, 15, 0, 0, MC, CONDIM](),
            "k=9  does NOT spill (87,156 B fits) — the ceiling today")
    t.truth(je_spills[DT, 66, 16, 0, 0, MC, CONDIM](),
            "k=10 SPILLS (102,468 B over) — was a COMPILE FAILURE")
    t.truth(je_spills[DT, 78, 18, 0, 0, MC, CONDIM](),
            "k=12 SPILLS (136,836 B over) — was a COMPILE FAILURE")

    # ── C: ⚠ THE OLD RULE WOULD HAVE GOT B WRONG. Without this the gate
    # only says the new code agrees with itself.
    print("--- C: the old `Je`-alone rule, on the same models ---")
    var je10 = _je_bytes[66, 16]()
    var je12 = _je_bytes[78, 18]()
    t.truth(je10 <= 64 * 1024,
            String("k=10 Je alone = ", je10, " B — UNDER the old 64 KB, so the"
                   " old rule did not spill and the kernel did not compile"))
    t.truth(je12 <= 64 * 1024,
            String("k=12 Je alone = ", je12, " B — likewise"))

    # ── D: after spilling, the block actually fits ───────────────────────
    print("--- D: spilling is enough — the rest fits ---")
    var r10 = newton_shared_elems[66, 16, 0, 0, MC, CONDIM, False]() * 4
    var r12 = newton_shared_elems[78, 18, 0, 0, MC, CONDIM, False]() * 4
    t.truth(r10 <= SOLVER_SHARED_BUDGET,
            String("k=10 with Je spilled: ", r10, " B fits"))
    t.truth(r12 <= SOLVER_SHARED_BUDGET,
            String("k=12 with Je spilled: ", r12, " B fits"))
    # ⚠ AND WHERE IT STOPS BEING ENOUGH, so nobody reads "P4 unblocks k" as
    # unbounded. Past this the three NV*NV arrays are the binding term.
    var r14 = newton_shared_elems[90, 20, 0, 0, MC, CONDIM, False]() * 4
    t.truth(r14 > SOLVER_SHARED_BUDGET,
            String("k=14 with Je spilled: ", r14, " B still OVER — spilling"
                   " reaches k=13, not further"))

    # ── E: ⚠ NO SHIPPED MODEL CHANGES ITS MIND. Widening the budget from
    # "Je vs 64 KB" to "the total vs the device limit" could easily have made
    # models that run today start spilling — a straight perf regression, since
    # a spilled `Je` is re-read from global across every Newton iteration.
    # The six models `je_budget`'s own table records must keep their answer.
    print("--- E: the six models in je_budget's table are unmoved ---")
    t.truth(not je_spills[DT, 22, 78, 0, 0, 16, CONDIM](),
            "quadruped       (nv 22) still does NOT spill")
    t.truth(not je_spills[DT, 27, 96, 0, 0, 32, CONDIM](),
            "humanoid        (nv 27) still does NOT spill")
    t.truth(not je_spills[DT, 28, 156, 0, 0, 24, 6](),
            "quadruped_fetch (nv 28) still does NOT spill")
    t.truth(je_spills[DT, 62, 185, 0, 0, 64, CONDIM](),
            "humanoid_CMU    (nv 62) still SPILLS")
    t.truth(je_spills[DT, 79, 206, 0, 0, 24, CONDIM](),
            "dog             (nv 79) still SPILLS")
    t.truth(je_spills[DT, 85, 227, 0, 0, 28, CONDIM](),
            "dog_fetch       (nv 85) still SPILLS")

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error(
            "test_newton_shared_budget: " + String(t.fails) + " failed"
        )
