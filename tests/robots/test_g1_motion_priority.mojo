"""Motion prioritization: the formula, the clamp, and the quantization bound.

The interesting assertions are the two that could silently be wrong:

  [2] the CLAMP is flat outside [0.5, 2.0]. A motion at EMD 5 must get exactly
      the same weight as one at 2.0 — otherwise a hopeless clip (or a garbage
      eval reading) takes the whole batch.
  [4] the expansion's realised share is within the quantization bound of the
      exact priority share. It is NOT equal, by construction, so asserting
      equality would either fail or force a tolerance so loose it proves
      nothing. The bound is derived from REPEAT, not tuned.

Run:
    pixi run mojo run -I . tests/robots/test_g1_motion_priority.mojo
"""

from std.math import abs
from std.testing import assert_true

from noeira.envs.robots.g1_motion_priority import (
    G1_PRIO_MIN, G1_PRIO_MAX, G1_PRIO_SCALE,
    g1_motion_priority, g1_priority_shares, g1_fill_motion_table,
    g1_fill_window_table, g1_realised_share,
)


def test_formula() raises:
    print("[1] priority = 2^(2*clamp(emd, 0.5, 2)) ...")
    # the two endpoints the reference's literals pin
    var lo = g1_motion_priority(G1_PRIO_MIN)
    var hi = g1_motion_priority(G1_PRIO_MAX)
    print("      emd 0.5 ->", lo, "   emd 2.0 ->", hi, "   spread", hi / lo)
    assert_true(abs(lo - 2.0) < 1e-12, "emd 0.5 must give 2^1 = 2, got " + String(lo))
    assert_true(abs(hi - 16.0) < 1e-12, "emd 2.0 must give 2^4 = 16, got " + String(hi))
    assert_true(abs(hi / lo - 8.0) < 1e-12, "the spread must be 8x")
    # monotone in between
    var prev = 0.0
    for i in range(11):
        var e = G1_PRIO_MIN + (G1_PRIO_MAX - G1_PRIO_MIN) * Float64(i) / 10.0
        var p = g1_motion_priority(e)
        assert_true(p > prev, "priority must increase with EMD")
        prev = p


def test_clamp_is_flat_outside() raises:
    print("[2] the clamp is FLAT outside [0.5, 2.0] ...")
    var at_min = g1_motion_priority(G1_PRIO_MIN)
    var at_max = g1_motion_priority(G1_PRIO_MAX)
    print("      emd 0.0 ->", g1_motion_priority(0.0),
          "  emd 5.0 ->", g1_motion_priority(5.0),
          "  emd 100 ->", g1_motion_priority(100.0))
    assert_true(
        abs(g1_motion_priority(0.0) - at_min) < 1e-12
        and abs(g1_motion_priority(0.2) - at_min) < 1e-12,
        "below the clamp must be FLAT — a solved motion must not fall to zero",
    )
    assert_true(
        abs(g1_motion_priority(5.0) - at_max) < 1e-12
        and abs(g1_motion_priority(100.0) - at_max) < 1e-12,
        "above the clamp must be FLAT — one hopeless clip, or one garbage eval"
        " reading, must not be able to take the whole batch",
    )


def test_motion_table_matches_the_shares() raises:
    """A uniform draw over the filled table must land at `share_m`."""
    print("[3] the motion table realises the priority shares ...")
    var emd = List[Float64]()
    for i in range(20):
        emd.append(0.4 + 0.09 * Float64(i))      # straddles BOTH clamps
    var share = g1_priority_shares(emd)
    comptime L = 4096
    var table = List[Int]()
    g1_fill_motion_table(emd, L, table)
    assert_true(len(table) == L, "the table must be exactly L long, got " + String(len(table)))
    var worst = 0.0
    for m in range(len(emd)):
        var got = g1_realised_share(table, m)
        var rel = abs(got - share[m]) / share[m]
        if rel > worst:
            worst = rel
    print("      L =", L, " worst relative share error:", worst)
    # resolution is 1/L against shares of order 1/20, so this is tiny
    assert_true(
        worst < 0.02,
        "realised share is " + String(worst) + " off the priority share",
    )
    # and the spread must actually be REALISED, not flattened
    var s_lo = g1_realised_share(table, 0)
    var s_hi = g1_realised_share(table, len(emd) - 1)
    print("      best-tracked share", s_lo, "  worst-tracked share", s_hi,
          "  ratio", s_hi / s_lo)
    assert_true(
        s_hi / s_lo > 7.0,
        "the 8x priority spread collapsed to " + String(s_hi / s_lo)
        + "x in the table — the weighting is not reaching the sampler",
    )


def test_no_motion_is_dropped() raises:
    """A solved motion must keep slots. Dropping it is how it gets un-learned."""
    print("[4] a well-tracked motion keeps a slot ...")
    var emd = List[Float64]()
    emd.append(0.05)          # essentially solved
    for _ in range(199):
        emd.append(2.0)       # 199 hopeless clips competing for the table
    var table = List[Int]()
    g1_fill_motion_table(emd, 256, table)
    var c0 = g1_realised_share(table, 0)
    print("      solved motion share:", c0, " of", len(table), "slots")
    assert_true(
        c0 > 0.0,
        "the solved motion lost every slot — it will be un-learned, which is"
        " exactly what the clamp exists to prevent",
    )


def test_window_table_cycles() raises:
    """Cycling, not truncation: successive refreshes must reach every window."""
    print("[5] the window table cycles through a clip's windows ...")
    # one clip, 10 windows, a table with room for 4 of them
    var items = List[Int]()
    for i in range(10):
        items.append(1000 + i)
    var beg = List[Int](); beg.append(0)
    var end = List[Int](); end.append(10)
    var emd = List[Float64](); emd.append(1.0)
    var cursor = List[Int](); cursor.append(0)
    var seen = List[Int]()
    for _ in range(10):
        seen.append(0)
    var out = List[Int]()
    for _ in range(4):                     # four refreshes of a 4-slot table
        g1_fill_window_table(items, beg, end, emd, cursor, 4, out)
        for i in range(len(out)):
            seen[out[i] - 1000] = 1
    var n_seen = 0
    for i in range(10):
        n_seen += seen[i]
    print("      distinct windows reached over 4 refreshes:", n_seen, "of 10")
    assert_true(
        n_seen == 10,
        "only " + String(n_seen) + " of 10 windows were ever emitted — the"
        " table is truncating instead of cycling, so the tail of every long"
        " clip is dead data",
    )


def main() raises:
    print("=== G1 motion prioritization ===")
    test_formula()
    test_clamp_is_flat_outside()
    test_motion_table_matches_the_shares()
    test_no_motion_is_dropped()
    test_window_table_cycles()
    print("=== all passed ===")
