"""The Open-model browser's ordering and its two derived columns.

WHY THIS EXISTS
===============
`listdir` returns entries in whatever order the filesystem hands back, and the
browser showed them that way — so "open model" listed a Menagerie directory in
effectively arbitrary order. This gates the replacement.

⚠⚠ THE DATE FORMATTER IS THE PART THAT CAN BE QUIETLY WRONG. Days-to-civil has
no gradual failure mode near a leap year: it is exact for 2024-02-29 or it is a
day out for the next twelve months, and nobody reads a file browser's timestamp
closely enough to notice. The 25 cases below were generated from Python's
`datetime.fromtimestamp(..., utc)` — an EXTERNAL oracle, and the boundaries
(epoch, day rollover, three leap days, 2000, 2100, the 32-bit wrap) are chosen,
not sampled.

⚠ AND THE ORDER ARMS ASSERT WHAT MUST **NOT** MOVE. Reversing the sort must not
send the directories to the bottom, and a key with equal values must fall back
to the name — a listing that re-shuffles under a stationary cursor is worse
than one sorted by something you did not pick.

Run: pixi run mojo run -I . tests/physics3d/test_browser_sort.mojo
"""

from mojo_rl.physics3d.studio.panel import (
    _Entry, _sort_entries, _fmt_time, _fmt_size, _fold,
    SORT_NAME, SORT_SIZE, SORT_TIME,
)


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


def names_of(e: List[_Entry]) -> String:
    var s = String("")
    for i in range(len(e)):
        if i > 0:
            s += " "
        s += e[i].name
    return s^


def zoo() -> List[_Entry]:
    """⚠ DELIBERATELY OUT OF ORDER IN EVERY KEY, and mixed case. A fixture
    that arrives sorted by name would make the name arm vacuous."""
    var e = List[_Entry]()
    e.append(_Entry(String("zebra.xml"), 300, 500, False))
    e.append(_Entry(String("Ant.xml"), 100, 900, False))
    e.append(_Entry(String("assets"), 0, 100, True))
    e.append(_Entry(String("banana.xml"), 100, 700, False))
    e.append(_Entry(String("Zoo"), 0, 800, True))
    e.append(_Entry(String("apple.xml"), 900, 300, False))
    return e^


def main() raises:
    var t = Tally()
    print("=== the file browser's order and columns ===")

    # ── name, ascending ───────────────────────────────────────────────────
    print("--- name ascending ---")
    var a = zoo()
    _sort_entries(a, SORT_NAME, False)
    t.truth(names_of(a) == "assets Zoo Ant.xml apple.xml banana.xml zebra.xml",
            String("dirs first, then case-insensitive by name: ",
                   names_of(a)))
    # ⚠ THE CASE ARM IS THE POINT. Raw byte order puts every capitalised name
    # above every lowercase one, so `Ant.xml` and `apple.xml` adjacent — in
    # that order — is what says the fold ran.
    t.truth(_fold(String("Ant.XML")) == "ant.xml",
            "the fold is ASCII lowercase")

    # ── name, descending ──────────────────────────────────────────────────
    print("--- name descending ---")
    var b = zoo()
    _sort_entries(b, SORT_NAME, True)
    t.truth(names_of(b) == "Zoo assets zebra.xml banana.xml apple.xml Ant.xml",
            String("files reverse and the DIRS STAY ON TOP: ", names_of(b)))
    # ⚠⚠ THE NON-OBVIOUS HALF. A reversal that also flipped the dir/file
    # grouping would put the subdirectories below every file — the browser
    # would look broken rather than reversed.
    t.truth(b[0].is_dir and b[1].is_dir and not b[2].is_dir,
            "the two directories are still the first two rows")

    # ── size ──────────────────────────────────────────────────────────────
    print("--- size ---")
    var c = zoo()
    _sort_entries(c, SORT_SIZE, True)
    t.truth(names_of(c) == "assets Zoo apple.xml zebra.xml Ant.xml banana.xml",
            String("biggest first, NAME breaking the 100/100 tie: ",
                   names_of(c)))
    var d = zoo()
    _sort_entries(d, SORT_SIZE, False)
    t.truth(names_of(d) == "assets Zoo Ant.xml banana.xml zebra.xml apple.xml",
            String("smallest first: ", names_of(d)))
    # ⚠ THE TIE-BREAK IS NOT DECORATION. `Ant.xml` and `banana.xml` are both
    # 100 bytes; without it their order is whatever `listdir` happened to
    # return, and the listing re-shuffles between frames.
    t.truth(names_of(c).find(String("Ant.xml banana.xml")) != -1
            and names_of(d).find(String("Ant.xml banana.xml")) != -1,
            "equal sizes stay in NAME order in BOTH directions")

    # ── modified ──────────────────────────────────────────────────────────
    print("--- modified ---")
    var e = zoo()
    _sort_entries(e, SORT_TIME, True)
    # ⚠ `Zoo` BEFORE `assets` IS CORRECT, and it is where this test's first
    # draft was wrong. "Directories first" groups them; it does not exempt
    # them from the sort. Zoo's mtime is 800 and assets' is 100, so newest-
    # first puts Zoo on top — which is what Finder does with Folders on Top.
    # Under `size` they tie at 0 and fall back to the name, hence
    # "assets Zoo" there.
    t.truth(names_of(e) == "Zoo assets Ant.xml banana.xml zebra.xml apple.xml",
            String("newest first, dirs sorted among themselves too: ",
                   names_of(e)))
    var f = zoo()
    _sort_entries(f, SORT_TIME, False)
    t.truth(names_of(f) == "assets Zoo apple.xml zebra.xml banana.xml Ant.xml",
            String("oldest first: ", names_of(f)))

    # ⚠ NON-VACUITY: the four orders above must not all be the same string,
    # which they would be if `_sort_entries` ignored its arguments.
    var seen = List[String]()
    seen.append(names_of(a))
    seen.append(names_of(b))
    seen.append(names_of(c))
    seen.append(names_of(e))
    var distinct = 0
    for i in range(len(seen)):
        var dup = False
        for j in range(i):
            if seen[i] == seen[j]:
                dup = True
        if not dup:
            distinct += 1
    t.truth(distinct == 4,
            String("the four keys produce four DIFFERENT orders — got ",
                   distinct))

    # ── a sorted input stays sorted, and a reversed one is fixed ─────────
    print("--- idempotence ---")
    var g = zoo()
    _sort_entries(g, SORT_NAME, False)
    var once = names_of(g)
    _sort_entries(g, SORT_NAME, False)
    t.truth(names_of(g) == once, "sorting an already-sorted list is a no-op")

    # ── the date column, against Python's datetime ───────────────────────
    print("--- _fmt_time vs datetime.fromtimestamp(utc) ---")
    var ts = List[Int]()
    var want = List[String]()
    # ⚠ 0 IS DELIBERATELY ABSENT FROM THIS TABLE — it is the "stat refused"
    # sentinel and is gated below. `1` covers the same civil arithmetic.
    ts.append(1); want.append(String("1970-01-01 00:00"))
    ts.append(86399); want.append(String("1970-01-01 23:59"))
    ts.append(86400); want.append(String("1970-01-02 00:00"))
    ts.append(207388625); want.append(String("1976-07-28 07:57"))
    ts.append(311111476); want.append(String("1979-11-10 19:51"))
    ts.append(404285458); want.append(String("1982-10-24 05:30"))
    ts.append(647892280); want.append(String("1990-07-13 18:04"))
    ts.append(951782400); want.append(String("2000-02-29 00:00"))
    ts.append(1078012800); want.append(String("2004-02-29 00:00"))
    ts.append(1104537600); want.append(String("2005-01-01 00:00"))
    ts.append(1234567890); want.append(String("2009-02-13 23:31"))
    ts.append(1390851129); want.append(String("2014-01-27 19:32"))
    ts.append(1570621945); want.append(String("2019-10-09 11:52"))
    ts.append(1583020800); want.append(String("2020-03-01 00:00"))
    ts.append(1695753999); want.append(String("2023-09-26 18:46"))
    ts.append(1709164800); want.append(String("2024-02-29 00:00"))
    ts.append(1786824990); want.append(String("2026-08-15 20:16"))
    ts.append(2147483647); want.append(String("2038-01-19 03:14"))
    ts.append(2301595692); want.append(String("2042-12-07 20:08"))
    ts.append(2503055454); want.append(String("2049-04-26 13:10"))
    ts.append(2795742289); want.append(String("2058-08-05 03:04"))
    ts.append(3527346213); want.append(String("2081-10-10 18:23"))
    ts.append(4071050725); want.append(String("2099-01-02 15:25"))
    ts.append(4102444800); want.append(String("2100-01-01 00:00"))

    var bad = 0
    for i in range(len(ts)):
        var got = _fmt_time(ts[i])
        if got != want[i]:
            bad += 1
            if bad <= 5:
                print("       ", ts[i], ": got", got, "want", want[i])
    t.truth(bad == 0,
            String(len(ts), " timestamps match Python exactly (", bad,
                   " wrong)"))
    t.truth(len(ts) == 24, String("the comparison was not empty: ", len(ts)))
    # ⚠⚠ 0 IS THE "stat REFUSED" SENTINEL, not a date. A broken symlink or a
    # permission hole still gets a row (the browser's job is to show what is
    # there), and the column reads "--" rather than confidently claiming the
    # file was last touched in 1970. The cost is that a genuine
    # 1970-01-01T00:00:00Z file also reads "--"; there is no such file.
    t.truth(_fmt_time(0) == "--" and _fmt_time(-1) == "--",
            "a zero or negative stamp shows '--'")
    t.truth(_fmt_time(1) == "1970-01-01 00:00",
            "and one second later is a real date (the sentinel is exact)")

    # ── the size column ───────────────────────────────────────────────────
    print("--- _fmt_size ---")
    t.truth(_fmt_size(0, True) == "--", "a directory has no size")
    t.truth(_fmt_size(999, False) == "999 B", "bytes below 1 KiB")
    t.truth(_fmt_size(1024, False) == "1 KB", "the KiB boundary")
    t.truth(_fmt_size(4935, False) == "5 KB",
            String("ant.xml's real size rounds: ", _fmt_size(4935, False)))
    t.truth(_fmt_size(1024 * 1024, False) == "1 MB", "the MiB boundary")

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_browser_sort: " + String(t.fails) + " failed")
