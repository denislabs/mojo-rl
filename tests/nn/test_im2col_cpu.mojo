# +--------------------------------------------------------------------------+ #
# | `_im2col_cpu`, against an obviously-correct reference
# +--------------------------------------------------------------------------+ #
"""Gate the CPU im2col — the function that is 60% of ACT's CPU forward.

    pixi run mojo run -I . tests/nn/test_im2col_cpu.mojo

⚠⚠ **THE REFERENCE IS WRITTEN HERE, NOT IMPORTED.** `_im2col_cpu` was rewritten
for speed (bounds test hoisted out of the inner loop, addresses strength-
reduced, separate K==1 path), so the thing it must be checked against is a
statement of WHAT IM2COL IS — five nested loops and one conditional, slow and
obvious — and not any part of the code under test. A gate that shares its
reference implementation is blind; this repo has that failure recorded twice.

⚠⚠ **THE SHAPES A REAL MODEL USES ARE NOT AN ADVERSARIAL TEST.** The rewrite's
prototype passed all 11 distinct ResNet18 shapes while missing a clamp that
lets the zero-fill run off the end of a row whenever `P >= K` — because no
ResNet has `P >= K`. So the shape table below is in two halves: the ones ACT
actually runs, and the ones chosen to break a bounds computation. The second
half is the half that earns this file.

⚠ EXACT EQUALITY. im2col moves bytes; it does no arithmetic, so there is no
rounding to tolerate and anything but bit equality is a bug.

⚠⚠ **AND IT COVERS BOTH DISPATCH PATHS.** `_im2col_cpu` puts itself on more
than one core above a size threshold, so half the shapes below are there to
run the THREADED body against the same single-threaded reference — a cross-
thread write that the compiler folds away, or a row written twice, is exactly
the kind of failure that a benchmark reports as a speed-up. Every line of the
output says which path it took; if they ever all say `[serial]`, this file has
stopped testing the code that runs.

⚠ Covers BOTH layouts. NCHW takes the rewritten path, NHWC still takes the
original loop, and the reference does not know the difference — it indexes
through the same `_col_off` / `_in_off` the rest of the file uses.
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.conv2d import (
    LAYOUT_NCHW,
    LAYOUT_NHWC,
    _col_off,
    _im2col_cpu,
    _in_off,
    im2col_uses_threads,
)


def reference[
    IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, OH: Int, OW: Int,
    LAYOUT: Int,
](ref x: List[Scalar[DT]], in_off: Int, mut col: List[Scalar[DT]]):
    """What im2col IS. One output element per (oh, ow, ic, kh, kw), taken from
    the input when it lands inside it and zero when it does not."""
    comptime CK = IC * K * K
    for oh in range(OH):
        for ow in range(OW):
            var row_off = (oh * OW + ow) * CK
            for ic in range(IC):
                for kh in range(K):
                    for kw in range(K):
                        var ih = oh * S + kh - P
                        var iw = ow * S + kw - P
                        var c = row_off + _col_off[LAYOUT, IC, K](ic, kh, kw)
                        if ih < 0 or ih >= H or iw < 0 or iw >= W:
                            col[c] = Scalar[DT](0)
                        else:
                            col[c] = x[
                                in_off + _in_off[LAYOUT, IC, H, W](ic, ih, iw)
                            ]


def shape[
    IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, OH: Int, OW: Int,
    LAYOUT: Int = LAYOUT_NCHW,
](name: String, mut failures: Int, mut compared: Int) raises:
    comptime CK = IC * K * K
    comptime N = OH * OW * CK
    comptime IN_N = IC * H * W
    # ⚠ AN OFFSET INPUT, so a path that ignores `in_off` cannot pass. The real
    # caller walks a batch with `b * IN_FLAT`.
    comptime OFF = IN_N

    var x = List[Scalar[DT]](unsafe_uninit_length=IN_N * 2)
    var st = UInt64(0x9E3779B97F4A7C15)
    for i in range(IN_N * 2):
        st = st * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        # Distinct, exactly representable values: a duplicate could let a
        # wrong-but-nearby index compare equal by luck.
        x[i] = Scalar[DT](Float64(Int(st >> 44)) / 1048576.0)

    # ⚠ A POISON FILL, NOT ZEROS. Every element must be WRITTEN by the code
    # under test; pre-zeroing would let a path that simply skips an element
    # agree with a reference that wrote a legitimate zero there.
    #
    # ⚠⚠ AND A CANARY PAST THE END, because the bug this function is most
    # likely to have does not produce a wrong VALUE — it produces a write past
    # the row it owns. Inside the buffer a later iteration overwrites the
    # damage, so a plain value comparison sees nothing; only at the very last
    # row does it leave the buffer, and then it is someone else's memory.
    # Sabotage-tested: with the `kw_lo` clamp removed this file passed
    # completely until the canary existed.
    comptime GUARD = 16
    var got = List[Scalar[DT]](length=N + GUARD, fill=Scalar[DT](-12345.0))
    var want = List[Scalar[DT]](length=N, fill=Scalar[DT](-54321.0))
    _im2col_cpu[IC, K, S, P, H, W, OH, OW, LAYOUT](x, OFF, got)
    reference[IC, K, S, P, H, W, OH, OW, LAYOUT](x, OFF, want)

    var diff = 0
    var first = -1
    var unwritten = 0
    for i in range(N):
        if got[i] == Scalar[DT](-12345.0):
            unwritten += 1
        if got[i] != want[i]:
            diff += 1
            if first < 0:
                first = i
    var trampled = 0
    for i in range(N, N + GUARD):
        if got[i] != Scalar[DT](-12345.0):
            trampled += 1
    compared += N
    if diff != 0 or unwritten != 0 or trampled != 0:
        failures += 1
        print(
            "  FAIL " + name + ": " + String(diff) + " of " + String(N)
            + " differ (first at " + String(first) + "), " + String(unwritten)
            + " never written, " + String(trampled) + " bytes PAST THE END"
        )
    else:
        # ⚠ REPORT WHICH PATH RAN. `_im2col_cpu` splits itself across cores
        # above a size threshold, and a gate whose every shape fell on the
        # serial side would be green while never executing the threaded body
        # at all — the vacuity failure, one level up from the values.
        # ⚠ THE PREDICATE IS IMPORTED, NOT RESTATED. Rewriting `OH >= 16 and
        # ELEMS >= 200_000` here would leave the gate reporting a path the
        # implementation had since stopped taking.
        comptime par = im2col_uses_threads[OH, N]()
        print(
            "  ok   " + name + "  " + String(N) + " elements  "
            + ("[threads]" if par else "[serial] ")
        )


def main() raises:
    print("=" * 70)
    print("_im2col_cpu vs an independent reference")
    print("=" * 70)
    var failures = 0
    var compared = 0

    # ── half one: what ACT actually runs (ResNet18 at 240x320) ────────────
    print(" ResNet18 / 240x320 — the shapes ACT runs")
    shape[3, 7, 2, 3, 240, 320, 120, 160]("conv1  7x7 s2 p3", failures, compared)
    shape[64, 3, 1, 1, 60, 80, 60, 80]("layer1 3x3 s1 p1", failures, compared)
    shape[64, 3, 2, 1, 60, 80, 30, 40]("l2.0c1 3x3 s2 p1", failures, compared)
    shape[64, 1, 2, 0, 60, 80, 30, 40]("l2down 1x1 s2 p0", failures, compared)
    shape[512, 3, 1, 1, 8, 10, 8, 10]("layer4 3x3 s1 p1", failures, compared)

    # ── half two: shapes chosen to break a bounds computation ─────────────
    print(" adversarial — no ResNet has these")
    # P >= K: the `kw` window can fall ENTIRELY outside the input. This is the
    # case the prototype got wrong, and it corrupts the NEXT row rather than
    # producing a wrong value, so it is invisible in the column it belongs to.
    shape[2, 1, 1, 2, 5, 5, 9, 9]("K=1 P=2  window outside", failures, compared)
    shape[3, 2, 1, 3, 4, 4, 9, 9]("K=2 P=3  window outside", failures, compared)
    # Kernel WIDER than the input: every row is partly padding, both ends.
    shape[2, 5, 1, 1, 3, 3, 1, 1]("K=5 > W=3", failures, compared)
    # Stride past the kernel — output columns that skip input entirely.
    shape[2, 2, 3, 0, 9, 9, 3, 3]("S=3 > K=2", failures, compared)
    # Degenerate extents: one channel, one output pixel, a 1-wide input.
    shape[1, 3, 1, 1, 1, 1, 1, 1]("IC=1 H=W=1", failures, compared)
    shape[2, 3, 1, 0, 3, 1, 1, 1]("W=1 column input", failures, compared)
    shape[2, 3, 2, 1, 4, 4, 2, 2]("even H,W with pad", failures, compared)
    # ⚠⚠ THE SHAPE THAT EARNS THE CANARY, and it had to be DERIVED. Every
    # condition below is load-bearing; drop one and the sabotaged code passes:
    #
    #   P > K            the `kw` window can fall entirely left of the input
    #   OW == 1          ...at the LAST output column, so nothing overwrites
    #                    the spill afterwards. Needs W + 2P - K < S.
    #   OH > 1           so the LAST output ROW has a VALID `ih`. With one row
    #                    the `ih < 0` branch fires first and the `kw` window is
    #                    never consulted. Needs H + 2P - K >= S, hence H > W.
    #
    # K=2 P=3 S=6 H=6 W=1 -> OH=2, OW=1: at (oh=1, ow=0) the row index is 3,
    # inside the input, while the column window starts 3 left of it. Unclamped,
    # the zero-fill writes three values into a two-wide slot and the last one
    # lands one past the end of the buffer.
    #
    # Two earlier attempts at "adversarial" shapes did NOT trip it — the first
    # by intuition, the second by half the derivation — and both PASSED against
    # deliberately broken code. That is what vacuity looks like from the
    # inside: a green gate over a bug it cannot reach.
    shape[2, 2, 6, 3, 6, 1, 2, 1]("K=2 P=3 S=6 OW=1 OH=2  spills off the end",
                                  failures, compared)

    # ── the dispatch boundary itself ──────────────────────────────────────
    # ⚠ A THRESHOLD NEEDS A SHAPE ON EACH SIDE OF IT, or it is only ever
    # tested in one direction. These two differ by one output row: OH=16 is
    # the first value that threads, OH=15 the last that does not, and both
    # carry enough elements to clear the work condition.
    shape[64, 3, 1, 1, 16, 220, 16, 220]("OH=16  first threaded", failures,
                                         compared)
    shape[64, 3, 1, 1, 15, 220, 15, 220]("OH=15  last serial", failures,
                                         compared)

    # ── the other layout, which keeps the original loop ───────────────────
    print(" NHWC — the untouched path")
    shape[4, 3, 1, 1, 6, 7, 6, 7, LAYOUT_NHWC]("nhwc 3x3 s1 p1", failures,
                                              compared)
    shape[3, 1, 2, 0, 8, 8, 4, 4, LAYOUT_NHWC]("nhwc 1x1 s2 p0", failures,
                                              compared)
    shape[2, 1, 1, 2, 5, 5, 9, 9, LAYOUT_NHWC]("nhwc K=1 P=2", failures,
                                              compared)

    print("")
    if failures != 0:
        raise Error(String(failures) + " shape(s) FAILED")
    print(
        "[PASS] " + String(compared) + " elements compared across 18 shapes,"
        " all exact"
    )
