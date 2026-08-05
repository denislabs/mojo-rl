"""Gate for data-platform Stage 2b — windowed residency.

The stated gate is "window refill produces the same rows as a full-store
gather", so the comparison is against `ResidentColumn` (whole-column
residency) on identical indices, element-by-element.

Also pinned, because they are the design and not incidental:
  * a row outside the window RAISES rather than silently refilling — a
    refill-on-miss API would look like a gather and perform like disk;
  * `advance()` walks the column in non-overlapping slabs and reports
    exhaustion rather than wrapping silently;
  * `refills` counts slab reads, so a thrashing access pattern is visible.

Run:
    pixi run mojo run -I . tests/data/test_windowed.mojo
"""

from std.random import seed
from std.testing import assert_equal, assert_true

from mojo_rl.data import (
    ColumnSpec,
    IndexBatch,
    IDX_DT,
    ResidentColumn,
    TrajectoryStore,
    TrajectoryStoreWriter,
    WindowSampler,
    WindowedColumn,
)


comptime OUT = "/tmp/mojo_rl_windowed.h5"
comptime N_ROWS: Int = 96
comptime DIM: Int = 5
comptime EP: Int = 16
comptime WIN: Int = 24


def expected(row: Int, col: Int) -> Float32:
    return Float32(row) * 7.0 + Float32(col) * 0.5


def build() raises:
    print("[setup] write", N_ROWS, "rows ...")
    var cols = List[ColumnSpec]()
    cols.append(ColumnSpec(String("state"), DType.float32, DIM))
    var w = TrajectoryStoreWriter(
        String(OUT), cols^, env_id=String("windowed-test"), chunk_rows=8
    )
    var buf = List[Scalar[DType.float32]](unsafe_uninit_length=EP * DIM)
    var row = 0
    for _ in range(N_ROWS // EP):
        for i in range(EP):
            for c in range(DIM):
                buf[i * DIM + c] = expected(row + i, c)
        w.append[DType.float32](
            String("state"), buf.unsafe_ptr().as_unsafe_any_origin(), EP
        )
        w.end_episode()
        row += EP
    w.close()
    print("       OK")


def test_matches_full_residency() raises:
    """The stated gate: same rows, same values, as a fully-resident gather."""
    print("[1] windowed gather == full-residency gather ...")
    var s = TrajectoryStore(String(OUT))
    var full = ResidentColumn[DType.float32].load(s, String("state"))
    var win = WindowedColumn[DType.float32].make(s, String("state"), WIN)

    var total_checked = 0
    var start = 0
    while start + WIN <= N_ROWS:
        win.seek(s, start)
        # Every row of the window, in a deliberately scrambled order so a
        # sequential-copy bug cannot pass.
        var h = List[Scalar[IDX_DT]](unsafe_uninit_length=WIN)
        for k in range(WIN):
            h[k] = Scalar[IDX_DT](start + (k * 7 + 3) % WIN)
        var idx = IndexBatch(h^)

        var got = List[Scalar[DType.float32]]()
        win.gather_host(idx, got)
        var want = List[Scalar[DType.float32]]()
        full.gather_host(idx, want)

        assert_equal(len(got), len(want), "gather size")
        for i in range(len(want)):
            assert_equal(
                got[i], want[i],
                "window @" + String(start) + " element " + String(i),
            )
        total_checked += len(want)
        start += WIN

    assert_true(total_checked > 0, "no windows were checked")
    print("     ", total_checked, "elements identical across",
          N_ROWS // WIN, "windows  OK")


def test_out_of_window_raises() raises:
    print("[2] a row outside the window must RAISE ...")
    var s = TrajectoryStore(String(OUT))
    var win = WindowedColumn[DType.float32].make(s, String("state"), WIN)
    win.seek(s, 0)

    var h = List[Scalar[IDX_DT]](unsafe_uninit_length=2)
    h[0] = Scalar[IDX_DT](0)
    h[1] = Scalar[IDX_DT](WIN + 5)          # outside
    var idx = IndexBatch(h^)
    var out = List[Scalar[DType.float32]]()
    var raised = False
    try:
        win.gather_host(idx, out)
    except:
        raised = True
    assert_true(raised, "out-of-window row must raise, not refill silently")

    # And gathering before any seek must raise too.
    var fresh = WindowedColumn[DType.float32].make(s, String("state"), WIN)
    var h2 = List[Scalar[IDX_DT]](unsafe_uninit_length=1)
    h2[0] = Scalar[IDX_DT](0)
    var idx2 = IndexBatch(h2^)
    raised = False
    try:
        fresh.gather_host(idx2, out)
    except:
        raised = True
    assert_true(raised, "gather with no resident window must raise")
    print("      OK")


def test_advance_and_refill_count() raises:
    print("[3] advance() walks the column; refills are counted ...")
    var s = TrajectoryStore(String(OUT))
    var win = WindowedColumn[DType.float32].make(s, String("state"), WIN)

    var steps = 0
    while win.advance(s):
        steps += 1
        if steps > 100:
            raise Error("advance() did not terminate")
    # N_ROWS/WIN windows: the first advance seeks to 0, then it slides.
    assert_equal(steps, N_ROWS // WIN, "non-overlapping window count")
    assert_equal(win.refills, N_ROWS // WIN, "one slab read per window")
    assert_true(
        win.bytes_resident() < N_ROWS * DIM * 4,
        "the window must be smaller than the whole column",
    )
    print("      ", steps, "windows,", win.refills, "slab reads  OK")


def test_window_sampler_stays_inside() raises:
    """`WindowSampler` must only ever draw resident rows — that is the whole
    contract that makes windowed residency usable."""
    print("[4] WindowSampler draws only resident rows ...")
    var s = TrajectoryStore(String(OUT))
    var win = WindowedColumn[DType.float32].make(s, String("state"), WIN)
    win.seek(s, 2 * WIN)

    seed(20260805)
    var sampler = WindowSampler(win.window_start, win.window_rows)
    var idx = sampler.draw(64)
    for k in range(idx.size()):
        var r = Int(idx.host[k])
        assert_true(
            win.resident(r),
            "sampler drew row " + String(r) + " outside ["
            + String(win.window_start) + ", " + String(win.window_end()) + ")",
        )
    # It must actually spread over the window, not sit on one row.
    var lo = Int(idx.host[0])
    var hi = Int(idx.host[0])
    for k in range(idx.size()):
        var r = Int(idx.host[k])
        if r < lo:
            lo = r
        if r > hi:
            hi = r
    assert_true(hi - lo > WIN // 2, "draw is degenerate, not spread")

    # And the gather through the window must succeed for all of them.
    var out = List[Scalar[DType.float32]]()
    win.gather_host(idx, out)
    assert_equal(len(out), idx.size() * DIM, "gather size")
    print("      spread", lo, "->", hi, " OK")


def main() raises:
    build()
    test_matches_full_residency()
    test_out_of_window_raises()
    test_advance_and_refill_count()
    test_window_sampler_stays_inside()
    print("\n[PASS] windowed residency — Stage 2b")
