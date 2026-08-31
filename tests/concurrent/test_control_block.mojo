"""ControlBlock — atomics across a real pthread, and the miscompile gate.

Run: pixi run mojo run -I . tests/concurrent/test_control_block.mojo

The third case is the important one. It is a regression gate for a SILENT
MISCOMPILE, not a feature test: with a `List[Int64]` backing the slab, the
second of two straight-line reads folds back to the List's initializer and
returns 0 instead of the value six threads just wrote.

⚠ DO NOT PUT A `print` BETWEEN THE TWO READS IN `stale_read_after_join`. An
opaque call in between is enough to stop the fold, and the gate becomes
vacuous — it passes on the broken backing too. That is exactly how the bug got
past the first driver written for it.
"""

from mojo_rl.core.concurrent.block import (
    CELLS_PER_LINE,
    ControlBlock,
    ControlBlockView,
)
from mojo_rl.core.concurrent.thread import (
    OpaquePtr,
    ThreadHandle,
    null_opaque,
    opaque_from_address,
)


comptime THREADS = 6
comptime BUMPS = 5000


def _bump(arg: OpaquePtr) -> OpaquePtr:
    """Worker: `fetch_add` cell 0, `BUMPS` times."""
    var v = ControlBlockView(Int(arg))
    for _ in range(BUMPS):
        _ = v.fetch_add(0, Int64(1))
    return null_opaque()


def _run_bumpers(addr: Int) raises:
    var ts = List[ThreadHandle]()
    for _ in range(THREADS):
        ts.append(ThreadHandle.spawn[_bump](opaque_from_address(addr)))
    for i in range(len(ts)):
        ts[i].join()


def test_roundtrip() raises:
    var blk = ControlBlock(16)
    var v = blk.view()
    var checked = 0
    for i in range(16):
        v.release_store(i, Int64(i * 7 - 3))
        if v.acquire_load(i) != Int64(i * 7 - 3):
            raise Error("release/acquire roundtrip failed at cell " + String(i))
        checked += 1
        v.relaxed_store(i, Int64(-i))
        if v.relaxed_load(i) != Int64(-i):
            raise Error("relaxed roundtrip failed at cell " + String(i))
        checked += 1
    if blk.n_cells() != 16:
        raise Error("n_cells lied: " + String(blk.n_cells()))
    print("  roundtrip:", checked, "of 32 cell reads matched")


def test_fetch_add_across_threads() raises:
    var blk = ControlBlock(16)
    _run_bumpers(blk.addr())
    var got = blk.view().acquire_load(0)
    var want = Int64(THREADS * BUMPS)
    if got != want:
        raise Error(
            "fetch_add lost updates: "
            + String(got)
            + " of an expected "
            + String(want)
            + " from "
            + String(THREADS)
            + " threads"
        )
    print(
        "  fetch_add:",
        THREADS,
        "threads x",
        BUMPS,
        "=",
        got,
        "exactly (0 lost)",
    )


def test_stale_read_after_join() raises:
    """THE MISCOMPILE GATE. Read, read again, with nothing in between.

    On the `calloc` backing both reads see the threads' writes. On a
    `List[Int64]` backing the second read is 0 — the compiler folds it to the
    initializer, because the workers wrote through a pointer built from an
    `Int` (no provenance edge) and `pthread_create` goes out through
    `external_call` (no clobber it believes in).
    """
    var blk = ControlBlock(16)
    _run_bumpers(blk.addr())
    var want = Int64(THREADS * BUMPS)

    # ⚠ NOTHING BETWEEN THESE THREE LINES. See the module docstring.
    var first = blk.view().acquire_load(0)
    var moved = blk^
    var second = moved.view().acquire_load(0)

    if first != want:
        raise Error(
            "first read after join was " + String(first) + ", want "
            + String(want)
        )
    if second != want:
        raise Error(
            "STALE READ: the second straight-line read of the same address"
            " returned " + String(second) + ", want " + String(want)
            + ". The slab is being folded back to its initializer — see the"
            " warning on ControlBlock. Is it still `calloc`ed?"
        )
    print(
        "  stale-read gate: both straight-line reads =", second, "(want", want,
        ")",
    )


def test_line_padding() raises:
    if CELLS_PER_LINE * 8 != 64:
        raise Error(
            "CELLS_PER_LINE describes a "
            + String(CELLS_PER_LINE * 8)
            + "-byte line, expected 64"
        )
    print("  padding: CELLS_PER_LINE =", CELLS_PER_LINE, "= one 64B line")


def main() raises:
    print("=" * 62)
    print("ControlBlock — atomics, threads, and the fold gate")
    print("=" * 62)
    test_roundtrip()
    test_fetch_add_across_threads()
    test_stale_read_after_join()
    test_line_padding()
    print("[PASS] control_block")
