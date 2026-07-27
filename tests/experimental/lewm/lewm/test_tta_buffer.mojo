"""TTAWindowBuffer test (AdaJEPA TTA Phase 1, CPU, host-only).

Encodes frame/action values as `cycle * 100 + env` so ordering and
frame↔action slot alignment are checkable exactly:

  1. not ready before T completed pairs; ready at exactly T,
  2. fill yields the LAST T pairs oldest→newest, frame row t paired with
     the action pushed in the same cycle,
  3. ring wrap: after T+2 cycles the window starts at cycle 2,
  4. done-env freeze: an env that stops pushing keeps its last window,
  5. donor fallback: an env with a partial ring gets the donor's window,
  6. re-pushing a frame before its action overwrites the staged slot
     (render-then-done-at-execute orphan case).
"""

from std.memory import alloc
from std.testing import assert_true, assert_equal

from mojo_rl.nn.constants import DT
from mojo_rl.experimental.lewm.tta_buffer import TTAWindowBuffer


comptime B = 3
comptime T = 3
comptime IMG_DIM = 4
comptime ACT = 2

comptime Buf = TTAWindowBuffer[B, T, IMG_DIM, ACT]


def _frame_val(cyc: Int, b: Int) -> Scalar[DT]:
    return Scalar[DT](cyc * 100 + b)


def _act_val(cyc: Int, b: Int) -> Scalar[DT]:
    return Scalar[DT](cyc * 100 + b) + Scalar[DT](0.5)


def main() raises:
    print("=" * 70)
    print("TTAWindowBuffer test (AdaJEPA TTA Phase 1, CPU)")
    print("=" * 70)

    var buf = Buf()
    var fscratch = alloc[Scalar[DT]](IMG_DIM)
    var ascratch = alloc[Scalar[DT]](ACT)
    var pix = alloc[Scalar[DT]](B * T * IMG_DIM)
    var act = alloc[Scalar[DT]](B * T * ACT)

    # ── 1+2: push T cycles on all envs; env 2 stops after cycle 0 ──────
    comptime N_CYCLES = T + 2  # 5 cycles: 0..4
    for cyc in range(N_CYCLES):
        assert_equal(
            buf.ready(),
            cyc >= T,
            "ready exactly once T pairs are complete",
        )
        for b in range(B):
            if b == 2 and cyc >= 1:
                continue  # env 2 "done" after cycle 0 (partial ring)
            for i in range(IMG_DIM):
                fscratch[i] = _frame_val(cyc, b)
            buf.push_frame(b, fscratch)
            for i in range(ACT):
                ascratch[i] = _act_val(cyc, b)
            buf.push_action(b, ascratch)
    assert_true(buf.ready(), "ready after T+ pairs")

    # ── 3: window = last T cycles (2, 3, 4) oldest→newest, aligned ─────
    assert_true(buf.fill(pix, act), "fill succeeds once ready")
    for b in range(2):  # envs 0, 1 have full rings
        for t in range(T):
            var cyc = N_CYCLES - T + t  # 2, 3, 4
            for i in range(IMG_DIM):
                assert_equal(
                    pix[(b * T + t) * IMG_DIM + i],
                    _frame_val(cyc, b),
                    "frame row t = cycle N-T+t (ring wrap + order)",
                )
            for i in range(ACT):
                assert_equal(
                    act[(b * T + t) * ACT + i],
                    _act_val(cyc, b),
                    "action row t pushed same cycle as frame row t",
                )

    # ── 5: env 2 (1 pair < T) borrows donor env 0's window ─────────────
    for t in range(T):
        var cyc = N_CYCLES - T + t
        assert_equal(
            pix[(2 * T + t) * IMG_DIM],
            _frame_val(cyc, 0),
            "partial-ring env copies the donor (env 0) window",
        )
        assert_equal(
            act[(2 * T + t) * ACT],
            _act_val(cyc, 0),
            "partial-ring env copies donor actions",
        )

    # ── 4: freeze — env 1 stops pushing; env 0 advances one more cycle ──
    for i in range(IMG_DIM):
        fscratch[i] = _frame_val(5, 0)
    buf.push_frame(0, fscratch)
    for i in range(ACT):
        ascratch[i] = _act_val(5, 0)
    buf.push_action(0, ascratch)
    assert_true(buf.fill(pix, act), "fill after uneven advance")
    assert_equal(
        pix[(0 * T + T - 1) * IMG_DIM],
        _frame_val(5, 0),
        "env 0 window advanced to cycle 5",
    )
    assert_equal(
        pix[(1 * T + T - 1) * IMG_DIM],
        _frame_val(4, 1),
        "done env 1 window frozen at its last real pair",
    )

    # ── 6: orphan frame (render pushed, done at execute) is overwritten ─
    for i in range(IMG_DIM):
        fscratch[i] = Scalar[DT](-777.0)
    buf.push_frame(1, fscratch)  # no push_action → pair incomplete
    assert_true(buf.fill(pix, act), "fill ignores incomplete pair")
    assert_equal(
        pix[(1 * T + T - 1) * IMG_DIM],
        _frame_val(4, 1),
        "orphan staged frame must not enter the window",
    )

    # ── empty buffer: fill refuses ──────────────────────────────────────
    var empty = Buf()
    assert_true(not empty.ready(), "fresh buffer not ready")
    assert_true(not empty.fill(pix, act), "fill returns False with no donor")

    fscratch.free()
    ascratch.free()
    pix.free()
    act.free()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
