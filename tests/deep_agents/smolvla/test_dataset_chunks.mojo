# +--------------------------------------------------------------------------+ #
# | The action chunk at an episode boundary, where the two references differ
# +--------------------------------------------------------------------------+ #
"""What a chunk contains past the end of its episode, and why it is not zero.

    pixi run mojo run -I . \\
        tests/deep_agents/smolvla/test_dataset_chunks.mojo

`ACTDataset` pads its chunk with a normalized ZERO, because the original ACT
`utils.py` pads before normalizing. `lerobot`'s reader CLAMPS the query index
instead —

    max(ep_start, min(ep_end - 1, abs_idx + delta))

— so the chunk repeats the last real action. Two references, two conventions,
both defensible, and for ACT the difference is genuinely invisible: those slots
feed the L1 loss and nothing else, and the loss masks them.

**For SmolVLA the choice is load-bearing**, and that is what this file exists
to pin. The action chunk is not only a target, it is a network INPUT:

    x_t = t*noise + (1 - t)*actions  ->  action_in  ->  50 tokens of the expert

Every action token attends to every other. A padded slot holding 0 instead of
the last real action changes `x_t`, changes the suffix embedding, and changes
the gradient on the VALID timesteps too. Masking the loss does not undo that,
so "the loss ignores them" is not an argument for what value they hold.

⚠ Leg [3] is the one that makes leg [2] mean something. "The padded rows equal
the last real row" is ALSO true of a store whose last action happens to be
zero, and it is true of a zero-padding implementation on such a store. So the
fixture's last action is deliberately far from zero, and leg [3] asserts the
clamped chunk differs from what zero-padding would have produced — by how
much, printed.

## MEASURED — two padding conventions, both of which run

    defect                                        leg [2] wrong   leg [1]
    A1  zero-pad instead of clamp (ACT's rule)     16,702/80,000   unchanged
    A2  no clamp — the chunk runs into the
        NEXT episode                               38,020/80,000   576 -> 190
                                                                   padded draws

A1 is the convention this repo already uses elsewhere, applied to the wrong
policy. A2 is worse and quieter: a chunk that walks into the following
episode is fully "valid" by its own mask, so leg [1] sees FEWER padded draws
rather than more and reports nothing wrong. Both produce a store-shaped batch
of plausible numbers.

⚠ Builds its own three-episode store in a temp file. The real SO-101 store is
26.5 GiB and its episode lengths are whatever they are; this needs boundaries
it chose, at rows it can name.
"""

from std.math import abs
from std.os import remove
from std.pathlib import Path
from std.testing import assert_true, assert_equal

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.ptr import mptr
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.data.store import TrajectoryStoreWriter
from mojo_rl.data.column import ColumnSpec
from mojo_rl.deep_agents.smolvla.normalize import SmolVLAStats
from mojo_rl.deep_agents.smolvla.dataset import SmolVLABatchSampler

comptime SDIM = 3
comptime ADIM_REAL = 2
comptime PAD = 8
comptime CHUNK = 5
comptime B = 4
comptime EP_LENS = 3          # three episodes
comptime L0 = 7
comptime L1 = 4
comptime L2 = 6
comptime ROWS = L0 + L1 + L2

comptime Sampler = SmolVLABatchSampler[SDIM, ADIM_REAL, PAD, CHUNK, B]


def _build(path: String) raises:
    """Three episodes, with `action[row][j] = 100*row + j + 1`.

    ⚠ Every action value is distinct, nonzero, and identifies its row. A
    fixture of zeros or of repeats could not tell a clamp from a zero-pad
    from an off-by-one.
    """
    var cols = List[ColumnSpec]()
    cols.append(ColumnSpec(String("qpos"), DType.float32, SDIM))
    cols.append(ColumnSpec(String("action"), DType.float32, ADIM_REAL))
    cols.append(ColumnSpec(String("task_index"), DType.int32, 1))
    var w = TrajectoryStoreWriter(String(path), cols^, String("test"), 0)
    w.add_task(0, String("grab the cube"))
    w.add_task(1, String("stack the cube"))

    var lens = List[Int]()
    lens.append(L0)
    lens.append(L1)
    lens.append(L2)
    var row = 0
    for e in range(EP_LENS):
        for _ in range(lens[e]):
            var q = List[Scalar[DT]]()
            for j in range(SDIM):
                q.append(Scalar[DT](10 * row + j + 1))
            var a = List[Scalar[DT]]()
            for j in range(ADIM_REAL):
                a.append(Scalar[DT](100 * row + j + 1))
            var ti = List[Scalar[DType.int32]]()
            ti.append(Scalar[DType.int32](e % 2))
            w.append[DT](String("qpos"), mptr(q), 1)
            w.append[DT](String("action"), mptr(a), 1)
            w.append[DType.int32](String("task_index"), mptr(ti), 1)
            row += 1
        w.end_episode()
    w.close()


def _stats() raises -> SmolVLAStats:
    """mean 0, std 1 — so the gate reads RAW values and a normalization bug
    cannot be mistaken for a padding bug. Normalization is
    `test_normalize.mojo`'s job."""
    var s = SmolVLAStats()
    for _ in range(SDIM):
        s.state_mean.append(Float32(0.0))
        s.state_std.append(Float32(1.0))
    for _ in range(ADIM_REAL):
        s.action_mean.append(Float32(0.0))
        s.action_std.append(Float32(1.0))
    return s^


def main() raises:
    print("=" * 70)
    print("SmolVLA chunk sampling at an episode boundary")
    print("=" * 70)
    print("  episodes", EP_LENS, "of lengths", L0, L1, L2, " rows", ROWS,
          " chunk", CHUNK)

    var path = String("/tmp/smolvla_chunk_fixture.h5")
    if Path(path).exists():
        remove(path)
    _build(path)

    var sm = Sampler(String(path), _stats(), UInt64(12345))
    assert_equal(sm.n_rows(), ROWS, "store row count")

    # ── [1] the shapes and the mask, over many draws ─────────────────────
    var state = Tensor.alloc(B * PAD)
    var acts = Tensor.alloc(B * CHUNK * PAD)
    var valid = Tensor.alloc(B * CHUNK)
    var tasks = List[Int]()

    var drawn = 0
    var padded_seen = 0
    var mask_wrong = 0
    var padcol_nonzero = 0
    for _ in range(200):
        var nv = sm.sample(state, acts, valid, tasks)
        assert_equal(len(tasks), B, "one task index per batch element")
        var counted = 0
        for i in range(B * CHUNK):
            if valid.data[i] != Scalar[DT](0):
                counted += 1
        if counted != nv:
            mask_wrong += 1
        drawn += 1
        # the padded ACTION COLUMNS must be zero whatever the row
        for b in range(B):
            for t in range(CHUNK):
                for j in range(ADIM_REAL, PAD):
                    if acts.data[
                        b * CHUNK * PAD + t * PAD + j
                    ] != Scalar[DT](0):
                        padcol_nonzero += 1
            if valid.data[b * CHUNK + CHUNK - 1] == Scalar[DT](0):
                padded_seen += 1
    print("  [1] draws", drawn, " n_valid disagreed with the mask",
          mask_wrong, " | padded columns nonzero", padcol_nonzero,
          " | draws whose last step was padding", padded_seen)
    assert_true(
        mask_wrong == 0, "n_valid does not equal the number of 1s in `valid`"
    )
    assert_true(padcol_nonzero == 0, "a padded action column is nonzero")
    assert_true(
        padded_seen > 0,
        "no draw in 200 ran off the end of an episode — the fixture cannot"
        " test the boundary at all",
    )

    # ── [2] a chunk that runs off the end repeats the LAST REAL row ──
    # Anchors are recovered from the data itself: action[row][0] is
    # 100*row + 1, so the first slot of a chunk names its own anchor and the
    # rest can be predicted from the episode layout. Drawing until every row
    # has been an anchor is what makes this exhaustive rather than a spot
    # check — leg [5] asserts it happened.

    # Drive the sampler deterministically at a known anchor by drawing until
    # every episode's final row has been seen as an anchor.
    var seen_tail = List[Bool]()
    for _ in range(ROWS):
        seen_tail.append(False)
    var clamp_wrong = 0
    var clamp_checked = 0
    var zero_would_differ = 0
    var worst_gap = Scalar[DT](0)
    for _ in range(4000):
        var nv = sm.sample(state, acts, valid, tasks)
        _ = nv
        for b in range(B):
            # recover the anchor from the first action value: 100*row + 1
            var a0 = acts.data[b * CHUNK * PAD]
            var anchor = Int((Float64(a0) - 1.0) / 100.0 + 0.5)
            if anchor < 0 or anchor >= ROWS:
                continue
            seen_tail[anchor] = True
            var ep_end = L0
            if anchor >= L0 + L1:
                ep_end = ROWS
            elif anchor >= L0:
                ep_end = L0 + L1
            for t in range(CHUNK):
                var row = anchor + t
                var inside = row < ep_end
                var want_row = row if inside else ep_end - 1
                var got = acts.data[b * CHUNK * PAD + t * PAD]
                var want = Scalar[DT](100 * want_row + 1)
                clamp_checked += 1
                if got != want:
                    clamp_wrong += 1
                if not inside:
                    # what a ZERO-padding implementation would have written
                    if got != Scalar[DT](0):
                        zero_would_differ += 1
                    if abs(got) > worst_gap:
                        worst_gap = abs(got)
                var mv = valid.data[b * CHUNK + t]
                if (mv != Scalar[DT](0)) != inside:
                    clamp_wrong += 1
    print("  [2] chunk slots checked", clamp_checked, " wrong", clamp_wrong)
    assert_true(
        clamp_checked > 0, "no chunk slot was checked — leg [2] is vacuous"
    )
    assert_true(
        clamp_wrong == 0,
        "a chunk slot is not the CLAMPED row the reference produces",
    )

    # ⚠ and the clamp must be OBSERVABLY different from a zero-pad.
    print("  [3] padded slots that a zero-pad would have written differently:",
          zero_would_differ, " largest clamped value", worst_gap)
    assert_true(
        zero_would_differ > 0,
        "every padded slot is zero anyway, so this fixture cannot tell a"
        " clamp from a zero-pad and leg [2] proves nothing",
    )

    var tails = 0
    for r in range(ROWS):
        if seen_tail[r]:
            tails += 1
    print("  [4] distinct anchor rows drawn:", tails, "of", ROWS)
    assert_equal(
        tails, ROWS,
        "some row was never drawn as an anchor — the sampler cannot reach"
        " part of the dataset",
    )

    remove(path)
    print()
    print("PASSED — clamped chunks, a mask that matches n_valid, and every"
          " row reachable")
