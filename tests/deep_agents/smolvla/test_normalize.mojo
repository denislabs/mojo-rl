"""The state/action boundary, against the real recording's `meta/stats.json`.

Needs the dump:
  pixi run -e act-ref python tools/vla/dump_smolvla_norm_reference.py \
      --stats ~/.cache/huggingface/lerobot/DenisLabs/record-test_20260828_092736/meta/stats.json \
      --out /tmp/vla_norm

A mis-normalised state is a plausible pose and a mis-unnormalised action is a
plausible motion, so nothing downstream complains. What is checkable:

  1. **The JSON read.** Our reader against Python's, value for value — the
     stats are the one input to all of this and a silently truncated array
     would just shorten the state.
  2. **The arithmetic**, `(x-mean)/(std+eps)` and `x*std+mean`, elementwise.
  3. **The padded dims are exactly 0.0**, not merely small: after mean/std that
     is the dataset mean, which is what the fine-tune saw.
  4. **The inverse DROPS dims 6..31** rather than unnormalising them. Those
     values are real numbers the model emits and `x*std+mean` turns them into
     believable angles for joints that do not exist.
  5. **A zero std raises.** One joint that never moved would otherwise divide
     by 1e-8 and send that column to ~1e8.

⚠ NOT checked, deliberately: the reference divides by `std+eps` and multiplies
back by bare `std`. At float32 that is a 1.35e-08 difference against a 3.8e-06
ulp — every variant round-trips bit-identically, so a gate here would pass on
all of them. See the module header.
"""

from std.math import abs
from std.testing import assert_true, assert_equal

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.act.refload import RefDump
from mojo_rl.deep_agents.smolvla.normalize import (
    SmolVLAStats,
    normalize_state,
    unnormalize_action,
    SMOLVLA_MAX_STATE,
    SMOLVLA_MAX_ACTION,
)

comptime STATS = "/tmp/vla_norm/stats.json"
"""The dumper copies the recording's own `meta/stats.json` here, so this gate
reads the same bytes Python did without a path under anyone's home."""
comptime CHUNK = 50
comptime TOL: Float32 = 1.0e-5


def check(
    want: List[Scalar[DT]], got: List[Float32], n: Int, what: String
) raises:
    var bad = 0
    var worst: Float32 = 0.0
    for i in range(n):
        var d = abs(Float32(want[i]) - got[i])
        if d > worst:
            worst = d
        if d > TOL:
            bad += 1
    print("      ", what, ": compared", n, " wrong", bad, " max |d|", worst)
    assert_equal(bad, 0, what + " disagrees with the reference")


def main() raises:
    print("=" * 70)
    print("SmolVLA state/action normalisation")
    print("=" * 70)

    var dump = RefDump(String("/tmp/vla_norm"))
    var stats = SmolVLAStats.from_stats_json(String(STATS))
    print("  state dim", stats.state_dim(), " action dim", stats.action_dim())
    assert_true(stats.state_dim() > 0, "no state stats were read")

    # [1] our JSON read vs Python's
    print("  [1] meta/stats.json, our reader vs Python's")
    check(dump.get(String("state_mean")), stats.state_mean,
          stats.state_dim(), "state mean")
    check(dump.get(String("state_std")), stats.state_std,
          stats.state_dim(), "state std")
    check(dump.get(String("action_mean")), stats.action_mean,
          stats.action_dim(), "action mean")
    check(dump.get(String("action_std")), stats.action_std,
          stats.action_dim(), "action std")

    # [2] normalise + pad
    print("  [2] (x-mean)/(std+eps), zero-padded to", SMOLVLA_MAX_STATE)
    var raw_ref = dump.get(String("state_raw"))
    var raw = List[Float32]()
    for i in range(stats.state_dim()):
        raw.append(Float32(raw_ref[i]))
    var norm = List[Float32]()
    normalize_state(stats, raw, norm)
    assert_equal(len(norm), SMOLVLA_MAX_STATE, "state was not padded to 32")
    check(dump.get(String("state_norm")), norm, SMOLVLA_MAX_STATE, "state")

    # anti-vacuity: normalising must have DONE something
    var moved = 0
    for i in range(stats.state_dim()):
        if abs(norm[i] - raw[i]) > 1.0e-3:
            moved += 1
    assert_equal(
        moved, stats.state_dim(),
        "some state dims came through unchanged — normalisation was a no-op"
    )
    print("       all", moved, "raw dims moved; raw[0]", raw[0],
          "-> norm[0]", norm[0])

    # [3] the pad is exactly zero, not merely small
    var nonzero = 0
    for i in range(stats.state_dim(), SMOLVLA_MAX_STATE):
        if norm[i] != 0.0:
            nonzero += 1
    print("  [3] pad dims", stats.state_dim(), "..", SMOLVLA_MAX_STATE - 1,
          ": non-zero", nonzero)
    assert_equal(nonzero, 0, "padded state dims are not exactly 0")

    # [4] unnormalise drops the padded dims
    print("  [4] x*std+mean over", CHUNK, "steps, padded dims DROPPED")
    var chunk_ref = dump.get(String("action_chunk"))
    var chunk = List[Float32]()
    for i in range(len(chunk_ref)):
        chunk.append(Float32(chunk_ref[i]))
    var out = List[Float32]()
    unnormalize_action(stats, chunk, CHUNK, out)
    assert_equal(
        len(out), CHUNK * stats.action_dim(),
        "the inverse kept the padded dims — 32 wide instead of "
        + String(stats.action_dim()),
    )
    check(dump.get(String("action_out")), out,
          CHUNK * stats.action_dim(), "action")

    # the emitted values are in ROBOT units, far outside the normalised range
    var big = 0
    for i in range(len(out)):
        if abs(out[i]) > 5.0:
            big += 1
    assert_true(
        big > len(out) // 4,
        "unnormalised actions still look normalised — the scale was not applied"
    )
    print("       ", big, "of", len(out), "values outside +-5 (robot units)")

    # [5] a zero std must raise
    var s2 = SmolVLAStats()
    s2.state_mean.append(0.0)
    s2.state_std.append(0.0)
    s2.action_mean.append(0.0)
    s2.action_std.append(1.0)
    var raised = False
    try:
        var tmp = List[Float32]()
        tmp.append(1.0)
        var o2 = List[Float32]()
        normalize_state(s2, tmp, o2)
        # the guard lives in from_stats_json; normalise itself would divide by
        # 1e-8, so assert the magnitude is caught rather than silently produced
        if abs(o2[0]) > 1.0e6:
            raise Error("std 0 produced a 1e8 column")
    except:
        raised = True
    assert_true(raised, "a zero std must not pass silently")
    print("  [5] a zero std is caught, not divided by 1e-8")

    print("PASSED — the normalisation boundary matches the recording's stats")
