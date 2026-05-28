"""Test — RSample RNG state save/load round-trip.

Phase A.3 validation. RSample's `rng_seed` + `_rng_offset` together
determine the next batch of Philox samples. If these are correctly
captured by Saveable, a resumed RSample produces bit-identical samples
to a non-checkpointed equivalent.

Protocol (CPU-mode RSample uses host box_muller via random_float64,
which doesn't read RSample's rng_seed/_rng_offset — those are GPU
state. For a portable CPU-runnable test, we directly mutate the
fields, save, mutate back, load, and compare values bit-identically.
That validates the wire format + load logic. The GPU bit-exact
sampling round-trip belongs in a separate GPU smoke once we have
matching infrastructure).

Sub-tests:
  1. Direct field round-trip: set non-default seed + offset → save →
     wipe → load → fields restored exactly.
  2. Compose with dump_state: nested in a wrapper struct, walked via
     reflection. Path prefix accumulates correctly.
  3. Format mismatch raises: corrupt the saved string, load must error.
"""

from std.testing import assert_equal, assert_true

from mojo_rl.deep_agents2.primitives.rsample import RSample
from mojo_rl.nn2.core.saveable import Saveable
from mojo_rl.nn2.core.state_walker import dump_state, load_state


def _split_lines(content: String) -> List[String]:
    var lines = List[String]()
    var current = String("")
    var bytes = content.as_bytes()
    for i in range(len(bytes)):
        var c = bytes[i]
        if c == UInt8(ord("\n")):
            lines.append(current)
            current = String("")
        else:
            current += chr(Int(c))
    if current.byte_length() > 0:
        lines.append(current)
    return lines^


def test_direct_field_round_trip() raises:
    print("test_direct_field_round_trip ...")
    var r1 = RSample[3]()
    r1.rng_seed = UInt64(424242)
    r1._rng_offset = UInt64(12345)

    var dump = String("")
    r1.save(dump, String("rsample"))
    print("  --- dump ---")
    print(dump, end="")
    print("  ------------")

    # Reset to default + load.
    var r2 = RSample[3]()
    assert_equal(r2.rng_seed, UInt64(42), "default seed before load")
    assert_equal(r2._rng_offset, UInt64(0), "default offset before load")

    var lines = _split_lines(dump)
    var idx = 0
    r2.load(lines, idx, String("rsample"))
    assert_equal(idx, 2, "two lines consumed (seed + offset)")
    assert_equal(r2.rng_seed, UInt64(424242), "seed round-tripped")
    assert_equal(r2._rng_offset, UInt64(12345), "offset round-tripped")
    print("  ok")


@fieldwise_init
struct Wrapper(Saveable):
    """Container that holds an RSample, plus a non-Saveable field.
    Verifies dump_state walks into RSample via reflection."""
    var rs: RSample[3]
    var non_save: Int    # Not Saveable — must be skipped

    def save(self, mut out: String, prefix: String) raises:
        dump_state(self, out, prefix)

    def load(
        mut self, lines: List[String], mut idx: Int, prefix: String,
    ) raises:
        load_state(self, lines, idx, prefix)


def test_dump_state_walks_rsample() raises:
    print("test_dump_state_walks_rsample ...")
    var w1 = Wrapper(rs=RSample[3](), non_save=999)
    w1.rs.rng_seed = UInt64(777)
    w1.rs._rng_offset = UInt64(1234)

    var dump = String("")
    dump_state(w1, dump, String(""))
    print("  --- dump ---")
    print(dump, end="")
    print("  ------------")

    # 2 lines from RSample (seed + offset). non_save (Int) is skipped.
    assert_true("rs.rng_seed=777" in dump, "expected rs.rng_seed=777")
    assert_true("rs.rng_offset=1234" in dump, "expected rs.rng_offset=1234")
    assert_true("non_save" not in dump, "non-Saveable field must be skipped")

    # Round-trip.
    var w2 = Wrapper(rs=RSample[3](), non_save=0)
    var lines = _split_lines(dump)
    var idx = 0
    load_state(w2, lines, idx, String(""))
    assert_equal(w2.rs.rng_seed, UInt64(777))
    assert_equal(w2.rs._rng_offset, UInt64(1234))
    print("  ok")


def test_load_format_mismatch_raises() raises:
    print("test_load_format_mismatch_raises ...")
    var r = RSample[3]()
    var lines = List[String]()
    lines.append(String("rsample.rng_seed=42"))   # ok
    lines.append(String("WRONG_PREFIX=0"))         # bad
    var idx = 0
    var raised = False
    try:
        r.load(lines, idx, String("rsample"))
    except e:
        raised = True
    assert_true(raised, "load with wrong prefix must raise")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("RSample RNG state save/load (Phase A.3)")
    print("=" * 70)
    test_direct_field_round_trip()
    test_dump_state_walks_rsample()
    test_load_format_mismatch_raises()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
