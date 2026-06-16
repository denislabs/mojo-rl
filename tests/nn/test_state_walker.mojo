"""Test — reflection-walked save/load over Saveable structs.

Phase A.1 smoke. Mirrors `tests/nn/spikes/spike_reflect_checkpoint.mojo`
but exercises the *production* `Saveable` trait + `SaveScalar[DT]` +
`SaveI` + `dump_state` / `load_state` from `mojo_rl.nn.core`.

Four sub-tests:
  1. Flat walk: a single struct with SaveScalar + SaveI + non-Saveable
     fields. Verifies field-by-field line emission and that non-Saveable
     fields are skipped.
  2. Recursive walk: a container holding two nested Saveable structs.
     Verifies dotted prefix accumulation across levels.
  3. Round-trip: dump → re-parse → load into a fresh instance.
     Verifies bit-identical state recovery.
  4. Load error handling: a corrupted line raises an informative error.
"""

from std.testing import assert_equal, assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.saveable import Saveable
from mojo_rl.nn.core.save_scalar import SaveScalar, SaveI
from mojo_rl.nn.core.state_walker import dump_state, load_state


# ──────────────────────────────────────────────────────────────────────
# Test fixtures — module-scope (Mojo nightly disallows function-local
# struct defs).
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct Flat(Copyable, Movable, ImplicitlyDestructible):
    var gamma: SaveScalar[DT]
    var tau: SaveScalar[DT]
    var step: SaveI
    var run_id: String       # Not Saveable — must be skipped


@fieldwise_init
struct AdamLike(Saveable, Copyable):
    var m: SaveScalar[DT]
    var v: SaveScalar[DT]
    var step_count: SaveI

    def save(self, mut out: String, prefix: String) raises:
        dump_state(self, out, prefix)

    def load(
        mut self, lines: List[String], mut idx: Int, prefix: String,
    ) raises:
        load_state(self, lines, idx, prefix)


@fieldwise_init
struct TrainerLike(Copyable, Movable, ImplicitlyDestructible):
    var gamma: SaveScalar[DT]
    var tau: SaveScalar[DT]
    var global_step: SaveI
    var actor_opt: AdamLike
    var critic_opt: AdamLike
    var run_id: String       # Not Saveable — must be skipped


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


def _split_lines(content: String) -> List[String]:
    """Local helper — same logic as checkpoint.mojo's _split_lines."""
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


def _count_newlines(s: String) -> Int:
    var n = 0
    var bytes = s.as_bytes()
    for i in range(len(bytes)):
        if bytes[i] == UInt8(ord("\n")):
            n += 1
    return n


def test_flat_walk() raises:
    print("test_flat_walk ...")
    var s = Flat(
        gamma=SaveScalar[DT](Scalar[DT](0.99)),
        tau=SaveScalar[DT](Scalar[DT](0.005)),
        step=SaveI(1000),
        run_id="run_001",
    )
    var out = String("")
    dump_state(s, out, "")
    print("  --- flat dump ---")
    print(out, end="")
    print("  -----------------")
    # 3 Saveable fields → 3 lines. run_id (String) is silently skipped.
    assert_equal(_count_newlines(out), 3)
    assert_true("gamma=0.99" in out, "expected `gamma=0.99` in dump")
    assert_true("tau=0.005" in out, "expected `tau=0.005` in dump")
    assert_true("step=1000" in out, "expected `step=1000` in dump")
    assert_true("run_id" not in out, "non-Saveable run_id must be skipped")
    print("  ok")


def test_recursive_walk() raises:
    print("test_recursive_walk ...")
    var t = TrainerLike(
        gamma=SaveScalar[DT](Scalar[DT](0.99)),
        tau=SaveScalar[DT](Scalar[DT](0.005)),
        global_step=SaveI(50000),
        actor_opt=AdamLike(
            m=SaveScalar[DT](Scalar[DT](0.1)),
            v=SaveScalar[DT](Scalar[DT](0.01)),
            step_count=SaveI(50000),
        ),
        critic_opt=AdamLike(
            m=SaveScalar[DT](Scalar[DT](-0.05)),
            v=SaveScalar[DT](Scalar[DT](0.02)),
            step_count=SaveI(50000),
        ),
        run_id="trainer_v1",
    )
    var out = String("")
    dump_state(t, out, "")
    print("  --- recursive dump ---")
    print(out, end="")
    print("  ----------------------")
    # 3 top-level Saveables + 3 per nested AdamLike × 2 = 9 lines.
    assert_equal(_count_newlines(out), 9)

    # Top-level scalars (no prefix).
    assert_true("gamma=0.99" in out)
    assert_true("tau=0.005" in out)
    assert_true("global_step=50000" in out)

    # Recursed paths (dotted prefix from container's name).
    assert_true("actor_opt.m=0.1" in out)
    assert_true("actor_opt.v=0.01" in out)
    assert_true("actor_opt.step_count=50000" in out)
    assert_true("critic_opt.m=-0.05" in out)
    assert_true("critic_opt.v=0.02" in out)
    assert_true("critic_opt.step_count=50000" in out)
    assert_true("run_id" not in out)
    print("  ok")


def test_round_trip() raises:
    print("test_round_trip ...")
    var orig = TrainerLike(
        gamma=SaveScalar[DT](Scalar[DT](0.97)),
        tau=SaveScalar[DT](Scalar[DT](0.0125)),
        global_step=SaveI(12345),
        actor_opt=AdamLike(
            m=SaveScalar[DT](Scalar[DT](0.7)),
            v=SaveScalar[DT](Scalar[DT](0.49)),
            step_count=SaveI(12345),
        ),
        critic_opt=AdamLike(
            m=SaveScalar[DT](Scalar[DT](-0.3)),
            v=SaveScalar[DT](Scalar[DT](0.09)),
            step_count=SaveI(12345),
        ),
        run_id="ignored",
    )
    var out = String("")
    dump_state(orig, out, "")

    # Reload into a fresh instance with junk values; load_state must
    # overwrite them.
    var fresh = TrainerLike(
        gamma=SaveScalar[DT](Scalar[DT](0.0)),
        tau=SaveScalar[DT](Scalar[DT](0.0)),
        global_step=SaveI(0),
        actor_opt=AdamLike(
            m=SaveScalar[DT](Scalar[DT](0.0)),
            v=SaveScalar[DT](Scalar[DT](0.0)),
            step_count=SaveI(0),
        ),
        critic_opt=AdamLike(
            m=SaveScalar[DT](Scalar[DT](0.0)),
            v=SaveScalar[DT](Scalar[DT](0.0)),
            step_count=SaveI(0),
        ),
        run_id="fresh",
    )
    var lines = _split_lines(out)
    var idx = 0
    load_state(fresh, lines, idx, "")
    assert_equal(idx, 9, "all 9 Saveable lines must be consumed")

    # Compare every Saveable field bit-identically (no tolerance —
    # text round-trip should be exact for these clean Float32 values).
    assert_true(fresh.gamma.v == orig.gamma.v, "gamma round-trip")
    assert_true(fresh.tau.v == orig.tau.v, "tau round-trip")
    assert_equal(fresh.global_step.v, orig.global_step.v)
    assert_true(fresh.actor_opt.m.v == orig.actor_opt.m.v)
    assert_true(fresh.actor_opt.v.v == orig.actor_opt.v.v)
    assert_equal(fresh.actor_opt.step_count.v, orig.actor_opt.step_count.v)
    assert_true(fresh.critic_opt.m.v == orig.critic_opt.m.v)
    assert_true(fresh.critic_opt.v.v == orig.critic_opt.v.v)
    assert_equal(fresh.critic_opt.step_count.v, orig.critic_opt.step_count.v)
    print("  ok")


def test_corrupted_line_raises() raises:
    print("test_corrupted_line_raises ...")
    # A SaveScalar.load with a missing `=` must raise.
    var fresh = SaveScalar[DT](Scalar[DT](0.0))
    var lines = List[String]()
    lines.append(String("bogus_line_no_equals"))
    var idx = 0
    var raised = False
    try:
        fresh.load(lines, idx, "expected_prefix")
    except e:
        raised = True
    assert_true(raised, "corrupt line must raise")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("test_state_walker — Phase A.1 reflection save/load")
    print("=" * 70)
    test_flat_walk()
    test_recursive_walk()
    test_round_trip()
    test_corrupted_line_raises()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
