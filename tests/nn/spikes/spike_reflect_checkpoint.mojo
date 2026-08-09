"""Spike — reflection-walked trainer state dump.

Go/no-go for the Phase A checkpoint path: can we walk a trainer-like
struct (mixed fields: scalar hyperparams, optimizer state, RNG state,
step counters, a nested optimizer block) at comptime and serialize
every Saveable field, recursing into nested Saveable containers?

If this works:
  - one generic `save_state[T]` covers every current and future trainer
  - no per-trainer `save_checkpoint` / `load_checkpoint` boilerplate
  - replaces `walkers.mojo::for_each_param_auto` with a broader
    surface that also catches AdamState, PhiloxState, step counters,
    replay cursors

Design (same lesson as the metrics spike):
  Reflection can only call methods that are visible through a `conforms_to`
  narrowing gate. So leaves and containers both implement a `Saveable`
  trait whose `save(out, prefix)` does the actual serialization. Scalar
  leaves are wrapped (`SaveScalar[DType.float32]`, `SaveI`) — mirrors how `Param` wraps
  scalars in walkers.mojo.

Three checks:
  1. Single-level walk: SaveScalar[DType.float32]/SaveI fields produce one line each.
  2. Recursive walk: nested AdamState struct (itself Saveable) walks
     its own fields when its .save is called.
  3. Mixed bundles: non-Saveable fields (raw String run-id) skipped.
"""

from std.reflection import reflect
from std.testing import assert_equal, assert_true


# ---------------------------------------------------------------------
# Saveable — marker trait. Containers + scalar wrappers both conform.
# ---------------------------------------------------------------------
trait Saveable(Copyable, Movable, Deinitable):
    def save(self, mut out: String, prefix: String) raises:
        ...


# ---------------------------------------------------------------------
# Scalar leaf wrappers. One parametric `Scalar[DT]` wrapper (gated by
# `is_floating_point()` per the codebase idiom) covers Float32/Float64,
# and a separate `SaveI` covers integer counters (replay cursor, step,
# etc. — these genuinely need Int, not Scalar[DT]).
# ---------------------------------------------------------------------
@fieldwise_init
struct SaveScalar[DT: DType](Saveable):
    var v: Scalar[Self.DT]

    def save(self, mut out: String, prefix: String) raises:
        comptime assert Self.DT.is_floating_point(), "dtype must be floating point"
        out += prefix + "=" + String(self.v) + "\n"


@fieldwise_init
struct SaveI(Saveable):
    var v: Int

    def save(self, mut out: String, prefix: String) raises:
        out += prefix + "=" + String(self.v) + "\n"


# ---------------------------------------------------------------------
# Nested container — AdamState — itself Saveable, so the parent walker
# calls .save on it without knowing about its internal fields. Recursion
# is achieved by AdamState's own save calling dump_state on itself.
# ---------------------------------------------------------------------
@fieldwise_init
struct AdamState(Saveable):
    var m: SaveScalar[DType.float32]
    var v: SaveScalar[DType.float32]
    var step: SaveI

    def save(self, mut out: String, prefix: String) raises:
        # Internal recursion: delegate to the generic dumper over Self.
        dump_state(self, out, prefix)


# ---------------------------------------------------------------------
# TrainerLike — the kind of struct we want to checkpoint. Mix of:
#   - scalar hyperparams (gamma, tau)
#   - step counter
#   - nested optimizer state (recursive)
#   - non-Saveable field (run_id) that must be skipped
# ---------------------------------------------------------------------
@fieldwise_init
struct TrainerLike(Copyable, Movable, Deinitable):
    var gamma: SaveScalar[DType.float32]
    var tau: SaveScalar[DType.float32]
    var step: SaveI
    var actor_opt: AdamState
    var critic_opt: AdamState
    var run_id: String        # Not Saveable — must be skipped silently


# ---------------------------------------------------------------------
# The one generic dumper. Walks every Saveable field; non-Saveable
# fields are silently skipped (mirrors walkers.mojo's IsParam filter).
# Recursion is explicit: when a Saveable container's `.save` is called,
# it can call dump_state on itself.
# ---------------------------------------------------------------------
def dump_state[T: AnyType](t: T, mut out: String, prefix: String) raises:
    comptime names = reflect[T].field_names()
    comptime types = reflect[T].field_types()
    var sep = "." if prefix.byte_length() > 0 else ""

    comptime for idx in range(reflect[T].field_count()):
        comptime ft = types[idx]
        comptime field_name = names[idx]
        comptime if conforms_to(ft, Saveable):
            ref val = reflect[T].field_ref[idx](t)
            val.save(out, prefix + sep + String(field_name))


@fieldwise_init
struct Flat(Copyable, Movable, Deinitable):
    var gamma: SaveScalar[DType.float32]
    var tau: SaveScalar[DType.float32]
    var step: SaveI
    var run_id: String     # Not Saveable


def test_flat_walk() raises:
    """Single-level walk: SaveScalar[DType.float32]/SaveI fields emit one line each.
    The String field 'run_id' is non-Saveable — must be skipped."""

    var s = Flat(
        gamma=SaveScalar[DType.float32](0.99),
        tau=SaveScalar[DType.float32](0.005),
        step=SaveI(1000),
        run_id="run_001",
    )
    var out = String("")
    dump_state(s, out, "")
    print("--- flat dump ---")
    print(out, end="")
    print("------------------")

    # Three lines (gamma, tau, step) — run_id silently dropped.
    var line_count = 0
    for i in range(out.byte_length()):
        if out[byte=i] == String("\n"):
            line_count += 1
    assert_equal(line_count, 3)

    assert_true("gamma=0.99" in out)
    assert_true("tau=0.005" in out)
    assert_true("step=1000" in out)
    assert_true("run_id" not in out)
    print("  PASS: flat walk skips non-Saveable, emits 3 lines.")


def test_recursive_walk() raises:
    """Trainer with two nested AdamStates. Verifies the
    container.save(out, prefix) → dump_state(self, ...) recursion."""
    var t = TrainerLike(
        gamma=SaveScalar[DType.float32](0.99),
        tau=SaveScalar[DType.float32](0.005),
        step=SaveI(50000),
        actor_opt=AdamState(
            m=SaveScalar[DType.float32](0.1), v=SaveScalar[DType.float32](0.01), step=SaveI(50000)
        ),
        critic_opt=AdamState(
            m=SaveScalar[DType.float32](-0.05), v=SaveScalar[DType.float32](0.02), step=SaveI(50000)
        ),
        run_id="trainer_v1",
    )
    var out = String("")
    dump_state(t, out, "")
    print("--- recursive dump ---")
    print(out, end="")
    print("----------------------")

    # Expected: 3 scalar lines from trainer + 3 from actor_opt + 3 from critic_opt = 9 lines.
    var line_count = 0
    for i in range(out.byte_length()):
        if out[byte=i] == String("\n"):
            line_count += 1
    assert_equal(line_count, 9)

    # Top-level scalars
    assert_true("gamma=0.99" in out)
    assert_true("tau=0.005" in out)
    assert_true("step=50000" in out)
    # Recursed paths
    assert_true("actor_opt.m=0.1" in out)
    assert_true("actor_opt.v=0.01" in out)
    assert_true("actor_opt.step=50000" in out)
    assert_true("critic_opt.m=-0.05" in out)
    assert_true("critic_opt.v=0.02" in out)
    assert_true("critic_opt.step=50000" in out)
    # Non-Saveable field dropped
    assert_true("run_id" not in out)
    print("  PASS: recursive walk emits 9 dotted-path lines.")


def main() raises:
    print("=" * 60)
    print("spike_reflect_checkpoint — Phase A checkpoint reflection spike")
    print("=" * 60)
    test_flat_walk()
    test_recursive_walk()
    print("=" * 60)
    print("ALL PASSED — reflection-walked checkpoint is GO for Phase A.")
    print("=" * 60)
