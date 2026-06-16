"""Spike — reflection-walked MetricsBundle → Logger.log_scalar.

Go/no-go for the Phase A logging path: can a `@fieldwise_init` bundle
of metric fields be walked at comptime and forwarded as
`(field_name, value, step)` triples to a Logger backend?

If this works:
  - per-algorithm metrics live as one struct (one field per metric)
  - adding a metric = adding a field
  - one generic `log_bundle` covers every algo (SAC/DDPG/TD3/...)
  - no per-agent `_train_diagnostics` boilerplate

Design (after probing what Float64 conforms to):
  Float64 does NOT conform to `Floatable` (the trait is for things that
  convert TO a float; Float64 already is one), and we can't extend
  built-in numeric types with new trait conformances. So we follow
  walkers.mojo: define a tiny marker trait whose method does the
  cast itself.

  Codebase idiom (per user guidance): one parametric `LogScalar[DT]`
  wrapper, gated by `comptime assert DT.is_floating_point()`. Mirrors
  how the rest of nn handles `Scalar[DT]`. Bundle author writes
  `var critic_loss: LogScalar[DT]` (DT comes from nn.constants).
"""

from std.reflection import reflect
from std.testing import assert_equal, assert_true


# ---------------------------------------------------------------------
# Logger-shaped trait — local, doesn't depend on the production Logger
# ---------------------------------------------------------------------
trait MiniLogger(Copyable, Movable, ImplicitlyDeletable):
    def log_scalar(mut self, name: String, value: Float64, step: Int) raises:
        ...


struct ListLogger(MiniLogger):
    var names: List[String]
    var values: List[Float64]
    var steps: List[Int]

    def __init__(out self):
        self.names = List[String]()
        self.values = List[Float64]()
        self.steps = List[Int]()

    def __init__(out self, *, copy: Self):
        self.names = copy.names.copy()
        self.values = copy.values.copy()
        self.steps = copy.steps.copy()

    def log_scalar(mut self, name: String, value: Float64, step: Int) raises:
        self.names.append(name)
        self.values.append(value)
        self.steps.append(step)


# ---------------------------------------------------------------------
# Metric — marker trait whose method does the Float64 cast.
# Walker gates on conforms_to(ft, Metric) and calls .to_f64().
# ---------------------------------------------------------------------
trait Metric(Copyable, Movable, ImplicitlyDeletable):
    def to_f64(self) -> Float64:
        ...


# ---------------------------------------------------------------------
# Parametric `Scalar[DT]` wrapper — replaces hand-rolled LogF64/LogF32.
# Compile-time gated to floating-point DT, matches the rest of nn.
# ---------------------------------------------------------------------
@fieldwise_init
struct LogScalar[DT: DType](Metric):
    var v: Scalar[Self.DT]

    def to_f64(self) -> Float64:
        comptime assert Self.DT.is_floating_point(), "dtype must be floating point"
        return Float64(self.v)


# ---------------------------------------------------------------------
# Realistic SAC metrics bundle. Five Metric fields, all Float32-backed
# (matches the project's default DT = DType.float32).
# ---------------------------------------------------------------------
comptime DT_F32 = DType.float32

@fieldwise_init
struct SACMetrics(Copyable, Movable, ImplicitlyDeletable):
    var critic_loss: LogScalar[DT_F32]
    var actor_loss: LogScalar[DT_F32]
    var alpha: LogScalar[DT_F32]
    var q_mean: LogScalar[DT_F32]
    var entropy: LogScalar[DT_F32]


# ---------------------------------------------------------------------
# The one generic logging helper. If this compiles + runs correctly,
# we have the Phase A logger ergonomics for free.
# ---------------------------------------------------------------------
def log_bundle[
    T: AnyType,
    L: MiniLogger,
](mut logger: L, m: T, step: Int) raises:
    """Walk every Metric-conforming field of `m`, emit one log_scalar
    per field. Non-Metric fields are skipped silently."""
    comptime names = reflect[T].field_names()
    comptime types = reflect[T].field_types()
    comptime for idx in range(reflect[T].field_count()):
        comptime ft = types[idx]
        comptime field_name = names[idx]
        comptime if conforms_to(ft, Metric):
            ref val = reflect[T].field_ref[idx](m)
            logger.log_scalar(String(field_name), val.to_f64(), step)


def test_metrics_bundle_walk() raises:
    var m = SACMetrics(
        critic_loss=LogScalar[DT_F32](Scalar[DT_F32](0.42)),
        actor_loss=LogScalar[DT_F32](Scalar[DT_F32](-1.7)),
        alpha=LogScalar[DT_F32](Scalar[DT_F32](0.20)),
        q_mean=LogScalar[DT_F32](Scalar[DT_F32](-12.5)),
        entropy=LogScalar[DT_F32](Scalar[DT_F32](1.34)),
    )
    var logger = ListLogger()
    log_bundle(logger, m, 1000)

    assert_equal(len(logger.names), 5)
    assert_equal(len(logger.values), 5)
    assert_equal(len(logger.steps), 5)

    assert_equal(logger.names[0], String("critic_loss"))
    assert_equal(logger.names[1], String("actor_loss"))
    assert_equal(logger.names[2], String("alpha"))
    assert_equal(logger.names[3], String("q_mean"))
    assert_equal(logger.names[4], String("entropy"))

    # F32 round-trip is ~1e-6 — loosen the tolerance accordingly.
    assert_true((logger.values[0] - 0.42).__abs__() < 1e-6)
    assert_true((logger.values[1] - (-1.7)).__abs__() < 1e-6)
    assert_true((logger.values[2] - 0.20).__abs__() < 1e-6)
    assert_true((logger.values[3] - (-12.5)).__abs__() < 1e-6)
    assert_true((logger.values[4] - 1.34).__abs__() < 1e-6)

    for i in range(5):
        assert_equal(logger.steps[i], 1000)

    print("  PASS: SACMetrics walked, 5 log_scalar calls issued.")
    for i in range(len(logger.names)):
        print(
            "    log_scalar(",
            logger.names[i],
            ", ",
            String(logger.values[i])[byte=:8],
            ", step=",
            logger.steps[i],
            ")",
            sep="",
        )


# ---------------------------------------------------------------------
# Second check: empty bundle is fine.
# ---------------------------------------------------------------------
@fieldwise_init
struct EmptyMetrics(Copyable, Movable, ImplicitlyDeletable):
    pass


def test_empty_bundle() raises:
    var m = EmptyMetrics()
    var logger = ListLogger()
    log_bundle(logger, m, 0)
    assert_equal(len(logger.names), 0)
    print("  PASS: empty bundle issues 0 log calls.")


# ---------------------------------------------------------------------
# Third check: heterogeneous DT bundle — Float32 + Float64.
# Verifies one parametric LogScalar covers multiple dtypes.
# ---------------------------------------------------------------------
@fieldwise_init
struct MixedMetrics(Copyable, Movable, ImplicitlyDeletable):
    var loss_f32: LogScalar[DType.float32]
    var lr_f64:   LogScalar[DType.float64]


def test_mixed_bundle() raises:
    var m = MixedMetrics(
        loss_f32=LogScalar[DType.float32](Scalar[DType.float32](0.5)),
        lr_f64=LogScalar[DType.float64](Scalar[DType.float64](1e-3)),
    )
    var logger = ListLogger()
    log_bundle(logger, m, 42)

    assert_equal(len(logger.names), 2)
    assert_equal(logger.names[0], String("loss_f32"))
    assert_equal(logger.names[1], String("lr_f64"))
    assert_true((logger.values[0] - 0.5).__abs__() < 1e-6)
    assert_true((logger.values[1] - 1e-3).__abs__() < 1e-9)
    print("  PASS: parametric LogScalar[DT] handles F32 + F64 in one walk.")


# ---------------------------------------------------------------------
# Fourth check: non-Metric fields are silently skipped.
# ---------------------------------------------------------------------
@fieldwise_init
struct MetricsWithExtras(Copyable, Movable, ImplicitlyDeletable):
    var critic_loss: LogScalar[DT_F32]
    var raw_step_counter: Int         # Not Metric — must be skipped
    var actor_loss: LogScalar[DT_F32]
    var run_id: String                 # Not Metric — must be skipped


def test_skip_non_metric() raises:
    var m = MetricsWithExtras(
        critic_loss=LogScalar[DT_F32](Scalar[DT_F32](0.1)),
        raw_step_counter=999,
        actor_loss=LogScalar[DT_F32](Scalar[DT_F32](0.2)),
        run_id="run_001",
    )
    var logger = ListLogger()
    log_bundle(logger, m, 7)
    assert_equal(len(logger.names), 2)
    assert_equal(logger.names[0], String("critic_loss"))
    assert_equal(logger.names[1], String("actor_loss"))
    print("  PASS: non-Metric fields (Int, String) silently skipped.")


def main() raises:
    print("=" * 60)
    print("spike_reflect_metrics — Phase A logger reflection spike")
    print("=" * 60)
    test_metrics_bundle_walk()
    test_empty_bundle()
    test_mixed_bundle()
    test_skip_non_metric()
    print("=" * 60)
    print("ALL PASSED — reflection-walked MetricsBundle is GO for Phase A.")
    print("=" * 60)
