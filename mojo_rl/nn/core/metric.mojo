"""Metric — marker trait for fields that can be reduced to Float64 for
emission to a Logger backend.

Phase A.5. Mirrors `Saveable` from A.1: Float64 doesn't conform to
`Floatable` (see `feedback-mojo-float64-not-floatable.md`), so we use
a marker trait whose own `to_f64()` method does the cast itself.
Codebase convention: one parametric `LogScalar[DT]` wrapper gated by
`Self.DT.is_floating_point()`.

A `MetricsBundle` is a plain `@fieldwise_init` struct whose fields are
either `LogScalar[DT]` (or any other `Metric` conformer) — those get
walked + logged — or non-Metric (silently skipped by the walker).
This lets bundles carry incidental fields like step counters without
them leaking into the Logger backend.

Concrete bundles live next to their trainers (`sac_metrics.mojo`,
`ddpg_metrics.mojo`, `td3_metrics.mojo`) — they're trainer-specific.
"""


trait Metric(Copyable, Movable, ImplicitlyDestructible):
    """A value that can be reduced to Float64 for emission via
    `Logger.log_scalar`. Conforms must provide `to_f64(self)`."""

    def to_f64(self) -> Float64:
        ...


# ──────────────────────────────────────────────────────────────────────
# Parametric scalar wrapper. Use this for every metric field whose
# storage is `Scalar[DT]`. Gate uses the codebase's
# `comptime assert dtype.is_floating_point()` idiom — see
# `mojo_rl/nn/autodiff/fused/activation.mojo:120`.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct LogScalar[DT: DType](Metric):
    var v: Scalar[Self.DT]

    def to_f64(self) -> Float64:
        comptime assert Self.DT.is_floating_point(), "dtype must be floating point"
        return Float64(self.v)
