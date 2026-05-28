"""Reflection-walked Logger emit.

Phase A.5. Walks every `Metric`-conforming field of a bundle, emits one
`log_scalar(name, value, step)` per field. Non-Metric fields silently
skipped (lets bundles carry non-metric state like step counters
without it leaking into the logger).

Verified by `tests/nn2/spikes/spike_reflect_metrics.mojo` (4/4 GREEN
2026-05-23) — same shape as `dump_state` / `load_state` for Saveable.

Zero-overhead short-circuit: `Logger` trait carries a comptime
`ENABLED: Bool = True` (False on `NoOpLogger`). The `comptime if not
L.ENABLED: return` at the entry of `log_bundle` makes the entire walk
disappear at compile time when L is NoOpLogger — Mojo emits no field
reads, no name allocs, no method calls.
"""

from std.reflection import reflect

from mojo_rl.core.logger import Logger
from .metric import Metric


def log_bundle[T: AnyType, L: Logger](
    mut logger: L, ref m: T, step: Int,
) raises:
    """Walk every Metric-conforming field of `m`; emit one log_scalar
    per field. Non-Metric fields skipped silently.

    Zero overhead when L = NoOpLogger (comptime-elided)."""
    comptime if not L.ENABLED:
        return
    comptime names = reflect[T].field_names()
    comptime types = reflect[T].field_types()
    comptime for idx in range(reflect[T].field_count()):
        comptime ft = types[idx]
        comptime field_name = names[idx]
        comptime if conforms_to(ft, Metric):
            ref val = reflect[T].field_ref[idx](m)
            logger.log_scalar(String(field_name), val.to_f64(), step)
