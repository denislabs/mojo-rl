"""Reflection-walked save/load over any `@fieldwise_init` struct.

Phase A.1. Mirrors `walkers.mojo::for_each_param_auto`'s pattern: gate
on `conforms_to(ft, Saveable)`, call the trait method. Recursion is
explicit — the container's own `save` / `load` calls `dump_state` /
`load_state` on `self`.

Usage from a container:

    struct AdamState(Saveable):
        var m: SaveScalar[DT]
        var v: SaveScalar[DT]
        var step_count: SaveI

        def save(self, mut out: String, prefix: String) raises:
            dump_state(self, out, prefix)

        def load(mut self, lines, mut idx, prefix) raises:
            load_state(self, lines, idx, prefix)

The walker accumulates a dotted prefix automatically: at the top level
(prefix=""), fields produce `field_name=value`; one level down,
`field_name.sub_name=value`. Non-Saveable fields are silently skipped
(matches the IsParam filter's semantics — open-by-default).

Why explicit recursion: Mojo nightly's reflection cannot generically
dispatch a recursive call `dump_state(val, ...)` where `val` is a
generic ref through a `conforms_to(_, Saveable)` gate. The container
calls the walker on itself, which IS a concrete type — Mojo can
specialize. Spike-verified in
`tests/nn2/spikes/spike_reflect_checkpoint.mojo`.
"""

from std.reflection import reflect

from .saveable import Saveable


def dump_state[T: AnyType](
    ref t: T, mut out: String, prefix: String,
) raises:
    """Walk every `Saveable`-conforming field of `t` and append its
    serialized lines to `out`. Prefix accumulates dotted-path naming
    across levels.

    `ref t: T` is a read-only reference: doesn't require Copyable, and
    accepts both mut and non-mut callers (`Saveable.save(self, ...)`
    invokes it on `self` which is non-mut)."""
    comptime names = reflect[T].field_names()
    comptime types = reflect[T].field_types()
    var sep = "." if prefix.byte_length() > 0 else ""
    comptime for idx in range(reflect[T].field_count()):
        comptime ft = types[idx]
        comptime field_name = names[idx]
        comptime if conforms_to(ft, Saveable):
            ref val = reflect[T].field_ref[idx](t)
            val.save(out, prefix + sep + String(field_name))


def load_state[T: AnyType](
    mut t: T, lines: List[String], mut idx: Int, prefix: String,
) raises:
    """Walk every `Saveable`-conforming field of `t` and consume one or
    more lines from `lines[idx:]`, advancing `idx`. Field walk order
    must match the order at save time (guaranteed by reflection over a
    fixed struct definition)."""
    comptime names = reflect[T].field_names()
    comptime types = reflect[T].field_types()
    var sep = "." if prefix.byte_length() > 0 else ""
    comptime for f_idx in range(reflect[T].field_count()):
        comptime ft = types[f_idx]
        comptime field_name = names[f_idx]
        comptime if conforms_to(ft, Saveable):
            ref val = reflect[T].field_ref[f_idx](t)
            val.load(lines, idx, prefix + sep + String(field_name))
