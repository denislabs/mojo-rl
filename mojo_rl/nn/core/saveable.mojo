"""Saveable — uniform serialize/deserialize surface for reflection-walked
struct checkpointing.

Phase A.1. Mirrors the `IsParam` / `for_each_param_auto` pattern from
`walkers.mojo`: a marker trait whose own method does the serialization,
combined with a generic `dump_state` / `load_state` walker that filters
fields via `conforms_to(ft, Saveable)`.

Conforming types:
  - `SaveScalar[DT]` (floating-point scalar wrapper, gated by
    `Self.DT.is_floating_point()`)
  - `SaveI` (Int counter wrapper)
  - Containers (e.g. `AdamState`, `SACConfig`) that delegate to
    `dump_state(self, ...)` / `load_state(self, ...)` from their own
    `save` / `load` methods (recursion is explicit, not reflection-magic)

Why a trait method rather than a `Float64(field)` cast in the walker:
  `Float64` does NOT conform to `Floatable`, so the walker can't
  generically construct a Float64 from an opaque `AnyType` field.
  Marker-trait dispatch sidesteps this. See feedback memo
  `feedback-mojo-float64-not-floatable.md`.

Line format used by SaveScalar / SaveI / Param.save:
  <dotted.path>=<value>\n

Containers do NOT emit a header line by default — they pass-through
their children's lines transparently with a prefix-accumulated dotted
path (e.g. `actor_opt.m=0.5`). This keeps the format flat and
line-oriented, matching v1's existing line-based parser.
"""


trait Saveable(Movable, ImplicitlyDestructible):
    """Container or scalar-leaf that can serialize itself to a String
    body and read itself back. The walkers (`state_walker.mojo`) gate on
    `conforms_to(ft, Saveable)` to find conforming fields.

    Parents are `Movable & ImplicitlyDestructible` only (matching the
    `Module` trait). Copyable is NOT required because real conformers
    like `Param` carry `List[Scalar[DT]]` storage which is not
    `ImplicitlyCopyable`. The walker doesn't need to copy values — it
    walks refs and calls trait methods in place.

    Required methods:
      - `save(self, mut out, prefix)` — append serialized lines to `out`.
      - `load(mut self, lines, mut idx, prefix)` — consume one or more
        lines from `lines[idx:]`, advancing `idx` past the consumed
        lines. Raises if a line's prefix doesn't match the expected
        path (catches format drift early).

    Container recursion pattern:
        struct AdamState(Saveable):
            var m: SaveScalar[DT]
            var v: SaveScalar[DT]
            var step_count: SaveI

            def save(self, mut out, prefix) raises:
                dump_state(self, out, prefix)
            def load(mut self, lines, mut idx, prefix) raises:
                load_state(self, lines, idx, prefix)
    """

    def save(self, mut out: String, prefix: String) raises:
        ...

    def load(
        mut self, lines: List[String], mut idx: Int, prefix: String,
    ) raises:
        ...
