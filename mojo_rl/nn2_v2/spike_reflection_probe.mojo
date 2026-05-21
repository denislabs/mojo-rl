"""Probe Mojo nightly's real reflection API (Follow-up #5).

User pointed to the actual docs:
https://mojolang.org/docs/manual/metaprogramming/reflection/

Key features available:
  - `reflect[T]`         — handle for type T
  - `.field_count()`     — number of fields
  - `.field_names()`     — InlineArray of StaticString names
  - `.field_types()`     — InlineArray of types
  - `.field_ref[idx](v)` — `ref` access to a field by comptime index
  - `.field_offset[index=i]()` — byte offset (alignment-aware)
  - `.name()`, `.base_name()` — fully-qualified / base type name
  - `conforms_to(T, Trait)` — comptime trait conformance check
  - `type_of(x)`, `origin_of(x)` — value-to-type/lifetime
  - `call_location()`, `source_location()` — for assertion error messages
  - `get_function_name[fn]()`, `get_linkage_name[fn]()`

This probe confirms which work in our nightly
(Mojo 1.0.0b2.dev2026052006) and builds toward an auto-derived
`for_each_param` for nn2 leaves.
"""

from std.reflection import reflect


@fieldwise_init
struct Sensor(Movable & ImplicitlyDestructible):
    var id: Int
    var label: String
    var reading: Float64


# ─── Probe 1: field metadata extraction ──────────────────────────────


def probe_field_count_names_types():
    comptime n = reflect[Sensor].field_count()
    comptime names = reflect[Sensor].field_names()
    print("Sensor field_count =", n)
    comptime for i in range(n):
        comptime nm = names[i]
        print("  field[", i, "] name =", String(nm))


# ─── Probe 2: field_ref access by index ──────────────────────────────


def probe_field_ref():
    var s = Sensor(42, String("temp"), 21.5)
    # Mutate via reflection: bump id by 1.
    ref id_ref = reflect[Sensor].field_ref[0](s)
    id_ref += 1
    print("after reflect-mutate: s.id =", s.id)  # 43


# ─── Probe 3: base_name / name ───────────────────────────────────────


def probe_names():
    print("name:     ", reflect[Sensor].name())
    print("base_name:", reflect[Sensor].base_name())


def main() raises:
    probe_field_count_names_types()
    print()
    probe_field_ref()
    print()
    probe_names()
