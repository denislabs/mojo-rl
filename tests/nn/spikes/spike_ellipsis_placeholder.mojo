"""Spike DR.7 — does `...` placeholder in TileTensor type params help?

The user observed MAX kernels using forms like
    `TileTensor[address_space=AddressSpace.GENERIC, ...]`
and hypothesized `...` may act as a wildcard for unspecified params,
opening a path for type-erased TileTensors in Variant / List / fields.

**Result: REJECTED.** Mojo nightly does NOT treat `...` as a type-position
wildcard. The compiler error is decisive:

    `TileTensor[DType.float32, ?, ?, address_space=?, linear_idx_type=?,
     element_size=?] is not concrete, use '[]' to bind missing parameters`

So `TileTensor[DT, ...]` parses but leaves a non-concrete type (with the
remaining 6 params as `?`), which cannot be:
  - used as a variable type annotation
  - used as a `List[T]` element type
  - used as a `Variant[T1, T2]` component type
  - used as a function-parameter type

The full TileTensor parameter list (from the error message) is 7 params:
    `(DType, LayoutType, layout, origin, address_space, linear_idx_type,
     element_size)`
all of which must be bound concretely at any storage / variant / list
position.

The `...` in MAX kernels we saw is likely either:
  - parameter-pack expansion in *argument* contexts (`*Args, ...`)
  - end-of-arg-list ellipsis in *documentation*
…not a type-position wildcard.

**Conclusion**: this avenue is closed. The fixed-arity-2/3 approach from
DR.2 attempt B (each input declares its own `L: TensorLayout, O: MutOrigin`
generics) remains the only viable path for multi-input Modules. Combined
with DR.6's verdict that Variant cannot wrap TileTensors at the type level
(LayoutType vs Layout mismatch), the conclusion is stable.

This file demonstrates the rejection by including the problematic syntax
in commented form so future readers can see what was tried and why it
doesn't compile.
"""

from layout import TileTensor, TensorLayout, row_major

from mojo_rl.nn.constants import DT


def main() raises:
    print("=" * 70)
    print("DR.7 — `...` placeholder in TileTensor type params")
    print("=" * 70)

    var a = List[Scalar[DT]](length=4, fill=Scalar[DT](1.0))
    var ta = TileTensor(a.unsafe_ptr(), row_major[2, 2]())
    print("  Baseline concrete TileTensor: ta[0,0]=", ta[0, 0])

    # The following lines DO NOT COMPILE (commented):
    #
    #   var ta_typed: TileTensor[DT, ...] = ta
    #       → error: 'TileTensor[..., ?, ?, ..., ?, ?]' is not concrete,
    #         use '[]' to bind missing parameters
    #
    #   var lst: List[TileTensor[DT, ...]] = ...
    #       → error: 'List' parameter 'T' has 'Movable' type, but value
    #         has type 'AnyStruct[TileTensor[..., ?, ?, ...]]'
    #
    #   var v: Variant[TileTensor[DT, ...], TileTensor[DT, ...]] = ...
    #       → error: 'Variant' parameter 'Ts' has 'Movable' type, but
    #         value has type 'AnyStruct[TileTensor[..., ?, ?, ...]]'
    #
    #   def f(input: TileTensor[DT, ...]): ...
    #       → same not-concrete error
    #
    # See `spike_ellipsis_placeholder.mojo.disabled` for the originals.

    print("  See file docstring: `...` is not a type-position wildcard.")
    print("  All four probes (variable annotation / List / Variant /")
    print("  function param) reject `TileTensor[DT, ...]` as non-concrete.")
    print("  Fixed-arity-2/3 traits remain the recommended path.")
    print("=" * 70)
