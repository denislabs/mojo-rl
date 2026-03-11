"""Minimal reproduction: Variadic.concat_types result is unusable.

concat_types returns a dependent type:
    [Ts: Variadic[Variadic[DiffOp]]] Variadic[DiffOp]

This type cannot be:
  - sized with Variadic.size()
  - indexed with [0], [1]
  - unpacked with * into struct variadic params (AutoDiffChain[*result])

For comparison, slice_types returns a resolved Variadic[DiffOp] that
CAN be sized, indexed, and unpacked — inconsistent behavior.

Run: cd mojo-rl && pixi run mojo run tests/test_concat_types.mojo

To reproduce errors: uncomment ONE failing test function at a time.
"""

from mojo_rl.nn.autodiff import (
    DiffOp,
    OpID,
    AutoDiffChain,
    FusedMatMulBias,
    FusedMatMulBiasReLU,
)
from std.builtin.variadics import Variadic


# =============================================================================
# WORKS: slice_types result can be sized, indexed, and unpacked
# =============================================================================
fn test_slice_works():
    print("=== slice_types (WORKS) ===")
    comptime ops = Variadic.types[
        T=DiffOp,
        FusedMatMulBiasReLU[2, 4],
        FusedMatMulBias[4, 1],
    ]
    comptime sliced = Variadic.slice_types[element_types=ops, start=0, end=2]
    comptime n = Variadic.size(sliced)  # OK
    print("  size =", n)
    print("  sliced[0].OP_ID =", sliced[0].OP_ID)  # OK
    comptime Fused = AutoDiffChain[*sliced]  # OK
    print("  AutoDiffChain IN=", Fused.IN_DIM, "OUT=", Fused.OUT_DIM)
    print("  PASS")


# =============================================================================
# FAIL 1: Variadic.size() on concat result
# Error: no matching function in call to 'size'
#   value cannot be converted from
#   '[Ts: Variadic[Variadic[DiffOp]]] Variadic[DiffOp]'
#   to 'Variadic[T]', it depends on an unresolved parameter 'T'
# =============================================================================
# fn test_concat_size():
#     comptime a = Variadic.types[T=DiffOp, FusedMatMulBiasReLU[2, 4]]
#     comptime b = Variadic.types[T=DiffOp, FusedMatMulBias[4, 1]]
#     comptime ab = Variadic.concat_types[DiffOp, a, b]
#     comptime n = Variadic.size(ab)  # ERROR


# =============================================================================
# FAIL 2: indexing concat result
# Error: parameter 'Ts' has 'Variadic[DiffOp]' type,
#        but value has type 'IntLiteral[0]'
# =============================================================================
# fn test_concat_index():
#     comptime a = Variadic.types[T=DiffOp, FusedMatMulBiasReLU[2, 4]]
#     comptime b = Variadic.types[T=DiffOp, FusedMatMulBias[4, 1]]
#     comptime ab = Variadic.concat_types[DiffOp, a, b]
#     print("  ab[0].OP_ID =", ab[0].OP_ID)  # ERROR


# =============================================================================
# FAIL 3: unpacking concat result with * into struct variadic param
# Error: only variadics can be unpacked
# =============================================================================
# fn test_concat_unpack():
#     comptime a = Variadic.types[T=DiffOp, FusedMatMulBiasReLU[2, 4]]
#     comptime b = Variadic.types[T=DiffOp, FusedMatMulBias[4, 1]]
#     comptime ab = Variadic.concat_types[DiffOp, a, b]
#     comptime Fused = AutoDiffChain[*ab]  # ERROR


fn main():
    test_slice_works()
    print()
    print("DONE")
