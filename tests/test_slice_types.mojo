"""Variadic.slice_types on parametric variadics — comptime assert provides evidence.

Key finding: `comptime assert Variadic.size(ops) >= end_value` provides
evidence for slice_types constraint when using literal end values.

Run: cd mojo-rl && pixi run mojo run tests/test_slice_types.mojo
"""

from mojo_rl.nn.autodiff import (
    DiffOp,
    OpID,
    AutoDiffChain,
    MatMul,
    BiasAdd,
    ReLUOp,
    TanhOp,
    FusedMatMulBias,
    FusedMatMulBiasReLU,
    FusedMatMulBiasTanh,
)
from std.builtin.variadics import Variadic


# =============================================================================
# Test 1: concrete (baseline)
# =============================================================================
fn test_concrete():
    print("=== Test 1: Concrete variadic ===")
    comptime ops = Variadic.types[
        T=DiffOp,
        MatMul[2, 4],
        BiasAdd[4],
        ReLUOp[4],
        MatMul[4, 1],
        BiasAdd[1],
    ]
    comptime tail = Variadic.slice_types[element_types=ops, start=3, end=5]
    comptime head = Variadic.slice_types[element_types=ops, start=0, end=3]
    print("  head N =", Variadic.size(head), "tail N =", Variadic.size(tail))
    print("  PASS")


# =============================================================================
# Test 2: parametric with comptime assert — literal end
# The assert must prove end <= size(ops). For end=5: assert size >= 5
# For end=3: assert size >= 3 (NOT >= 5!)
#
# NOTE: Transitive inequality reasoning does not work.
# `comptime assert size(ops) >= 5` does NOT imply `size(ops) >= 3`,
# even though 5 >= 3 is trivially true. Each distinct end value used
# in slice_types requires its own explicit assert. This means:
#   assert size >= 5  →  proves end=5 OK, but NOT end=3
#   assert size >= 3  →  proves end=3 OK
# Both are needed for two slices with end=5 and end=3.
# =============================================================================
fn test_parametric_two_slices[*OPS: DiffOp]():
    print("=== Test 2: Parametric two slices ===")
    comptime ops = Variadic.types[T=DiffOp, *OPS]
    # Need separate asserts for each end value — no transitive reasoning
    comptime assert Variadic.size(ops) >= 5, "need >= 5"
    comptime assert Variadic.size(ops) >= 3, "need >= 3"
    comptime tail = Variadic.slice_types[element_types=ops, start=3, end=5]
    comptime head = Variadic.slice_types[element_types=ops, start=0, end=3]
    print("  head N =", Variadic.size(head), "tail N =", Variadic.size(tail))
    print("  PASS")


# =============================================================================
# Test 3: parametric — Variadic.size(ops) as end
# This is the tricky one: can assert help prove size(ops) <= size(ops)?
# =============================================================================
fn test_parametric_size_end[*OPS: DiffOp]():
    print("=== Test 3: Parametric, size(ops) as end ===")
    comptime ops = Variadic.types[T=DiffOp, *OPS]
    comptime assert Variadic.size(ops) >= 3, "need >= 3"
    # Try: assert size(ops) <= size(ops) — tautology
    comptime assert Variadic.size(ops) <= Variadic.size(ops), "trivial"
    comptime tail = Variadic.slice_types[
        element_types=ops, start=3, end=Variadic.size(ops)
    ]
    print("  tail N =", Variadic.size(tail))
    print("  PASS")


# =============================================================================
# Test 4: slice + unpack into AutoDiffChain (parametric)
# =============================================================================
fn test_slice_unpack[*OPS: DiffOp]():
    print("=== Test 4: Parametric slice -> AutoDiffChain ===")
    comptime ops = Variadic.types[T=DiffOp, *OPS]
    comptime assert Variadic.size(ops) >= 2, "need >= 2"
    comptime sliced = Variadic.slice_types[element_types=ops, start=0, end=2]
    comptime Chain = AutoDiffChain[*sliced]
    print("  IN_DIM =", Chain.IN_DIM, "OUT_DIM =", Chain.OUT_DIM)
    print("  PASS")


# =============================================================================
# Test 5: RECURSIVE greedy fusion — the breakthrough!
# Uses slice_types + comptime assert to recursively consume op groups.
# =============================================================================
fn greedy_fuse[*OPS: DiffOp]():
    comptime ops = Variadic.types[T=DiffOp, *OPS]
    comptime N = Variadic.size(ops)

    comptime if N == 0:
        pass
    elif N == 2:
        comptime assert Variadic.size(ops) >= 2
        comptime if (
            ops[0].OP_ID == OpID.MATMUL._value
            and ops[1].OP_ID == OpID.BIAS_ADD._value
        ):
            print("    -> FusedMB[", ops[0].IN_DIM, ",", ops[0].OUT_DIM, "]")
    elif N >= 3:
        comptime assert Variadic.size(ops) >= 3
        comptime assert Variadic.size(ops) <= Variadic.size(ops)
        comptime if (
            ops[0].OP_ID == OpID.MATMUL._value
            and ops[1].OP_ID == OpID.BIAS_ADD._value
        ):
            comptime if ops[2].OP_ID == OpID.RELU._value:
                print(
                    "    -> FusedMBR[", ops[0].IN_DIM, ",", ops[0].OUT_DIM, "]"
                )
                comptime if N > 3:
                    comptime rest = Variadic.slice_types[
                        element_types=ops, start=3, end=Variadic.size(ops)
                    ]
                    greedy_fuse[*rest]()
            elif ops[2].OP_ID == OpID.TANH._value:
                print(
                    "    -> FusedMBT[", ops[0].IN_DIM, ",", ops[0].OUT_DIM, "]"
                )
                comptime if N > 3:
                    comptime rest = Variadic.slice_types[
                        element_types=ops, start=3, end=Variadic.size(ops)
                    ]
                    greedy_fuse[*rest]()
            elif ops[2].OP_ID == OpID.MATMUL._value:
                # M+B (no activation), next group starts at index 2
                print(
                    "    -> FusedMB[", ops[0].IN_DIM, ",", ops[0].OUT_DIM, "]"
                )
                comptime assert Variadic.size(ops) >= 2
                comptime rest = Variadic.slice_types[
                    element_types=ops, start=2, end=Variadic.size(ops)
                ]
                greedy_fuse[*rest]()


fn main():
    print()

    test_concrete()
    print()

    test_parametric_two_slices[
        MatMul[2, 4], BiasAdd[4], ReLUOp[4], MatMul[4, 1], BiasAdd[1]
    ]()
    print()

    test_parametric_size_end[
        MatMul[2, 4], BiasAdd[4], ReLUOp[4], MatMul[4, 1], BiasAdd[1]
    ]()
    print()

    test_slice_unpack[FusedMatMulBiasReLU[2, 4], FusedMatMulBias[4, 1]]()
    print()

    print("=== Test 5: Recursive greedy fusion ===")
    print("  Case: M+B (2 ops)")
    greedy_fuse[MatMul[3, 5], BiasAdd[5]]()
    print("  Case: M+B+R (3 ops)")
    greedy_fuse[MatMul[3, 5], BiasAdd[5], ReLUOp[5]]()
    print("  Case: M+B+R + M+B (5 ops)")
    greedy_fuse[
        MatMul[2, 4],
        BiasAdd[4],
        ReLUOp[4],
        MatMul[4, 1],
        BiasAdd[1],
    ]()
    print("  Case: M+B+R + M+B+T + M+B (8 ops)")
    greedy_fuse[
        MatMul[2, 8],
        BiasAdd[8],
        ReLUOp[8],
        MatMul[8, 4],
        BiasAdd[4],
        TanhOp[4],
        MatMul[4, 1],
        BiasAdd[1],
    ]()
    print("  Case: M+B+R x3 + M+B (11 ops)")
    greedy_fuse[
        MatMul[3, 64],
        BiasAdd[64],
        ReLUOp[64],
        MatMul[64, 32],
        BiasAdd[32],
        ReLUOp[32],
        MatMul[32, 16],
        BiasAdd[16],
        ReLUOp[16],
        MatMul[16, 2],
        BiasAdd[2],
    ]()
    print()

    print("ALL DONE")
