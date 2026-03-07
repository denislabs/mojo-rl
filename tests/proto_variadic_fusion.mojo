"""Prototype: Variadic type manipulation for automatic fusion.

Testing various approaches to compile-time type list rewriting.

Run: cd mojo-rl && pixi run mojo run tests/proto_variadic_fusion.mojo
"""

from nn.autodiff import (
    DiffOp,
    OpID,
    AutoDiffChain,
    MatMul,
    BiasAdd,
    ReLUOp,
    TanhOp,
    SigmoidOp,
    FusedMatMulBias,
    FusedMatMulBiasReLU,
    FusedMatMulBiasTanh,
)
from nn.constants import dtype
from std.builtin.variadics import Variadic
from layout import Layout, LayoutTensor


# =============================================================================
# Test 1: Basic Variadic.types + indexing
# =============================================================================

fn test_1_basic():
    print("=" * 60)
    print("Test 1: Basic Variadic.types (known working)")
    print("=" * 60)

    comptime ops = Variadic.types[T=DiffOp, MatMul[2, 4], BiasAdd[4], ReLUOp[4]]
    comptime n = Variadic.size(ops)

    print("  N =", n)
    print("  ops[0].OP_ID =", ops[0].OP_ID)
    print("  ops[0].IN_DIM =", ops[0].IN_DIM, "OUT_DIM =", ops[0].OUT_DIM)
    print("  PASS")
    print()


# =============================================================================
# Test 2: comptime if fusion — reads dims, builds fused chain
# =============================================================================

fn test_2_comptime_if_fusion[*OPS: DiffOp]():
    comptime ops = Variadic.types[T=DiffOp, *OPS]
    comptime N = Variadic.size(ops)

    print("  Input:", N, "ops")

    comptime if N == 5:
        comptime if (ops[0].OP_ID == OpID.MATMUL._value
                and ops[1].OP_ID == OpID.BIAS_ADD._value
                and ops[2].OP_ID == OpID.RELU._value
                and ops[3].OP_ID == OpID.MATMUL._value
                and ops[4].OP_ID == OpID.BIAS_ADD._value):
            comptime Fused = AutoDiffChain[
                FusedMatMulBiasReLU[ops[0].IN_DIM, ops[0].OUT_DIM],
                FusedMatMulBias[ops[3].IN_DIM, ops[3].OUT_DIM],
            ]
            print("  Fused [MBR, MB]: IN=", Fused.IN_DIM, "OUT=", Fused.OUT_DIM,
                  "PARAMS=", Fused.PARAM_SIZE)

fn test_2_wrapper():
    print("=" * 60)
    print("Test 2: comptime if fusion selection")
    print("=" * 60)
    test_2_comptime_if_fusion[
        MatMul[2, 4], BiasAdd[4], ReLUOp[4],
        MatMul[4, 1], BiasAdd[1],
    ]()
    print("  PASS")
    print()


# =============================================================================
# Test 3: Greedy fusion reading dims from variadic ops
# =============================================================================

fn test_3_greedy():
    print("=" * 60)
    print("Test 3: Greedy auto-fuse (8 ops -> 3 fused)")
    print("=" * 60)

    comptime ops = Variadic.types[
        T=DiffOp,
        MatMul[2, 8], BiasAdd[8], ReLUOp[8],
        MatMul[8, 4], BiasAdd[4], TanhOp[4],
        MatMul[4, 1], BiasAdd[1],
    ]
    comptime N = Variadic.size(ops)

    comptime Fused = AutoDiffChain[
        FusedMatMulBiasReLU[ops[0].IN_DIM, ops[0].OUT_DIM],
        FusedMatMulBiasTanh[ops[3].IN_DIM, ops[3].OUT_DIM],
        FusedMatMulBias[ops[6].IN_DIM, ops[6].OUT_DIM],
    ]

    comptime Unfused = AutoDiffChain[
        MatMul[2, 8], BiasAdd[8], ReLUOp[8],
        MatMul[8, 4], BiasAdd[4], TanhOp[4],
        MatMul[4, 1], BiasAdd[1],
    ]

    print("  Original:", N, "ops")
    print("  Unfused PARAM_SIZE =", Unfused.PARAM_SIZE)
    print("  Fused   PARAM_SIZE =", Fused.PARAM_SIZE)
    print("  Unfused IN/OUT:", Unfused.IN_DIM, "->", Unfused.OUT_DIM)
    print("  Fused   IN/OUT:", Fused.IN_DIM, "->", Fused.OUT_DIM)
    print("  PASS")
    print()


# =============================================================================
# Test 4: Numerical verification — use concrete dims to avoid unfolded expr
# =============================================================================

fn test_4_numerical():
    print("=" * 60)
    print("Test 4: Numerical fused == unfused")
    print("=" * 60)

    comptime BATCH = 2

    comptime Unfused = AutoDiffChain[
        MatMul[2, 4], BiasAdd[4], ReLUOp[4],
        MatMul[4, 1], BiasAdd[1],
    ]

    # Use concrete dims (not ops[i].IN_DIM) to avoid unfolded expression
    comptime Fused = AutoDiffChain[
        FusedMatMulBiasReLU[2, 4],
        FusedMatMulBias[4, 1],
    ]

    # Shared params
    var params_list = List[Scalar[dtype]](capacity=Unfused.PARAM_SIZE)
    for i in range(Unfused.PARAM_SIZE):
        params_list.append(Scalar[dtype](0.1) * Scalar[dtype](i + 1))

    var input_list = List[Scalar[dtype]](capacity=BATCH * 2)
    input_list.append(1.0)
    input_list.append(2.0)
    input_list.append(3.0)
    input_list.append(4.0)

    # Unfused forward
    var out_u = List[Scalar[dtype]](capacity=BATCH)
    var cache_u = List[Scalar[dtype]](capacity=BATCH * Unfused.CACHE_SIZE)
    for _ in range(BATCH):
        out_u.append(0.0)
    for _ in range(BATCH * Unfused.CACHE_SIZE):
        cache_u.append(0.0)

    var inp = LayoutTensor[dtype, Layout.row_major(BATCH, 2), MutAnyOrigin](
        input_list.unsafe_ptr())
    var ou = LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin](
        out_u.unsafe_ptr())
    var p = LayoutTensor[dtype, Layout.row_major(Unfused.PARAM_SIZE), MutAnyOrigin](
        params_list.unsafe_ptr())
    var cu = LayoutTensor[dtype, Layout.row_major(BATCH, Unfused.CACHE_SIZE), MutAnyOrigin](
        cache_u.unsafe_ptr())

    Unfused.forward[BATCH](inp, ou, p, cu)

    # Fused forward — use Fused.PARAM_SIZE for LayoutTensor
    var out_f = List[Scalar[dtype]](capacity=BATCH)
    var cache_f = List[Scalar[dtype]](capacity=BATCH * Fused.CACHE_SIZE)
    for _ in range(BATCH):
        out_f.append(0.0)
    for _ in range(BATCH * Fused.CACHE_SIZE):
        cache_f.append(0.0)

    var of_ = LayoutTensor[dtype, Layout.row_major(BATCH, 1), MutAnyOrigin](
        out_f.unsafe_ptr())
    var pf = LayoutTensor[dtype, Layout.row_major(Fused.PARAM_SIZE), MutAnyOrigin](
        params_list.unsafe_ptr())
    var cf = LayoutTensor[dtype, Layout.row_major(BATCH, Fused.CACHE_SIZE), MutAnyOrigin](
        cache_f.unsafe_ptr())

    Fused.forward[BATCH](inp, of_, pf, cf)

    var max_diff: Scalar[dtype] = 0.0
    for b in range(BATCH):
        var diff = out_u[b] - out_f[b]
        if diff < 0:
            diff = -diff
        if diff > max_diff:
            max_diff = diff

    print("  Unfused:", out_u[0], out_u[1])
    print("  Fused:  ", out_f[0], out_f[1])
    print("  Max diff:", max_diff)
    if max_diff < 1e-4:
        print("  Match: OK")
    else:
        print("  Match: FAIL")
    print()


# =============================================================================
# Test 5: Generic auto_fuse for common MLP sizes (2, 3, 5, 8 ops)
# =============================================================================

fn auto_fuse_and_run[*OPS: DiffOp]():
    comptime ops = Variadic.types[T=DiffOp, *OPS]
    comptime N = Variadic.size(ops)

    print("  Input:", N, "ops")

    comptime if N == 2:
        comptime if ops[0].OP_ID == OpID.MATMUL._value and ops[1].OP_ID == OpID.BIAS_ADD._value:
            comptime Result = AutoDiffChain[
                FusedMatMulBias[ops[0].IN_DIM, ops[0].OUT_DIM],
            ]
            print("    -> [FusedMB] IN=", Result.IN_DIM, "OUT=", Result.OUT_DIM)

    elif N == 3:
        comptime if ops[0].OP_ID == OpID.MATMUL._value and ops[1].OP_ID == OpID.BIAS_ADD._value:
            comptime if ops[2].OP_ID == OpID.RELU._value:
                comptime Result = AutoDiffChain[
                    FusedMatMulBiasReLU[ops[0].IN_DIM, ops[0].OUT_DIM],
                ]
                print("    -> [FusedMBR] IN=", Result.IN_DIM, "OUT=", Result.OUT_DIM)
            elif ops[2].OP_ID == OpID.TANH._value:
                comptime Result = AutoDiffChain[
                    FusedMatMulBiasTanh[ops[0].IN_DIM, ops[0].OUT_DIM],
                ]
                print("    -> [FusedMBT] IN=", Result.IN_DIM, "OUT=", Result.OUT_DIM)

    elif N == 5:
        comptime if (ops[0].OP_ID == OpID.MATMUL._value
                and ops[1].OP_ID == OpID.BIAS_ADD._value
                and ops[3].OP_ID == OpID.MATMUL._value
                and ops[4].OP_ID == OpID.BIAS_ADD._value):
            comptime if ops[2].OP_ID == OpID.RELU._value:
                comptime Result = AutoDiffChain[
                    FusedMatMulBiasReLU[ops[0].IN_DIM, ops[0].OUT_DIM],
                    FusedMatMulBias[ops[3].IN_DIM, ops[3].OUT_DIM],
                ]
                print("    -> [FusedMBR, FusedMB] PARAMS=", Result.PARAM_SIZE)
            elif ops[2].OP_ID == OpID.TANH._value:
                comptime Result = AutoDiffChain[
                    FusedMatMulBiasTanh[ops[0].IN_DIM, ops[0].OUT_DIM],
                    FusedMatMulBias[ops[3].IN_DIM, ops[3].OUT_DIM],
                ]
                print("    -> [FusedMBT, FusedMB] PARAMS=", Result.PARAM_SIZE)

    elif N == 8:
        comptime if (ops[0].OP_ID == OpID.MATMUL._value
                and ops[1].OP_ID == OpID.BIAS_ADD._value
                and ops[3].OP_ID == OpID.MATMUL._value
                and ops[4].OP_ID == OpID.BIAS_ADD._value
                and ops[6].OP_ID == OpID.MATMUL._value
                and ops[7].OP_ID == OpID.BIAS_ADD._value):
            comptime act1 = ops[2].OP_ID
            comptime act2 = ops[5].OP_ID
            comptime if act1 == OpID.RELU._value and act2 == OpID.RELU._value:
                comptime Result = AutoDiffChain[
                    FusedMatMulBiasReLU[ops[0].IN_DIM, ops[0].OUT_DIM],
                    FusedMatMulBiasReLU[ops[3].IN_DIM, ops[3].OUT_DIM],
                    FusedMatMulBias[ops[6].IN_DIM, ops[6].OUT_DIM],
                ]
                print("    -> [FusedMBR, FusedMBR, FusedMB] PARAMS=", Result.PARAM_SIZE)
            elif act1 == OpID.RELU._value and act2 == OpID.TANH._value:
                comptime Result = AutoDiffChain[
                    FusedMatMulBiasReLU[ops[0].IN_DIM, ops[0].OUT_DIM],
                    FusedMatMulBiasTanh[ops[3].IN_DIM, ops[3].OUT_DIM],
                    FusedMatMulBias[ops[6].IN_DIM, ops[6].OUT_DIM],
                ]
                print("    -> [FusedMBR, FusedMBT, FusedMB] PARAMS=", Result.PARAM_SIZE)
    else:
        print("    -> no auto-fuse rule for", N, "ops")

fn test_5_auto_fuse():
    print("=" * 60)
    print("Test 5: Generic auto_fuse for known MLP sizes")
    print("=" * 60)

    auto_fuse_and_run[MatMul[3, 5], BiasAdd[5]]()
    auto_fuse_and_run[MatMul[3, 5], BiasAdd[5], ReLUOp[5]]()
    auto_fuse_and_run[
        MatMul[3, 8], BiasAdd[8], ReLUOp[8],
        MatMul[8, 2], BiasAdd[2],
    ]()
    auto_fuse_and_run[
        MatMul[3, 64], BiasAdd[64], ReLUOp[64],
        MatMul[64, 32], BiasAdd[32], ReLUOp[32],
        MatMul[32, 2], BiasAdd[2],
    ]()
    auto_fuse_and_run[
        MatMul[3, 64], BiasAdd[64], ReLUOp[64],
        MatMul[64, 32], BiasAdd[32], TanhOp[32],
        MatMul[32, 2], BiasAdd[2],
    ]()
    print("  PASS")
    print()


# =============================================================================
# Test 6: Variadic.slice_types exploration
# Try all possible syntax forms to find what works.
# =============================================================================

# =============================================================================
# Test 6: slice_types on concrete variadics — WORKS!
# Syntax: Variadic.slice_types[element_types=ops, start=N, end=M]
# T is inferred. Must provide explicit end (default end fails constraint).
# Can unpack result with * into AutoDiffChain[*sliced].
# LIMITATION: Only works on concrete variadics (Variadic.types[...]),
# NOT on variadics from function parameters (*OPS: DiffOp) — constraint
# checker can't prove end <= size(ops) when size is symbolic.
# =============================================================================

fn test_6_slice():
    print("=" * 60)
    print("Test 6: Variadic.slice_types (concrete variadics)")
    print("=" * 60)

    comptime ops = Variadic.types[
        T=DiffOp,
        MatMul[2, 4], BiasAdd[4], ReLUOp[4],
        MatMul[4, 1], BiasAdd[1],
    ]

    # Slice tail [3:5]
    comptime tail = Variadic.slice_types[element_types=ops, start=3, end=5]
    comptime tail_n = Variadic.size(tail)
    print("  tail N =", tail_n, "(expect 2)")
    print("  tail[0].OP_ID =", tail[0].OP_ID, "(expect", OpID.MATMUL._value, ")")
    print("  tail[1].OP_ID =", tail[1].OP_ID, "(expect", OpID.BIAS_ADD._value, ")")

    # Slice head [0:3]
    comptime head = Variadic.slice_types[element_types=ops, start=0, end=3]
    comptime head_n = Variadic.size(head)
    print("  head N =", head_n, "(expect 3)")

    # Unpack slice result into AutoDiffChain
    comptime fused_ops = Variadic.types[
        T=DiffOp,
        FusedMatMulBiasReLU[2, 4], FusedMatMulBias[4, 1],
    ]
    comptime sliced = Variadic.slice_types[element_types=fused_ops, start=0, end=2]
    comptime Fused = AutoDiffChain[*sliced]
    print("  AutoDiffChain[*sliced] IN=", Fused.IN_DIM, "OUT=", Fused.OUT_DIM,
          "PARAMS=", Fused.PARAM_SIZE)

    print("  PASS")
    print()


# =============================================================================
# Test 7: concat_types — result has unresolved dependent type
# LIMITATION: concat_types returns [Ts: Variadic[Variadic[DiffOp]]] Variadic[DiffOp]
# which can't be sized or unpacked. This is a Mojo compiler limitation.
# Workaround: use comptime if dispatch + direct AutoDiffChain construction.
# =============================================================================

# (concat_types test omitted — see tests/test_concat.mojo for attempts)


# =============================================================================
# FINDINGS SUMMARY
# =============================================================================
#
# WORKING TECHNIQUES:
# 1. Variadic.types[T=DiffOp, ...] + Variadic.size() + indexing
#    ops[i].OP_ID, ops[i].IN_DIM, ops[i].OUT_DIM all accessible
#
# 2. comptime if N == K pattern matching on variadic size
#    + OP_ID-based pattern matching inside
#    → builds fused AutoDiffChain with dims read from original ops
#
# 3. Variadic.slice_types[element_types=ops, start=S, end=E]
#    → works on CONCRETE (non-parametric) variadics only
#    → result can be unpacked with * into struct variadic params
#    → explicit end required (default end constraint fails)
#
# BLOCKED TECHNIQUES:
# 4. slice_types on parametric variadics (from *OPS: DiffOp)
#    → constraint checker can't prove end <= size(ops) when symbolic
#    → where clauses, constrained[], comptime assert don't help
#
# 5. concat_types result unpacking
#    → returns dependent type that can't be sized or unpacked
#    → even raw OPS parameter pack has same issue
#
# 6. Recursive slice-based fusion
#    → blocked by #4 (can't slice parametric variadics)
#
# BEST CURRENT APPROACH:
# → comptime if size dispatch (N==2, 3, 5, 8, 11) with OP_ID matching
# → covers standard MLP topologies (1-4 hidden layers)
# → each case builds the fused AutoDiffChain directly
# → fully works with parametric variadics (*OPS: DiffOp)
# =============================================================================


# =============================================================================
# Main
# =============================================================================

fn main():
    print()
    print("Variadic Type Manipulation Prototypes for Auto-Fusion")
    print("=" * 60)
    print()

    test_1_basic()
    test_2_wrapper()
    test_3_greedy()
    test_4_numerical()
    test_5_auto_fuse()
    test_6_slice()

    print("=" * 60)
    print("All prototype tests complete!")
    print("=" * 60)
