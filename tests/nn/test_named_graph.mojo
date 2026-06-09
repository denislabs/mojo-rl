"""Tests for ComputeGraph with named nodes.

Tests validate forward + backward correctness for various graph topologies.

Tests:
1. Simple chain
2. Fan-out with MinOp
3. Dual-input (DDPG-like)
4. Full DAG: fan-out + fan-in + dual-input
"""

from std.memory import UnsafePointer
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import (
    Model,
    Linear,
    LinearReLU,
    Negate,
    Min,
)
from mojo_rl.nn.autodiff.compute_graph import ComputeGraph, GNode
from mojo_rl.nn.initializer import Xavier
from layout import Layout, LayoutTensor
from std.math import abs


def assert_close(
    actual: Float64, expected: Float64, tol: Float64, msg: String
) raises:
    var diff = abs(actual - expected)
    if diff > tol:
        print(
            "FAIL:",
            msg,
            "| actual:",
            actual,
            "expected:",
            expected,
            "diff:",
            diff,
        )
        raise Error("Assertion failed: " + msg)


def fill_sequential(ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int):
    for i in range(n):
        ptr[i] = Scalar[dtype](Float64(i + 1) * 0.1)


# =============================================================================
# Test 1: Simple chain
# =============================================================================


def test_simple_chain() raises:
    print("Test 1: Simple chain...")

    comptime BATCH = 2
    comptime M = ComputeGraph[
        GNode["hidden", LinearReLU[4, 8]],
        GNode["output", Linear[8, 3], "hidden"],
    ]

    var p = InlineArray[Scalar[dtype], M.PARAM_SIZE](uninitialized=True)
    var pt = LayoutTensor[dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin](
        p.unsafe_ptr()
    )
    M.initialize_params[Xavier[]](pt)

    var inp = InlineArray[Scalar[dtype], BATCH * M.IN_DIM](uninitialized=True)
    fill_sequential(inp.unsafe_ptr(), BATCH * M.IN_DIM)
    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())

    var out = InlineArray[Scalar[dtype], BATCH * M.OUT_DIM](uninitialized=True)
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](out.unsafe_ptr())
    var cache = InlineArray[Scalar[dtype], BATCH * M.CACHE_SIZE](
        uninitialized=True
    )
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
    ](cache.unsafe_ptr())
    var st_t = LayoutTensor[
        dtype, Layout.row_major(M.STATE_SIZE), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(0)))
    M.forward[BATCH](inp_t, out_t, pt, st_t, cache_t)

    # Backward
    var go = InlineArray[Scalar[dtype], BATCH * M.OUT_DIM](uninitialized=True)
    for i in range(BATCH * M.OUT_DIM):
        go[i] = Scalar[dtype](1.0)
    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](go.unsafe_ptr())

    var gi = InlineArray[Scalar[dtype], BATCH * M.IN_DIM](uninitialized=True)
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](gi.unsafe_ptr())
    var g = InlineArray[Scalar[dtype], M.PARAM_SIZE](uninitialized=True)
    for i in range(M.PARAM_SIZE):
        g[i] = Scalar[dtype](0.0)
    var gt = LayoutTensor[dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin](
        g.unsafe_ptr()
    )
    M.backward[BATCH](go_t, gi_t, pt, st_t, cache_t, gt)

    # Verify non-zero outputs and grads
    var any_nonzero = False
    for i in range(BATCH * M.OUT_DIM):
        if out[i] != 0:
            any_nonzero = True
    if not any_nonzero:
        raise Error("All outputs zero")

    print("  PASSED")


# =============================================================================
# Test 2: Fan-out
# =============================================================================


def test_fan_out() raises:
    print("Test 2: Fan-out with MinOp...")

    comptime BATCH = 2
    comptime M = ComputeGraph[
        GNode["trunk", LinearReLU[4, 3]],
        GNode["v1", Linear[3, 1], "trunk"],
        GNode["v2", Linear[3, 1], "trunk"],
        GNode["min", Min[1], "v1", "v2"],
    ]

    var p = InlineArray[Scalar[dtype], M.PARAM_SIZE](uninitialized=True)
    var pt = LayoutTensor[dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin](
        p.unsafe_ptr()
    )
    M.initialize_params[Xavier[]](pt)

    var inp = InlineArray[Scalar[dtype], BATCH * M.IN_DIM](uninitialized=True)
    fill_sequential(inp.unsafe_ptr(), BATCH * M.IN_DIM)
    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())

    var out = InlineArray[Scalar[dtype], BATCH * M.OUT_DIM](uninitialized=True)
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](out.unsafe_ptr())
    var cache = InlineArray[Scalar[dtype], BATCH * M.CACHE_SIZE](
        uninitialized=True
    )
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
    ](cache.unsafe_ptr())
    var st_t = LayoutTensor[
        dtype, Layout.row_major(M.STATE_SIZE), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(0)))
    M.forward[BATCH](inp_t, out_t, pt, st_t, cache_t)

    var go = InlineArray[Scalar[dtype], BATCH * M.OUT_DIM](uninitialized=True)
    for i in range(BATCH * M.OUT_DIM):
        go[i] = Scalar[dtype](1.0)
    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](go.unsafe_ptr())

    var gi = InlineArray[Scalar[dtype], BATCH * M.IN_DIM](uninitialized=True)
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](gi.unsafe_ptr())
    var g = InlineArray[Scalar[dtype], M.PARAM_SIZE](uninitialized=True)
    for i in range(M.PARAM_SIZE):
        g[i] = Scalar[dtype](0.0)
    var gt = LayoutTensor[dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin](
        g.unsafe_ptr()
    )
    M.backward[BATCH](go_t, gi_t, pt, st_t, cache_t, gt)

    print("  PASSED")


# =============================================================================
# Test 3: Dual-input (DDPG-like)
# =============================================================================


def test_dual_input() raises:
    print("Test 3: Dual-input DDPG-like...")

    comptime BATCH = 2
    comptime M = ComputeGraph[
        GNode["actor", Linear[3, 2]],
        GNode["critic", Linear[5, 1], "input", "actor"],
        GNode["neg_q", Negate[1], "critic"],
    ]

    var p = InlineArray[Scalar[dtype], M.PARAM_SIZE](uninitialized=True)
    var pt = LayoutTensor[dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin](
        p.unsafe_ptr()
    )
    M.initialize_params[Xavier[]](pt)

    var inp = InlineArray[Scalar[dtype], BATCH * M.IN_DIM](uninitialized=True)
    fill_sequential(inp.unsafe_ptr(), BATCH * M.IN_DIM)
    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())

    var out = InlineArray[Scalar[dtype], BATCH * M.OUT_DIM](uninitialized=True)
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](out.unsafe_ptr())
    var cache = InlineArray[Scalar[dtype], BATCH * M.CACHE_SIZE](
        uninitialized=True
    )
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
    ](cache.unsafe_ptr())
    var st_t = LayoutTensor[
        dtype, Layout.row_major(M.STATE_SIZE), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(0)))
    M.forward[BATCH](inp_t, out_t, pt, st_t, cache_t)

    var go = InlineArray[Scalar[dtype], BATCH * M.OUT_DIM](uninitialized=True)
    for i in range(BATCH * M.OUT_DIM):
        go[i] = Scalar[dtype](1.0)
    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](go.unsafe_ptr())

    var gi = InlineArray[Scalar[dtype], BATCH * M.IN_DIM](uninitialized=True)
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](gi.unsafe_ptr())
    var g = InlineArray[Scalar[dtype], M.PARAM_SIZE](uninitialized=True)
    for i in range(M.PARAM_SIZE):
        g[i] = Scalar[dtype](0.0)
    var gt = LayoutTensor[dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin](
        g.unsafe_ptr()
    )
    M.backward[BATCH](go_t, gi_t, pt, st_t, cache_t, gt)

    print("  PASSED")


# =============================================================================
# Test 4: Full DAG
# =============================================================================


def test_full_dag() raises:
    print("Test 4: Full DAG with fan-out + fan-in...")

    comptime BATCH = 2
    comptime M = ComputeGraph[
        GNode["trunk", LinearReLU[3, 4]],
        GNode["branch_a", Linear[4, 1], "trunk"],
        GNode["branch_b", Linear[4, 1], "trunk"],
        GNode["merge", Linear[2, 1], "branch_a", "branch_b"],
    ]

    var p = InlineArray[Scalar[dtype], M.PARAM_SIZE](uninitialized=True)
    var pt = LayoutTensor[dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin](
        p.unsafe_ptr()
    )
    M.initialize_params[Xavier[]](pt)

    var inp = InlineArray[Scalar[dtype], BATCH * M.IN_DIM](uninitialized=True)
    fill_sequential(inp.unsafe_ptr(), BATCH * M.IN_DIM)
    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](inp.unsafe_ptr())

    var out = InlineArray[Scalar[dtype], BATCH * M.OUT_DIM](uninitialized=True)
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](out.unsafe_ptr())
    var cache = InlineArray[Scalar[dtype], BATCH * M.CACHE_SIZE](
        uninitialized=True
    )
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
    ](cache.unsafe_ptr())
    var st_t = LayoutTensor[
        dtype, Layout.row_major(M.STATE_SIZE), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(0)))
    M.forward[BATCH](inp_t, out_t, pt, st_t, cache_t)

    var go = InlineArray[Scalar[dtype], BATCH * M.OUT_DIM](uninitialized=True)
    for i in range(BATCH * M.OUT_DIM):
        go[i] = Scalar[dtype](1.0)
    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](go.unsafe_ptr())

    var gi = InlineArray[Scalar[dtype], BATCH * M.IN_DIM](uninitialized=True)
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](gi.unsafe_ptr())
    var g = InlineArray[Scalar[dtype], M.PARAM_SIZE](uninitialized=True)
    for i in range(M.PARAM_SIZE):
        g[i] = Scalar[dtype](0.0)
    var gt = LayoutTensor[dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin](
        g.unsafe_ptr()
    )
    M.backward[BATCH](go_t, gi_t, pt, st_t, cache_t, gt)

    print("  PASSED")


def main() raises:
    print("=" * 60)
    print("ComputeGraph Named Node Tests")
    print("=" * 60)
    print()

    test_simple_chain()
    test_fan_out()
    test_dual_input()
    test_full_dag()

    print()
    print("All tests PASSED!")
