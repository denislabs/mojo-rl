"""Tests for ComputeGraph — compile-time DAG builder.

Tests:
1. Simple chain (should produce same result as Sequential)
2. Fan-out (one node feeds two downstream nodes)
3. Dual-input concat (two predecessors concatenated)
4. Full DAG with fan-out + fan-in (DDPG-like pattern)
5. Gradient correctness via finite differences
"""

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import (
    Model,
    Sequential,
    Linear,
    LinearReLU,
    Negate,
    Slice,
    Min,
)
from mojo_rl.nn.autodiff.compute_graph import ComputeGraph, GNode
from mojo_rl.nn.initializer import Xavier
from layout import Layout, LayoutTensor
from std.math import abs


# =============================================================================
# Test helpers
# =============================================================================


fn assert_close(
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


fn fill_sequential(ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int):
    for i in range(n):
        ptr[i] = Scalar[dtype](Float64(i + 1) * 0.1)


# =============================================================================
# Test 1: Simple chain — ComputeGraph vs Sequential
# =============================================================================


fn test_simple_chain() raises:
    """ComputeGraph with a linear chain should match Sequential."""
    print("Test 1: Simple chain (ComputeGraph vs Sequential)...")

    comptime BATCH = 2
    comptime IN = 4
    comptime HID = 8
    comptime OUT = 3

    # Define equivalent graphs
    comptime SeqModel = Sequential[LinearReLU[IN, HID], Linear[HID, OUT]]
    comptime GraphModel = ComputeGraph[
        GNode["hidden", LinearReLU[IN, HID]],  # node 0: input → hidden
        GNode["output", Linear[HID, OUT], "hidden"],  # node 1: hidden → output
    ]

    # Verify compile-time constants match
    print(
        "  SeqModel:   IN=",
        SeqModel.IN_DIM,
        "OUT=",
        SeqModel.OUT_DIM,
        "PARAM=",
        SeqModel.PARAM_SIZE,
    )
    print(
        "  GraphModel: IN=",
        GraphModel.IN_DIM,
        "OUT=",
        GraphModel.OUT_DIM,
        "PARAM=",
        GraphModel.PARAM_SIZE,
    )

    # Initialize with same params
    var params = InlineArray[Scalar[dtype], SeqModel.PARAM_SIZE](
        uninitialized=True
    )
    var params_t = LayoutTensor[
        dtype, Layout.row_major(SeqModel.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    SeqModel.initialize_params[Xavier[]](params_t)

    # The graph may have alignment padding — copy params carefully
    var graph_params = InlineArray[Scalar[dtype], GraphModel.PARAM_SIZE](
        uninitialized=True
    )
    for i in range(GraphModel.PARAM_SIZE):
        graph_params[i] = Scalar[dtype](0.0)
    # Copy Sequential params to ComputeGraph params
    # Sequential uses _seq_align4 between layers; ComputeGraph uses _align4
    # Both should have the same aligned offsets since they use the same pattern
    for i in range(SeqModel.PARAM_SIZE):
        graph_params[i] = params[i]
    var graph_params_t = LayoutTensor[
        dtype, Layout.row_major(GraphModel.PARAM_SIZE), MutAnyOrigin
    ](graph_params.unsafe_ptr())

    # Input
    var input_arr = InlineArray[Scalar[dtype], BATCH * IN](uninitialized=True)
    fill_sequential(input_arr.unsafe_ptr(), BATCH * IN)
    var input_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](input_arr.unsafe_ptr())

    # Forward Sequential
    var seq_out = InlineArray[Scalar[dtype], BATCH * OUT](uninitialized=True)
    var seq_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
    ](seq_out.unsafe_ptr())
    var seq_cache = InlineArray[Scalar[dtype], BATCH * SeqModel.CACHE_SIZE](
        uninitialized=True
    )
    var seq_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, SeqModel.CACHE_SIZE), MutAnyOrigin
    ](seq_cache.unsafe_ptr())
    SeqModel.forward[BATCH](input_t, seq_out_t, params_t, seq_cache_t)

    # Forward Graph
    var graph_out = InlineArray[Scalar[dtype], BATCH * OUT](uninitialized=True)
    var graph_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
    ](graph_out.unsafe_ptr())
    var graph_cache = InlineArray[Scalar[dtype], BATCH * GraphModel.CACHE_SIZE](
        uninitialized=True
    )
    var graph_cache_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, GraphModel.CACHE_SIZE),
        MutAnyOrigin,
    ](graph_cache.unsafe_ptr())
    GraphModel.forward[BATCH](
        input_t, graph_out_t, graph_params_t, graph_cache_t
    )

    # Compare outputs
    for i in range(BATCH * OUT):
        assert_close(
            Float64(graph_out[i]),
            Float64(seq_out[i]),
            1e-5,
            "output[" + String(i) + "]",
        )

    # Backward comparison
    var grad_out_arr = InlineArray[Scalar[dtype], BATCH * OUT](
        uninitialized=True
    )
    for i in range(BATCH * OUT):
        grad_out_arr[i] = Scalar[dtype](1.0)
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
    ](grad_out_arr.unsafe_ptr())

    # Seq backward
    var seq_gi = InlineArray[Scalar[dtype], BATCH * IN](uninitialized=True)
    var seq_gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](seq_gi.unsafe_ptr())
    var seq_grads = InlineArray[Scalar[dtype], SeqModel.PARAM_SIZE](
        uninitialized=True
    )
    for i in range(SeqModel.PARAM_SIZE):
        seq_grads[i] = Scalar[dtype](0.0)
    var seq_grads_t = LayoutTensor[
        dtype, Layout.row_major(SeqModel.PARAM_SIZE), MutAnyOrigin
    ](seq_grads.unsafe_ptr())
    SeqModel.backward[BATCH](
        grad_out_t, seq_gi_t, params_t, seq_cache_t, seq_grads_t
    )

    # Graph backward
    var graph_gi = InlineArray[Scalar[dtype], BATCH * IN](uninitialized=True)
    var graph_gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](graph_gi.unsafe_ptr())
    var graph_grads = InlineArray[Scalar[dtype], GraphModel.PARAM_SIZE](
        uninitialized=True
    )
    for i in range(GraphModel.PARAM_SIZE):
        graph_grads[i] = Scalar[dtype](0.0)
    var graph_grads_t = LayoutTensor[
        dtype, Layout.row_major(GraphModel.PARAM_SIZE), MutAnyOrigin
    ](graph_grads.unsafe_ptr())
    GraphModel.backward[BATCH](
        grad_out_t,
        graph_gi_t,
        graph_params_t,
        graph_cache_t,
        graph_grads_t,
    )

    # Compare grad_input
    for i in range(BATCH * IN):
        assert_close(
            Float64(graph_gi[i]),
            Float64(seq_gi[i]),
            1e-5,
            "grad_input[" + String(i) + "]",
        )

    # Compare param grads (up to SeqModel size, ignoring alignment)
    for i in range(SeqModel.PARAM_SIZE):
        assert_close(
            Float64(graph_grads[i]),
            Float64(seq_grads[i]),
            1e-5,
            "grad_param[" + String(i) + "]",
        )

    print("  PASSED")


# =============================================================================
# Test 2: Fan-out — one node feeds two downstream nodes
# =============================================================================


fn test_fan_out() raises:
    """Test fan-out: node 0 feeds both node 1 and node 2.

    Graph:
        input(2) → Linear[2,3](node 0) → act(3)
                                           ├→ Linear[3,1](node 1) → out1(1)
                                           └→ Linear[3,1](node 2) → out2(1)
        output = [out1, out2] via a final concat node

    But since we don't have a general concat node yet, let's test
    fan-out + Min: both paths produce scalar, then MinOp picks the min.

    Graph:
        input(4) → LinearReLU[4,3](node 0) → h(3)
                                                ├→ Linear[3,1](node 1) → v1(1)
                                                └→ Linear[3,1](node 2) → v2(1)
        Min[1](node 3, inputs from node 1 and 2) → min(v1,v2)
    """
    print("Test 2: Fan-out with MinOp...")

    comptime BATCH = 2

    comptime FanOutGraph = ComputeGraph[
        GNode["trunk", LinearReLU[4, 3]],  # 0: input → hidden
        GNode["v1", Linear[3, 1], "trunk"],  # 1: hidden → v1  (fan-out)
        GNode["v2", Linear[3, 1], "trunk"],  # 2: hidden → v2  (fan-out)
        GNode["min", Min[1], "v1", "v2"],  # 3: [v1, v2] → min
    ]

    print(
        "  FanOutGraph: IN=",
        FanOutGraph.IN_DIM,
        "OUT=",
        FanOutGraph.OUT_DIM,
        "PARAM=",
        FanOutGraph.PARAM_SIZE,
        "CACHE=",
        FanOutGraph.CACHE_SIZE,
    )

    # Check dimensions
    comptime if FanOutGraph.IN_DIM != 4:
        raise Error("IN_DIM should be 4")
    comptime if FanOutGraph.OUT_DIM != 1:
        raise Error("OUT_DIM should be 1")

    # Initialize params
    var params = InlineArray[Scalar[dtype], FanOutGraph.PARAM_SIZE](
        uninitialized=True
    )
    var params_t = LayoutTensor[
        dtype, Layout.row_major(FanOutGraph.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    FanOutGraph.initialize_params[Xavier[]](params_t)

    # Input
    var input_arr = InlineArray[Scalar[dtype], BATCH * 4](uninitialized=True)
    fill_sequential(input_arr.unsafe_ptr(), BATCH * 4)
    var input_t = LayoutTensor[dtype, Layout.row_major(BATCH, 4), MutAnyOrigin](
        input_arr.unsafe_ptr()
    )

    # Forward
    var output_arr = InlineArray[Scalar[dtype], BATCH * 1](uninitialized=True)
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
    ](output_arr.unsafe_ptr())
    var cache_arr = InlineArray[Scalar[dtype], BATCH * FanOutGraph.CACHE_SIZE](
        uninitialized=True
    )
    var cache_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, FanOutGraph.CACHE_SIZE),
        MutAnyOrigin,
    ](cache_arr.unsafe_ptr())

    FanOutGraph.forward[BATCH](input_t, output_t, params_t, cache_t)

    print("  Forward output:", output_arr[0], output_arr[1])

    # Backward
    var grad_out = InlineArray[Scalar[dtype], BATCH * 1](uninitialized=True)
    grad_out[0] = Scalar[dtype](1.0)
    grad_out[1] = Scalar[dtype](1.0)
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
    ](grad_out.unsafe_ptr())

    var grad_in = InlineArray[Scalar[dtype], BATCH * 4](uninitialized=True)
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 4), MutAnyOrigin
    ](grad_in.unsafe_ptr())
    var grads_arr = InlineArray[Scalar[dtype], FanOutGraph.PARAM_SIZE](
        uninitialized=True
    )
    for i in range(FanOutGraph.PARAM_SIZE):
        grads_arr[i] = Scalar[dtype](0.0)
    var grads_t = LayoutTensor[
        dtype, Layout.row_major(FanOutGraph.PARAM_SIZE), MutAnyOrigin
    ](grads_arr.unsafe_ptr())

    FanOutGraph.backward[BATCH](
        grad_out_t, grad_in_t, params_t, cache_t, grads_t
    )

    print(
        "  Backward grad_input:", grad_in[0], grad_in[1], grad_in[2], grad_in[3]
    )

    # Verify grad_input is non-zero (fan-out should accumulate)
    var any_nonzero = False
    for i in range(BATCH * 4):
        if abs(Float64(grad_in[i])) > 1e-10:
            any_nonzero = True
    if not any_nonzero:
        raise Error("grad_input is all zeros — fan-out accumulation failed")

    print("  PASSED")


# =============================================================================
# Test 3: Dual-input concat — DDPG-like skip connection
# =============================================================================


fn test_dual_input_concat() raises:
    """Test dual-input: graph input feeds into both node 0 and concat node.

    Graph (simplified DDPG):
        input(4) ──→ Linear[4,2](node 0) → action(2)
                 └─→ concat with action → [input(4), action(2)] = (6)
                     → Linear[6,1](node 2) → Q
                     → Negate[1](node 3) → -Q
    """
    print("Test 3: Dual-input concat (DDPG-like)...")

    comptime BATCH = 2

    # Note: node 1 has dual inputs: graph input (-1) and node 0
    # So node 1's OP_IN_DIM must be 4+2 = 6 (concat dim)
    # Using a Linear[6,1] that takes the concat as input
    comptime DDPGGraph = ComputeGraph[
        GNode["actor", Linear[4, 2]],  # 0: obs → action
        GNode["critic", Linear[6, 1], "input", "actor"],  # 1: [obs, action] → Q (dual input)
        GNode["neg_q", Negate[1], "critic"],  # 2: → -Q
    ]

    print(
        "  DDPGGraph: IN=",
        DDPGGraph.IN_DIM,
        "OUT=",
        DDPGGraph.OUT_DIM,
        "PARAM=",
        DDPGGraph.PARAM_SIZE,
    )

    comptime if DDPGGraph.IN_DIM != 4:
        raise Error("IN_DIM should be 4")
    comptime if DDPGGraph.OUT_DIM != 1:
        raise Error("OUT_DIM should be 1")

    # Initialize
    var params = InlineArray[Scalar[dtype], DDPGGraph.PARAM_SIZE](
        uninitialized=True
    )
    var params_t = LayoutTensor[
        dtype, Layout.row_major(DDPGGraph.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    DDPGGraph.initialize_params[Xavier[]](params_t)

    # Input
    var input_arr = InlineArray[Scalar[dtype], BATCH * 4](uninitialized=True)
    fill_sequential(input_arr.unsafe_ptr(), BATCH * 4)
    var input_t = LayoutTensor[dtype, Layout.row_major(BATCH, 4), MutAnyOrigin](
        input_arr.unsafe_ptr()
    )

    # Forward
    var output_arr = InlineArray[Scalar[dtype], BATCH * 1](uninitialized=True)
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
    ](output_arr.unsafe_ptr())
    var cache_arr = InlineArray[Scalar[dtype], BATCH * DDPGGraph.CACHE_SIZE](
        uninitialized=True
    )
    var cache_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, DDPGGraph.CACHE_SIZE),
        MutAnyOrigin,
    ](cache_arr.unsafe_ptr())

    DDPGGraph.forward[BATCH](input_t, output_t, params_t, cache_t)
    print("  Forward output:", output_arr[0], output_arr[1])

    # Backward
    var grad_out = InlineArray[Scalar[dtype], BATCH * 1](uninitialized=True)
    grad_out[0] = Scalar[dtype](1.0)
    grad_out[1] = Scalar[dtype](1.0)
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
    ](grad_out.unsafe_ptr())

    var grad_in = InlineArray[Scalar[dtype], BATCH * 4](uninitialized=True)
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 4), MutAnyOrigin
    ](grad_in.unsafe_ptr())
    var grads_arr = InlineArray[Scalar[dtype], DDPGGraph.PARAM_SIZE](
        uninitialized=True
    )
    for i in range(DDPGGraph.PARAM_SIZE):
        grads_arr[i] = Scalar[dtype](0.0)
    var grads_t = LayoutTensor[
        dtype, Layout.row_major(DDPGGraph.PARAM_SIZE), MutAnyOrigin
    ](grads_arr.unsafe_ptr())

    DDPGGraph.backward[BATCH](grad_out_t, grad_in_t, params_t, cache_t, grads_t)

    print(
        "  Backward grad_input:", grad_in[0], grad_in[1], grad_in[2], grad_in[3]
    )

    # grad_input should be non-zero (input feeds into both actor and critic)
    var any_nonzero = False
    for i in range(BATCH * 4):
        if abs(Float64(grad_in[i])) > 1e-10:
            any_nonzero = True
    if not any_nonzero:
        raise Error("grad_input is all zeros — dual-input backward failed")

    print("  PASSED")


# =============================================================================
# Test 4: Numerical gradient check for ComputeGraph
# =============================================================================


fn test_grad_check_simple_chain() raises:
    """Finite-difference gradient check for a simple chain graph."""
    print("Test 4: Gradient check (simple chain)...")

    comptime BATCH = 1
    comptime M = ComputeGraph[
        GNode["hidden", LinearReLU[3, 4]],
        GNode["output", Linear[4, 2], "hidden"],
    ]

    var params_arr = InlineArray[Scalar[dtype], M.PARAM_SIZE](
        uninitialized=True
    )
    var params_t = LayoutTensor[
        dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
    ](params_arr.unsafe_ptr())
    M.initialize_params[Xavier[]](params_t)

    var input_arr = InlineArray[Scalar[dtype], BATCH * 3](uninitialized=True)
    input_arr[0] = Scalar[dtype](0.5)
    input_arr[1] = Scalar[dtype](-0.3)
    input_arr[2] = Scalar[dtype](0.8)

    var grad_out_arr = InlineArray[Scalar[dtype], BATCH * 2](uninitialized=True)
    grad_out_arr[0] = Scalar[dtype](1.0)
    grad_out_arr[1] = Scalar[dtype](0.5)

    var input_t = LayoutTensor[dtype, Layout.row_major(BATCH, 3), MutAnyOrigin](
        input_arr.unsafe_ptr()
    )
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 2), MutAnyOrigin
    ](grad_out_arr.unsafe_ptr())

    # Analytical backward
    var cache_arr = InlineArray[Scalar[dtype], BATCH * M.CACHE_SIZE](
        uninitialized=True
    )
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
    ](cache_arr.unsafe_ptr())
    var output_arr = InlineArray[Scalar[dtype], BATCH * 2](uninitialized=True)
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 2), MutAnyOrigin
    ](output_arr.unsafe_ptr())

    M.forward[BATCH](input_t, output_t, params_t, cache_t)

    var grad_in_arr = InlineArray[Scalar[dtype], BATCH * 3](uninitialized=True)
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 3), MutAnyOrigin
    ](grad_in_arr.unsafe_ptr())
    var grads_arr = InlineArray[Scalar[dtype], M.PARAM_SIZE](uninitialized=True)
    for i in range(M.PARAM_SIZE):
        grads_arr[i] = Scalar[dtype](0.0)
    var grads_t = LayoutTensor[
        dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
    ](grads_arr.unsafe_ptr())

    M.backward[BATCH](grad_out_t, grad_in_t, params_t, cache_t, grads_t)

    # Finite difference
    var eps = Float64(1e-4)
    var max_rel_err: Float64 = 0.0

    for p_idx in range(M.PARAM_SIZE):
        var orig = params_arr[p_idx]

        # f(p + eps)
        params_arr[p_idx] = orig + Scalar[dtype](eps)
        var out_plus = InlineArray[Scalar[dtype], BATCH * 2](uninitialized=True)
        var out_plus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 2), MutAnyOrigin
        ](out_plus.unsafe_ptr())
        M.forward[BATCH](input_t, out_plus_t, params_t)

        # f(p - eps)
        params_arr[p_idx] = orig - Scalar[dtype](eps)
        var out_minus = InlineArray[Scalar[dtype], BATCH * 2](
            uninitialized=True
        )
        var out_minus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 2), MutAnyOrigin
        ](out_minus.unsafe_ptr())
        M.forward[BATCH](input_t, out_minus_t, params_t)

        params_arr[p_idx] = orig  # Restore

        # Compute numerical gradient
        var fd_grad: Float64 = 0.0
        for o in range(BATCH * 2):
            fd_grad += (
                Float64((out_plus[o] - out_minus[o]))
                / (2.0 * eps)
                * Float64(grad_out_arr[o])
            )

        var anal_grad = Float64(grads_arr[p_idx])
        var abs_err = abs(fd_grad - anal_grad)
        var denom = max(abs(fd_grad), abs(anal_grad))
        if denom > 1e-7:
            var rel_err = abs_err / denom
            if rel_err > max_rel_err:
                max_rel_err = rel_err

    print("  Max relative error:", max_rel_err)
    if max_rel_err > 0.01:
        raise Error("Gradient check failed: max rel error too high")

    print("  PASSED")


# =============================================================================
# Test 5: Gradient check for fan-out graph
# =============================================================================


fn test_grad_check_fan_out() raises:
    """Finite-difference gradient check for fan-out graph.

    Uses Linear[2,1] instead of Min to combine branches — Min is
    non-differentiable at v1=v2 which causes finite-diff to disagree.
    """
    print("Test 5: Gradient check (fan-out)...")

    comptime BATCH = 1
    # Fan-out: node 0 feeds both node 1 and node 2
    # Then Linear[2,1] smoothly combines them (differentiable everywhere)
    comptime M = ComputeGraph[
        GNode["trunk", LinearReLU[3, 4]],  # 0: shared trunk
        GNode["branch_a", Linear[4, 1], "trunk"],  # 1: branch A (fan-out)
        GNode["branch_b", Linear[4, 1], "trunk"],  # 2: branch B (fan-out)
        GNode["merge", Linear[2, 1], "branch_a", "branch_b"],  # 3: concat(A,B) → weighted sum
    ]

    var params_arr = InlineArray[Scalar[dtype], M.PARAM_SIZE](
        uninitialized=True
    )
    var params_t = LayoutTensor[
        dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
    ](params_arr.unsafe_ptr())
    M.initialize_params[Xavier[]](params_t)

    var input_arr = InlineArray[Scalar[dtype], BATCH * 3](uninitialized=True)
    input_arr[0] = Scalar[dtype](0.5)
    input_arr[1] = Scalar[dtype](-0.3)
    input_arr[2] = Scalar[dtype](0.8)

    var grad_out_arr = InlineArray[Scalar[dtype], BATCH * 1](uninitialized=True)
    grad_out_arr[0] = Scalar[dtype](1.0)

    var input_t = LayoutTensor[dtype, Layout.row_major(BATCH, 3), MutAnyOrigin](
        input_arr.unsafe_ptr()
    )
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
    ](grad_out_arr.unsafe_ptr())

    # Analytical backward
    var cache_arr = InlineArray[Scalar[dtype], BATCH * M.CACHE_SIZE](
        uninitialized=True
    )
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
    ](cache_arr.unsafe_ptr())
    var output_arr = InlineArray[Scalar[dtype], BATCH * 1](uninitialized=True)
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
    ](output_arr.unsafe_ptr())

    M.forward[BATCH](input_t, output_t, params_t, cache_t)

    var grad_in_arr = InlineArray[Scalar[dtype], BATCH * 3](uninitialized=True)
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 3), MutAnyOrigin
    ](grad_in_arr.unsafe_ptr())
    var grads_arr = InlineArray[Scalar[dtype], M.PARAM_SIZE](uninitialized=True)
    for i in range(M.PARAM_SIZE):
        grads_arr[i] = Scalar[dtype](0.0)
    var grads_t = LayoutTensor[
        dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
    ](grads_arr.unsafe_ptr())

    M.backward[BATCH](grad_out_t, grad_in_t, params_t, cache_t, grads_t)

    # Finite difference
    var eps = Float64(1e-4)
    var max_rel_err: Float64 = 0.0

    for p_idx in range(M.PARAM_SIZE):
        var orig = params_arr[p_idx]

        params_arr[p_idx] = orig + Scalar[dtype](eps)
        var out_plus = InlineArray[Scalar[dtype], BATCH * 1](uninitialized=True)
        var out_plus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](out_plus.unsafe_ptr())
        M.forward[BATCH](input_t, out_plus_t, params_t)

        params_arr[p_idx] = orig - Scalar[dtype](eps)
        var out_minus = InlineArray[Scalar[dtype], BATCH * 1](
            uninitialized=True
        )
        var out_minus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](out_minus.unsafe_ptr())
        M.forward[BATCH](input_t, out_minus_t, params_t)

        params_arr[p_idx] = orig

        var fd_grad = Float64(out_plus[0] - out_minus[0]) / (2.0 * eps)
        var anal_grad = Float64(grads_arr[p_idx])
        var abs_err = abs(fd_grad - anal_grad)
        var denom = max(abs(fd_grad), abs(anal_grad))
        if denom > 1e-7:
            var rel_err = abs_err / denom
            if rel_err > max_rel_err:
                max_rel_err = rel_err

    print("  Max relative error:", max_rel_err)
    if max_rel_err > 0.01:
        raise Error("Gradient check failed: max rel error too high")

    print("  PASSED")


# =============================================================================
# Test 6: Gradient check for dual-input concat graph
# =============================================================================


fn test_grad_check_dual_input() raises:
    """Finite-difference gradient check for dual-input (DDPG-like) graph."""
    print("Test 6: Gradient check (dual-input concat)...")

    comptime BATCH = 1
    comptime M = ComputeGraph[
        GNode["actor", Linear[3, 2]],  # 0: obs → action
        GNode["critic", Linear[5, 1], "input", "actor"],  # 1: [obs(3), action(2)] → Q
        GNode["neg_q", Negate[1], "critic"],  # 2: → -Q
    ]

    var params_arr = InlineArray[Scalar[dtype], M.PARAM_SIZE](
        uninitialized=True
    )
    var params_t = LayoutTensor[
        dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
    ](params_arr.unsafe_ptr())
    M.initialize_params[Xavier[]](params_t)

    var input_arr = InlineArray[Scalar[dtype], BATCH * 3](uninitialized=True)
    input_arr[0] = Scalar[dtype](0.5)
    input_arr[1] = Scalar[dtype](-0.3)
    input_arr[2] = Scalar[dtype](0.8)

    var grad_out_arr = InlineArray[Scalar[dtype], BATCH * 1](uninitialized=True)
    grad_out_arr[0] = Scalar[dtype](1.0)

    var input_t = LayoutTensor[dtype, Layout.row_major(BATCH, 3), MutAnyOrigin](
        input_arr.unsafe_ptr()
    )
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
    ](grad_out_arr.unsafe_ptr())

    # Analytical backward
    var cache_arr = InlineArray[Scalar[dtype], BATCH * M.CACHE_SIZE](
        uninitialized=True
    )
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
    ](cache_arr.unsafe_ptr())
    var output_arr = InlineArray[Scalar[dtype], BATCH * 1](uninitialized=True)
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
    ](output_arr.unsafe_ptr())

    M.forward[BATCH](input_t, output_t, params_t, cache_t)

    var grad_in_arr = InlineArray[Scalar[dtype], BATCH * 3](uninitialized=True)
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 3), MutAnyOrigin
    ](grad_in_arr.unsafe_ptr())
    var grads_arr = InlineArray[Scalar[dtype], M.PARAM_SIZE](uninitialized=True)
    for i in range(M.PARAM_SIZE):
        grads_arr[i] = Scalar[dtype](0.0)
    var grads_t = LayoutTensor[
        dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
    ](grads_arr.unsafe_ptr())

    M.backward[BATCH](grad_out_t, grad_in_t, params_t, cache_t, grads_t)

    # Finite difference for params
    var eps = Float64(1e-4)
    var max_rel_err_params: Float64 = 0.0

    for p_idx in range(M.PARAM_SIZE):
        var orig = params_arr[p_idx]

        params_arr[p_idx] = orig + Scalar[dtype](eps)
        var out_plus = InlineArray[Scalar[dtype], BATCH * 1](uninitialized=True)
        var out_plus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](out_plus.unsafe_ptr())
        M.forward[BATCH](input_t, out_plus_t, params_t)

        params_arr[p_idx] = orig - Scalar[dtype](eps)
        var out_minus = InlineArray[Scalar[dtype], BATCH * 1](
            uninitialized=True
        )
        var out_minus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](out_minus.unsafe_ptr())
        M.forward[BATCH](input_t, out_minus_t, params_t)

        params_arr[p_idx] = orig

        var fd_grad = Float64(out_plus[0] - out_minus[0]) / (2.0 * eps)
        var anal_grad = Float64(grads_arr[p_idx])
        var abs_err = abs(fd_grad - anal_grad)
        var denom = max(abs(fd_grad), abs(anal_grad))
        if denom > 1e-7:
            var rel_err = abs_err / denom
            if rel_err > max_rel_err_params:
                max_rel_err_params = rel_err

    print("  Max relative error (params):", max_rel_err_params)

    # Finite difference for input gradients
    var max_rel_err_input: Float64 = 0.0
    for in_idx in range(3):
        var orig = input_arr[in_idx]

        input_arr[in_idx] = orig + Scalar[dtype](eps)
        var out_plus = InlineArray[Scalar[dtype], BATCH * 1](uninitialized=True)
        var out_plus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](out_plus.unsafe_ptr())
        M.forward[BATCH](input_t, out_plus_t, params_t)

        input_arr[in_idx] = orig - Scalar[dtype](eps)
        var out_minus = InlineArray[Scalar[dtype], BATCH * 1](
            uninitialized=True
        )
        var out_minus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](out_minus.unsafe_ptr())
        M.forward[BATCH](input_t, out_minus_t, params_t)

        input_arr[in_idx] = orig

        var fd_grad = Float64(out_plus[0] - out_minus[0]) / (2.0 * eps)
        var anal_grad = Float64(grad_in_arr[in_idx])
        var abs_err = abs(fd_grad - anal_grad)
        var denom = max(abs(fd_grad), abs(anal_grad))
        if denom > 1e-7:
            var rel_err = abs_err / denom
            if rel_err > max_rel_err_input:
                max_rel_err_input = rel_err

    print("  Max relative error (input):", max_rel_err_input)

    if max_rel_err_params > 0.01:
        raise Error("Param gradient check failed")
    if max_rel_err_input > 0.01:
        raise Error("Input gradient check failed")

    print("  PASSED")


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    print("=" * 60)
    print("ComputeGraph Tests")
    print("=" * 60)

    test_simple_chain()
    test_fan_out()
    test_dual_input_concat()
    test_grad_check_simple_chain()
    test_grad_check_fan_out()
    test_grad_check_dual_input()

    print("=" * 60)
    print("All ComputeGraph tests PASSED!")
    print("=" * 60)
