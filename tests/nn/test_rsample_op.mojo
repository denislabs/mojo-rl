"""Test RSampleOp and MinOp DiffOps."""

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.autodiff.primitives import RSampleOp, MinOp
from layout import Layout, LayoutTensor
from std.math import exp, log, tanh
from std.random import random_float64


def test_rsample_op_shapes() raises:
    """Verify RSampleOp compile-time shape constants."""
    comptime A = 6
    comptime R = RSampleOp[A]

    # IN_DIM = 2 * action_dim (mean || tanh_log_std)
    constrained[R.IN_DIM == 12, "RSampleOp IN_DIM should be 2*A"]()
    # OUT_DIM = action_dim + 1 (action || log_prob)
    constrained[R.OUT_DIM == 7, "RSampleOp OUT_DIM should be A+1"]()
    # PARAM_SIZE = 0 (no learnable params)
    constrained[R.PARAM_SIZE == 0, "RSampleOp should have no params"]()
    # CACHE_SIZE = 3 * action_dim (action, noise, log_std)
    constrained[R.CACHE_SIZE == 18, "RSampleOp CACHE_SIZE should be 3*A"]()

    print("  [PASS] RSampleOp shapes correct")


def test_rsample_op_forward() raises:
    """Test RSampleOp forward pass produces valid actions and log_probs."""
    comptime A = 3
    comptime BS = 4

    # Create input: [mean || tanh_raw_log_std]
    var input_arr = InlineArray[Scalar[dtype], BS * 2 * A](uninitialized=True)
    for b in range(BS):
        for j in range(A):
            # mean values
            input_arr[b * 2 * A + j] = Scalar[dtype](0.5 * Float64(j))
            # tanh(raw_log_std) values in [-1, 1]
            input_arr[b * 2 * A + A + j] = Scalar[dtype](
                0.0
            )  # maps to log_std = -1.5

    var input_t = LayoutTensor[
        dtype, Layout.row_major(BS, 2 * A), MutAnyOrigin
    ](input_arr.unsafe_ptr())

    var output_arr = InlineArray[Scalar[dtype], BS * (A + 1)](
        uninitialized=True
    )
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BS, A + 1), MutAnyOrigin
    ](output_arr.unsafe_ptr())

    # Empty params (PARAM_SIZE = 0) — need at least 1 element
    var params_arr = InlineArray[Scalar[dtype], 1](uninitialized=True)
    var params_t = LayoutTensor[
        dtype, Layout.row_major(RSampleOp[A].PARAM_SIZE), MutAnyOrigin
    ](params_arr.unsafe_ptr())

    var cache_arr = InlineArray[Scalar[dtype], BS * 3 * A](uninitialized=True)
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BS, 3 * A), MutAnyOrigin
    ](cache_arr.unsafe_ptr())

    RSampleOp[A].eval[BS](input_t, output_t, params_t, cache_t)

    # Check actions are in [-1, 1] (tanh squashing)
    var all_valid = True
    for b in range(BS):
        for j in range(A):
            var action = Float64(output_arr[b * (A + 1) + j])
            if action < -1.0 or action > 1.0:
                all_valid = False
                print(
                    "  [FAIL] action out of range:",
                    action,
                    "at b=",
                    b,
                    "j=",
                    j,
                )

    if all_valid:
        print("  [PASS] RSampleOp forward: actions in [-1, 1]")

    # Check log_probs are finite negative numbers
    var lp_valid = True
    for b in range(BS):
        var lp = Float64(output_arr[b * (A + 1) + A])
        if lp != lp or lp > 10.0:  # NaN check or unreasonably positive
            lp_valid = False
            print("  [FAIL] log_prob invalid:", lp, "at b=", b)

    if lp_valid:
        print("  [PASS] RSampleOp forward: log_probs finite")


def test_rsample_op_backward() raises:
    """Test RSampleOp backward produces non-zero gradients."""
    comptime A = 2
    comptime BS = 2

    # Forward first
    var input_arr = InlineArray[Scalar[dtype], BS * 2 * A](uninitialized=True)
    for i in range(BS * 2 * A):
        input_arr[i] = Scalar[dtype](0.1 * Float64(i))

    var input_t = LayoutTensor[
        dtype, Layout.row_major(BS, 2 * A), MutAnyOrigin
    ](input_arr.unsafe_ptr())
    var output_arr = InlineArray[Scalar[dtype], BS * (A + 1)](
        uninitialized=True
    )
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BS, A + 1), MutAnyOrigin
    ](output_arr.unsafe_ptr())
    var params_arr = InlineArray[Scalar[dtype], 1](uninitialized=True)
    var params_t = LayoutTensor[
        dtype, Layout.row_major(RSampleOp[A].PARAM_SIZE), MutAnyOrigin
    ](params_arr.unsafe_ptr())
    var cache_arr = InlineArray[Scalar[dtype], BS * 3 * A](uninitialized=True)
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BS, 3 * A), MutAnyOrigin
    ](cache_arr.unsafe_ptr())

    RSampleOp[A].eval[BS](input_t, output_t, params_t, cache_t)

    # Backward
    var grad_out_arr = InlineArray[Scalar[dtype], BS * (A + 1)](
        uninitialized=True
    )
    # grad_action = -1/BS (from critic gradient), grad_log_prob = alpha/BS
    for b in range(BS):
        for j in range(A):
            grad_out_arr[b * (A + 1) + j] = Scalar[dtype](-1.0 / Float64(BS))
        grad_out_arr[b * (A + 1) + A] = Scalar[dtype](
            0.2 / Float64(BS)
        )  # alpha=0.2

    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BS, A + 1), MutAnyOrigin
    ](grad_out_arr.unsafe_ptr())
    var grad_in_arr = InlineArray[Scalar[dtype], BS * 2 * A](uninitialized=True)
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BS, 2 * A), MutAnyOrigin
    ](grad_in_arr.unsafe_ptr())
    var grad_params_arr = InlineArray[Scalar[dtype], 1](uninitialized=True)
    var grad_params_t = LayoutTensor[
        dtype, Layout.row_major(RSampleOp[A].PARAM_SIZE), MutAnyOrigin
    ](grad_params_arr.unsafe_ptr())

    RSampleOp[A].vjp[BS](
        grad_out_t, grad_in_t, params_t, cache_t, grad_params_t
    )

    # Check gradients are non-zero and finite
    var grad_ok = True
    var any_nonzero = False
    for i in range(BS * 2 * A):
        var g = Float64(grad_in_arr[i])
        if g != g:  # NaN
            grad_ok = False
            print("  [FAIL] NaN gradient at index", i)
        if g != 0.0:
            any_nonzero = True

    if grad_ok and any_nonzero:
        print("  [PASS] RSampleOp backward: gradients finite and non-zero")
    elif not any_nonzero:
        print("  [FAIL] RSampleOp backward: all gradients zero")


def test_min_op_shapes() raises:
    """Verify MinOp compile-time shape constants."""
    comptime M = MinOp[1]

    constrained[M.IN_DIM == 2, "MinOp IN_DIM should be 2*dim"]()
    constrained[M.OUT_DIM == 1, "MinOp OUT_DIM should be dim"]()
    constrained[M.PARAM_SIZE == 0, "MinOp should have no params"]()
    constrained[M.CACHE_SIZE == 1, "MinOp CACHE_SIZE should be dim"]()

    print("  [PASS] MinOp shapes correct")


def test_min_op_forward_backward() raises:
    """Test MinOp forward selects min and backward routes gradient."""
    comptime D = 1
    comptime BS = 4

    # Input: [Q1 || Q2] per sample
    var input_arr = InlineArray[Scalar[dtype], BS * 2](uninitialized=True)
    # Sample 0: Q1=5.0, Q2=3.0 → min=3.0 (Q2)
    input_arr[0] = Scalar[dtype](5.0)
    input_arr[1] = Scalar[dtype](3.0)
    # Sample 1: Q1=2.0, Q2=4.0 → min=2.0 (Q1)
    input_arr[2] = Scalar[dtype](2.0)
    input_arr[3] = Scalar[dtype](4.0)
    # Sample 2: Q1=1.0, Q2=1.0 → min=1.0 (Q1, tie)
    input_arr[4] = Scalar[dtype](1.0)
    input_arr[5] = Scalar[dtype](1.0)
    # Sample 3: Q1=-1.0, Q2=0.0 → min=-1.0 (Q1)
    input_arr[6] = Scalar[dtype](-1.0)
    input_arr[7] = Scalar[dtype](0.0)

    var input_t = LayoutTensor[dtype, Layout.row_major(BS, 2), MutAnyOrigin](
        input_arr.unsafe_ptr()
    )
    var output_arr = InlineArray[Scalar[dtype], BS](uninitialized=True)
    var output_t = LayoutTensor[dtype, Layout.row_major(BS, 1), MutAnyOrigin](
        output_arr.unsafe_ptr()
    )
    var params_arr = InlineArray[Scalar[dtype], 1](uninitialized=True)
    var params_t = LayoutTensor[
        dtype, Layout.row_major(MinOp[D].PARAM_SIZE), MutAnyOrigin
    ](params_arr.unsafe_ptr())
    var cache_arr = InlineArray[Scalar[dtype], BS](uninitialized=True)
    var cache_t = LayoutTensor[dtype, Layout.row_major(BS, 1), MutAnyOrigin](
        cache_arr.unsafe_ptr()
    )

    MinOp[D].eval[BS](input_t, output_t, params_t, cache_t)

    # Check forward values
    var fwd_ok = True
    var expected = InlineArray[Float64, 4](uninitialized=True)
    expected[0] = 3.0
    expected[1] = 2.0
    expected[2] = 1.0
    expected[3] = -1.0
    for b in range(BS):
        var val = Float64(output_arr[b])
        if val != expected[b]:
            fwd_ok = False
            print("  [FAIL] MinOp forward: expected", expected[b], "got", val)

    if fwd_ok:
        print("  [PASS] MinOp forward: correct min values")

    # Backward: all grad_output = -1.0
    var grad_out_arr = InlineArray[Scalar[dtype], BS](uninitialized=True)
    for b in range(BS):
        grad_out_arr[b] = Scalar[dtype](-1.0)

    var grad_out_t = LayoutTensor[dtype, Layout.row_major(BS, 1), MutAnyOrigin](
        grad_out_arr.unsafe_ptr()
    )
    var grad_in_arr = InlineArray[Scalar[dtype], BS * 2](uninitialized=True)
    var grad_in_t = LayoutTensor[dtype, Layout.row_major(BS, 2), MutAnyOrigin](
        grad_in_arr.unsafe_ptr()
    )
    var grad_params_arr = InlineArray[Scalar[dtype], 1](uninitialized=True)
    var grad_params_t = LayoutTensor[
        dtype, Layout.row_major(MinOp[D].PARAM_SIZE), MutAnyOrigin
    ](grad_params_arr.unsafe_ptr())

    MinOp[D].vjp[BS](grad_out_t, grad_in_t, params_t, cache_t, grad_params_t)

    # Check gradient routing
    var bwd_ok = True
    # Sample 0: Q2 selected → grad_Q1=0, grad_Q2=-1
    if Float64(grad_in_arr[0]) != 0.0 or Float64(grad_in_arr[1]) != -1.0:
        bwd_ok = False
        print("  [FAIL] MinOp backward sample 0")
    # Sample 1: Q1 selected → grad_Q1=-1, grad_Q2=0
    if Float64(grad_in_arr[2]) != -1.0 or Float64(grad_in_arr[3]) != 0.0:
        bwd_ok = False
        print("  [FAIL] MinOp backward sample 1")
    # Sample 2: tie → Q1 selected (<=) → grad_Q1=-1, grad_Q2=0
    if Float64(grad_in_arr[4]) != -1.0 or Float64(grad_in_arr[5]) != 0.0:
        bwd_ok = False
        print("  [FAIL] MinOp backward sample 2")
    # Sample 3: Q1 selected → grad_Q1=-1, grad_Q2=0
    if Float64(grad_in_arr[6]) != -1.0 or Float64(grad_in_arr[7]) != 0.0:
        bwd_ok = False
        print("  [FAIL] MinOp backward sample 3")

    if bwd_ok:
        print("  [PASS] MinOp backward: gradient routing correct")


def main() raises:
    print("=== RSampleOp / MinOp Tests ===")
    test_rsample_op_shapes()
    test_rsample_op_forward()
    test_rsample_op_backward()
    test_min_op_shapes()
    test_min_op_forward_backward()
    print("=== All tests passed ===")
