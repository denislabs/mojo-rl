"""Gradient check for LogStdBoundOp and DivideOp.

Both are zero-parameter DiffOps, so gradcheck targets the *input* gradient
(grad_input) via finite differences against the analytical vjp.
"""

from std.math import abs as _abs
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.autodiff import LogStdBoundOp, DivideOp


def _abs_f64(x: Float64) -> Float64:
    return -x if x < 0.0 else x


def _input_grad_check_log_std_bound[
    DIM: Int, BATCH: Int, MIN_NUM: Int, MIN_DEN: Int, MAX_NUM: Int, MAX_DEN: Int
](
    input_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad_output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    eps_fd: Float64 = 1e-3,
) -> Tuple[Float64, Float64, Int]:
    """Compare analytic grad_input vs finite-difference for LogStdBoundOp.

    Returns (max_abs_err, max_rel_err, num_checked).
    """
    comptime Op = LogStdBoundOp[DIM, MIN_NUM, MIN_DEN, MAX_NUM, MAX_DEN]
    comptime IN = Op.IN_DIM
    comptime OUT = Op.OUT_DIM
    comptime CACHE = Op.CACHE_SIZE

    var input_t = LayoutTensor[dtype, Layout.row_major(BATCH, IN), MutAnyOrigin](
        input_ptr
    )
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
    ](grad_output_ptr)

    # Forward
    var output_arr = InlineArray[Scalar[dtype], BATCH * OUT](uninitialized=True)
    var cache_arr = InlineArray[Scalar[dtype], BATCH * CACHE](uninitialized=True)
    var params_arr = InlineArray[Scalar[dtype], 1](uninitialized=True)
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
    ](output_arr.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, CACHE), MutAnyOrigin
    ](cache_arr.unsafe_ptr())
    var params_t = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](params_arr.unsafe_ptr())
    Op.eval[BATCH, dtype](input_t, output_t, params_t, cache_t)

    # Analytical grad_input
    var grad_in_arr = InlineArray[Scalar[dtype], BATCH * IN](uninitialized=True)
    for k in range(BATCH * IN):
        grad_in_arr[k] = Scalar[dtype](0.0)
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](grad_in_arr.unsafe_ptr())
    var grad_params_arr = InlineArray[Scalar[dtype], 1](uninitialized=True)
    var grad_params_t = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](grad_params_arr.unsafe_ptr())
    Op.vjp[BATCH, dtype](
        grad_out_t, grad_in_t, params_t, cache_t, grad_params_t
    )

    # FD grad_input: for each input element, perturb and recompute output.
    var max_abs: Float64 = 0.0
    var max_rel: Float64 = 0.0
    var num_checked: Int = 0
    for b in range(BATCH):
        for i in range(IN):
            var orig = input_ptr[b * IN + i]

            # f(x + eps)
            input_ptr[b * IN + i] = orig + Scalar[dtype](eps_fd)
            var out_plus = InlineArray[Scalar[dtype], BATCH * OUT](
                uninitialized=True
            )
            var cache_plus = InlineArray[Scalar[dtype], BATCH * CACHE](
                uninitialized=True
            )
            var out_plus_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
            ](out_plus.unsafe_ptr())
            var cache_plus_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, CACHE), MutAnyOrigin
            ](cache_plus.unsafe_ptr())
            Op.eval[BATCH, dtype](
                input_t, out_plus_t, params_t, cache_plus_t
            )

            # f(x - eps)
            input_ptr[b * IN + i] = orig - Scalar[dtype](eps_fd)
            var out_minus = InlineArray[Scalar[dtype], BATCH * OUT](
                uninitialized=True
            )
            var cache_minus = InlineArray[Scalar[dtype], BATCH * CACHE](
                uninitialized=True
            )
            var out_minus_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
            ](out_minus.unsafe_ptr())
            var cache_minus_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, CACHE), MutAnyOrigin
            ](cache_minus.unsafe_ptr())
            Op.eval[BATCH, dtype](
                input_t, out_minus_t, params_t, cache_minus_t
            )
            input_ptr[b * IN + i] = orig

            # Numerical = sum_j go_j * (f_plus_j - f_minus_j) / (2*eps)
            var num_grad: Float64 = 0.0
            for j in range(BATCH * OUT):
                var go = Float64(grad_output_ptr[j])
                num_grad += go * Float64(
                    out_plus[j] - out_minus[j]
                ) / (2.0 * eps_fd)

            var ana = Float64(grad_in_arr[b * IN + i])
            var err = _abs_f64(ana - num_grad)
            var denom = _abs_f64(ana) + _abs_f64(num_grad)
            var rel: Float64 = 0.0
            if denom > 1e-5:
                rel = err / denom
            if err > max_abs:
                max_abs = err
            if rel > max_rel:
                max_rel = rel
            num_checked += 1

    return (max_abs, max_rel, num_checked)


def _input_grad_check_divide[
    DIM: Int, BATCH: Int
](
    input_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad_output_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    eps_fd: Float64 = 1e-3,
) -> Tuple[Float64, Float64, Int]:
    comptime Op = DivideOp[DIM, 1e-8]
    comptime IN = Op.IN_DIM
    comptime OUT = Op.OUT_DIM
    comptime CACHE = Op.CACHE_SIZE

    var input_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](input_ptr)
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
    ](grad_output_ptr)

    var output_arr = InlineArray[Scalar[dtype], BATCH * OUT](uninitialized=True)
    var cache_arr = InlineArray[Scalar[dtype], BATCH * CACHE](uninitialized=True)
    var params_arr = InlineArray[Scalar[dtype], 1](uninitialized=True)
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
    ](output_arr.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, CACHE), MutAnyOrigin
    ](cache_arr.unsafe_ptr())
    var params_t = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](params_arr.unsafe_ptr())
    Op.eval[BATCH, dtype](input_t, output_t, params_t, cache_t)

    var grad_in_arr = InlineArray[Scalar[dtype], BATCH * IN](uninitialized=True)
    for k in range(BATCH * IN):
        grad_in_arr[k] = Scalar[dtype](0.0)
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](grad_in_arr.unsafe_ptr())
    var grad_params_arr = InlineArray[Scalar[dtype], 1](uninitialized=True)
    var grad_params_t = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](grad_params_arr.unsafe_ptr())
    Op.vjp[BATCH, dtype](
        grad_out_t, grad_in_t, params_t, cache_t, grad_params_t
    )

    var max_abs: Float64 = 0.0
    var max_rel: Float64 = 0.0
    var num_checked: Int = 0
    for b in range(BATCH):
        for i in range(IN):
            var orig = input_ptr[b * IN + i]

            input_ptr[b * IN + i] = orig + Scalar[dtype](eps_fd)
            var out_plus = InlineArray[Scalar[dtype], BATCH * OUT](
                uninitialized=True
            )
            var cache_plus = InlineArray[Scalar[dtype], BATCH * CACHE](
                uninitialized=True
            )
            var out_plus_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
            ](out_plus.unsafe_ptr())
            var cache_plus_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, CACHE), MutAnyOrigin
            ](cache_plus.unsafe_ptr())
            Op.eval[BATCH, dtype](
                input_t, out_plus_t, params_t, cache_plus_t
            )

            input_ptr[b * IN + i] = orig - Scalar[dtype](eps_fd)
            var out_minus = InlineArray[Scalar[dtype], BATCH * OUT](
                uninitialized=True
            )
            var cache_minus = InlineArray[Scalar[dtype], BATCH * CACHE](
                uninitialized=True
            )
            var out_minus_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
            ](out_minus.unsafe_ptr())
            var cache_minus_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, CACHE), MutAnyOrigin
            ](cache_minus.unsafe_ptr())
            Op.eval[BATCH, dtype](
                input_t, out_minus_t, params_t, cache_minus_t
            )
            input_ptr[b * IN + i] = orig

            var num_grad: Float64 = 0.0
            for j in range(BATCH * OUT):
                var go = Float64(grad_output_ptr[j])
                num_grad += go * Float64(
                    out_plus[j] - out_minus[j]
                ) / (2.0 * eps_fd)

            var ana = Float64(grad_in_arr[b * IN + i])
            var err = _abs_f64(ana - num_grad)
            var denom = _abs_f64(ana) + _abs_f64(num_grad)
            var rel: Float64 = 0.0
            if denom > 1e-5:
                rel = err / denom
            if err > max_abs:
                max_abs = err
            if rel > max_rel:
                max_rel = rel
            num_checked += 1

    return (max_abs, max_rel, num_checked)


def test_log_std_bound_gradcheck() raises:
    """LogStdBoundOp: input gradcheck for log_std_min=-10, log_std_max=2."""
    comptime DIM = 4
    comptime BATCH = 2

    var input_arr = InlineArray[Scalar[dtype], BATCH * DIM](uninitialized=True)
    # Mix of small / negative / positive / saturated raw values
    input_arr[0] = 0.0
    input_arr[1] = 1.0
    input_arr[2] = -1.0
    input_arr[3] = 2.5
    input_arr[4] = -3.0
    input_arr[5] = 0.5
    input_arr[6] = 0.25
    input_arr[7] = -0.5

    var grad_out_arr = InlineArray[Scalar[dtype], BATCH * DIM](uninitialized=True)
    grad_out_arr[0] = 1.3
    grad_out_arr[1] = -0.5
    grad_out_arr[2] = 0.7
    grad_out_arr[3] = 2.1
    grad_out_arr[4] = -1.1
    grad_out_arr[5] = 0.4
    grad_out_arr[6] = 0.6
    grad_out_arr[7] = -0.9

    var (max_abs, max_rel, n) = _input_grad_check_log_std_bound[
        DIM, BATCH, -10, 1, 2, 1
    ](
        input_arr.unsafe_ptr(),
        grad_out_arr.unsafe_ptr(),
        eps_fd=1e-3,
    )
    print("LogStdBoundOp gradcheck:", n, "params, max_abs=", max_abs,
          " max_rel=", max_rel)
    if max_rel > 1e-3:
        raise Error("LogStdBoundOp gradcheck failed (max_rel > 1e-3)")


def test_divide_gradcheck() raises:
    """DivideOp: input gradcheck for both grad_a and grad_b paths."""
    comptime DIM = 3
    comptime BATCH = 2

    var input_arr = InlineArray[Scalar[dtype], BATCH * 2 * DIM](
        uninitialized=True
    )
    # a values
    input_arr[0 * 2 * DIM + 0] = 6.0
    input_arr[0 * 2 * DIM + 1] = -4.0
    input_arr[0 * 2 * DIM + 2] = 1.5
    input_arr[1 * 2 * DIM + 0] = 1.5
    input_arr[1 * 2 * DIM + 1] = -2.0
    input_arr[1 * 2 * DIM + 2] = 9.0
    # b values (avoid near-zero denominators that would explode FD)
    input_arr[0 * 2 * DIM + DIM + 0] = 2.0
    input_arr[0 * 2 * DIM + DIM + 1] = 4.0
    input_arr[0 * 2 * DIM + DIM + 2] = 5.0
    input_arr[1 * 2 * DIM + DIM + 0] = 3.0
    input_arr[1 * 2 * DIM + DIM + 1] = -1.5
    input_arr[1 * 2 * DIM + DIM + 2] = 8.0

    var grad_out_arr = InlineArray[Scalar[dtype], BATCH * DIM](uninitialized=True)
    grad_out_arr[0] = 1.3
    grad_out_arr[1] = -0.5
    grad_out_arr[2] = 0.7
    grad_out_arr[3] = 2.1
    grad_out_arr[4] = -1.1
    grad_out_arr[5] = 0.4

    var (max_abs, max_rel, n) = _input_grad_check_divide[DIM, BATCH](
        input_arr.unsafe_ptr(),
        grad_out_arr.unsafe_ptr(),
        eps_fd=1e-3,
    )
    print("DivideOp gradcheck:", n, "params, max_abs=", max_abs,
          " max_rel=", max_rel)
    if max_rel > 1e-3:
        raise Error("DivideOp gradcheck failed (max_rel > 1e-3)")


def main() raises:
    test_log_std_bound_gradcheck()
    test_divide_gradcheck()
    print("OK — LogStdBoundOp and DivideOp pass CPU input gradcheck.")
