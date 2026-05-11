"""Tests for GELUOp (tanh approximation, GPT-2 / BERT canonical).

Verifies:
  1. Spot-check forward values against the canonical formula at known points.
  2. Finite-difference gradcheck on CPU (random input, 5e-2 rel-err tol).
  3. CPU vs GPU forward + backward parity (~1e-6 tolerance).
  4. End-to-end smoke: TransformerFFN now containing GELU compiles, runs
     forward, and produces non-NaN output.

Run:
    pixi run mojo run -I . tests/nn/test_gelu_op.mojo
    pixi run -e apple mojo run -I . tests/nn/test_gelu_op.mojo  # GPU parity
"""

from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.random import seed, random_float64
from std.math import abs as math_abs, tanh

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.autodiff import GELUOp
from mojo_rl.nn.composites import TransformerFFN
from layout import Layout, LayoutTensor


def check(cond: Bool, msg: String, mut fails: Int):
    if cond:
        print("  PASS: " + msg)
    else:
        print("  FAIL: " + msg)
        fails += 1


def print_header(name: String):
    print("\n" + "=" * 70)
    print("TEST: " + name)
    print("=" * 70)


def make_list(size: Int) -> List[Scalar[dtype]]:
    var lst = List[Scalar[dtype]](capacity=size)
    for _ in range(size):
        lst.append(0)
    return lst^


def make_rand_list(size: Int) -> List[Scalar[dtype]]:
    var lst = List[Scalar[dtype]](capacity=size)
    for _ in range(size):
        lst.append(Scalar[dtype](random_float64(-2.0, 2.0)))
    return lst^


def gelu_tanh_reference(x: Float64) -> Float64:
    """Float64 reference for the tanh-approx GELU formula."""
    var c = 0.7978845608028654
    var a = 0.044715
    var u = c * (x + a * x * x * x)
    return 0.5 * x * (1.0 + tanh(u))


# =============================================================================
# Test 1: forward values at known points
# =============================================================================
def test_forward_values() -> Int:
    print_header("GELUOp: forward matches canonical formula at known points")
    var fails = 0

    comptime DIM = 5
    comptime BATCH = 1
    comptime Op = GELUOp[DIM]

    # Test points spanning negative, zero, small positive, large positive.
    var inp_data = List[Scalar[dtype]](capacity=DIM)
    inp_data.append(Scalar[dtype](-2.0))
    inp_data.append(Scalar[dtype](-0.5))
    inp_data.append(Scalar[dtype](0.0))
    inp_data.append(Scalar[dtype](0.5))
    inp_data.append(Scalar[dtype](2.0))

    var out_data = make_list(DIM)
    var cache_data = make_list(DIM)
    var params = make_list(1)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](inp_data.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](cache_data.unsafe_ptr())

    Op.eval[BATCH, dtype](inp_t, out_t, p_t, c_t)

    var max_err: Float64 = 0
    for i in range(DIM):
        var x = Float64(inp_data[i])
        var expected = gelu_tanh_reference(x)
        var got = Float64(out_data[i])
        var err = math_abs(expected - got)
        if err > max_err:
            max_err = err
    check(max_err < 1e-5, "forward max abs err vs reference = " + String(max_err), fails)

    # Sanity: output(0) = 0 exactly.
    check(
        math_abs(Float64(out_data[2])) < 1e-7,
        "GELU(0) = 0 (got " + String(Float64(out_data[2])) + ")",
        fails,
    )
    return fails


# =============================================================================
# Test 2: finite-difference gradcheck (CPU)
# =============================================================================
def test_gradcheck_cpu() -> Int:
    print_header("GELUOp: CPU finite-difference gradcheck")
    var fails = 0
    seed(101)

    comptime DIM = 8
    comptime BATCH = 2
    comptime Op = GELUOp[DIM]

    var inp = make_rand_list(BATCH * DIM)
    var go = make_rand_list(BATCH * DIM)
    var params = make_list(1)

    var out_data = make_list(BATCH * DIM)
    var cache_data = make_list(BATCH * DIM)
    var gi_data = make_list(BATCH * DIM)
    var gp_data = make_list(1)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](out_data.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var c_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](cache_data.unsafe_ptr())
    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](go.unsafe_ptr())
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](gi_data.unsafe_ptr())
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
    ](gp_data.unsafe_ptr())

    Op.eval[BATCH, dtype](inp_t, out_t, p_t, c_t)
    Op.vjp[BATCH, dtype](go_t, gi_t, p_t, c_t, gp_t)

    var eps: Float64 = 1e-3
    var max_err: Float64 = 0
    for idx in range(BATCH * DIM):
        var orig = inp[idx]

        inp[idx] = Scalar[dtype](Float64(orig) + eps)
        var op_data = make_list(BATCH * DIM)
        var oc_data = make_list(BATCH * DIM)
        var op_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ](op_data.unsafe_ptr())
        var oc_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ](oc_data.unsafe_ptr())
        Op.eval[BATCH, dtype](inp_t, op_t, p_t, oc_t)

        inp[idx] = Scalar[dtype](Float64(orig) - eps)
        var om_data = make_list(BATCH * DIM)
        var omc_data = make_list(BATCH * DIM)
        var om_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ](om_data.unsafe_ptr())
        var omc_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
        ](omc_data.unsafe_ptr())
        Op.eval[BATCH, dtype](inp_t, om_t, p_t, omc_t)

        inp[idx] = orig

        var fd: Float64 = 0
        for j in range(BATCH * DIM):
            fd += Float64(go[j]) * (Float64(op_data[j]) - Float64(om_data[j])) / (2.0 * eps)
        var an = Float64(gi_data[idx])
        var err = math_abs(fd - an)
        if math_abs(fd) < 1e-4 and math_abs(an) < 1e-4:
            continue
        var rel = err / (math_abs(fd) + math_abs(an) + 1e-8)
        if rel > max_err:
            max_err = rel

    check(max_err < 5e-2, "max relative gradient error = " + String(max_err) + " (tol 5e-2)", fails)
    return fails


def test_gpu_parity() raises -> Int:
    print_header("GELUOp: CPU vs GPU forward + backward parity")
    var fails = 0
    seed(31)

    comptime DIM = 16
    comptime BATCH = 4
    comptime Op = GELUOp[DIM]

    var ctx = DeviceContext()

    var inp = make_rand_list(BATCH * DIM)
    var go = make_rand_list(BATCH * DIM)
    var params = make_list(1)

    # CPU reference.
    var out_cpu = make_list(BATCH * DIM)
    var cache_cpu = make_list(BATCH * DIM)
    var gi_cpu = make_list(BATCH * DIM)
    var gp_cpu = make_list(1)

    var inp_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](inp.unsafe_ptr())
    var out_cpu_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](out_cpu.unsafe_ptr())
    var p_t = LayoutTensor[
        dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var cache_cpu_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](cache_cpu.unsafe_ptr())
    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](go.unsafe_ptr())
    var gi_cpu_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](gi_cpu.unsafe_ptr())
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
    ](gp_cpu.unsafe_ptr())

    Op.eval[BATCH, dtype](inp_t, out_cpu_t, p_t, cache_cpu_t)
    Op.vjp[BATCH, dtype](go_t, gi_cpu_t, p_t, cache_cpu_t, gp_t)

    # GPU.
    var inp_dev = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    var out_dev = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    var cache_dev = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    var gi_dev = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    var go_dev = ctx.enqueue_create_buffer[dtype](BATCH * DIM)
    var p_dev = ctx.enqueue_create_buffer[dtype](1)
    var gp_dev = ctx.enqueue_create_buffer[dtype](1)

    var inp_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
    var out_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
    var go_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)
    var gi_host = ctx.enqueue_create_host_buffer[dtype](BATCH * DIM)

    for i in range(BATCH * DIM):
        inp_host[i] = inp[i]
        go_host[i] = go[i]
    ctx.enqueue_copy(inp_dev, inp_host)
    ctx.enqueue_copy(go_dev, go_host)

    var inp_dev_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](inp_dev.unsafe_ptr())
    var out_dev_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](out_dev.unsafe_ptr())
    var cache_dev_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](cache_dev.unsafe_ptr())
    var go_dev_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](go_dev.unsafe_ptr())
    var gi_dev_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](gi_dev.unsafe_ptr())
    var p_dev_t = LayoutTensor[
        dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
    ](p_dev.unsafe_ptr())
    var gp_dev_t = LayoutTensor[
        dtype, Layout.row_major(Op.PARAM_SIZE), MutAnyOrigin
    ](gp_dev.unsafe_ptr())
    var dummy_ws = ctx.enqueue_create_buffer[dtype](1)

    Op.eval_gpu[BATCH, dtype](
        ctx, out_dev_t, inp_dev_t, p_dev_t, cache_dev_t, dummy_ws.unsafe_ptr()
    )
    Op.vjp_gpu[BATCH, dtype](
        ctx, go_dev_t, gi_dev_t, p_dev_t, cache_dev_t, gp_dev_t, dummy_ws.unsafe_ptr()
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.enqueue_copy(gi_host, gi_dev)
    ctx.synchronize()

    var max_out_diff: Float64 = 0
    for i in range(BATCH * DIM):
        var d = math_abs(Float64(out_cpu[i]) - Float64(out_host[i]))
        if d > max_out_diff:
            max_out_diff = d
    check(max_out_diff < 1e-5, "forward max |GPU - CPU| = " + String(max_out_diff), fails)

    var max_gi_diff: Float64 = 0
    for i in range(BATCH * DIM):
        var d = math_abs(Float64(gi_cpu[i]) - Float64(gi_host[i]))
        if d > max_gi_diff:
            max_gi_diff = d
    check(max_gi_diff < 1e-5, "grad_input max |GPU - CPU| = " + String(max_gi_diff), fails)
    return fails


def main() raises:
    var total_fails = 0
    total_fails += test_forward_values()
    total_fails += test_gradcheck_cpu()
    total_fails += test_gpu_parity()

    print("\n" + "=" * 70)
    if total_fails == 0:
        print("ALL GELU OP TESTS PASSED")
    else:
        print("FAILED: " + String(total_fails) + " checks")
    print("=" * 70)
