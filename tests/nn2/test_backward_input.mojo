"""Phase 8.2 — `backward_input` semantics tests.

Covers:
  - Linear.backward_input grad_input parity vs Linear.backward grad_input
  - Linear.backward_input leaves grad_w / grad_b untouched (zero on a
    freshly-zeroed layer; preserved on a pre-loaded grad)
  - LayerNorm.backward_input grad_input parity + grad_gamma/grad_beta
    untouched
  - Sequential[Linear, ReLU, Linear].backward_input chains correctly:
    grad_input parity vs Sequential.backward, all inner grad_w zero
  - StopGradParams[Linear].backward routes to inner.backward_input:
    grad_input parity, grad_w stays zero
  - StopGradParams[Linear].forward is true passthrough
"""

from std.memory import alloc
from std.testing import assert_almost_equal, assert_equal
from layout import TileTensor, TensorLayout, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.layer_norm import LayerNorm
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential, StopGradParams
from mojo_rl.nn2.initializer import Xavier, Zero


def _fill_input(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int):
    for i in range(n):
        p[i] = Scalar[DT](0.1 + 0.13 * Float32(i % 17) - 0.07 * Float32(i % 5))


def _fill_grad(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int):
    for i in range(n):
        p[i] = Scalar[DT](-0.2 + 0.05 * Float32(i % 9) + 0.03 * Float32(i % 3))


# ──────────────────────────────────────────────────────────────────────────
# Linear.backward_input
# ──────────────────────────────────────────────────────────────────────────


def test_linear_backward_input_parity() raises:
    """Grad_input from `backward_input` matches grad_input from `backward`."""
    comptime IN = 5
    comptime OUT = 4
    comptime BATCH = 3

    var lin_a = Linear[IN, OUT].make[target="cpu", INIT=Xavier]()
    var lin_b = Linear[IN, OUT].make[target="cpu", INIT=Zero]()
    # Copy weights so both layers behave identically.
    for k in range(IN * OUT):
        lin_b.weight[k] = lin_a.weight[k]
    for k in range(OUT):
        lin_b.bias[k] = lin_a.bias[k]

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var go_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var gi_a_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var gi_b_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)

    _fill_input(in_buf, BATCH * IN)
    _fill_grad(go_buf, BATCH * OUT)
    for k in range(BATCH * IN):
        gi_a_buf[k] = 0.0
        gi_b_buf[k] = 0.0

    var input = TileTensor(in_buf, row_major[BATCH, IN]())
    var output_a = TileTensor(out_buf, row_major[BATCH, OUT]())
    var output_b = TileTensor(out_buf, row_major[BATCH, OUT]())

    # Forward to populate caches identically on both.
    lin_a.forward["cpu", BATCH](input, output_a)
    lin_b.forward["cpu", BATCH](input, output_b)

    var grad_output = TileTensor(go_buf, row_major[BATCH, OUT]())
    var grad_input_a = TileTensor(gi_a_buf, row_major[BATCH, IN]())
    var grad_input_b = TileTensor(gi_b_buf, row_major[BATCH, IN]())

    lin_a.backward["cpu", BATCH](grad_output, grad_input_a)
    lin_b.backward_input["cpu", BATCH](grad_output, grad_input_b)

    for k in range(BATCH * IN):
        assert_almost_equal(gi_a_buf[k], gi_b_buf[k], atol=1e-5)

    # Critical assertion: backward_input must NOT touch grad_w / grad_b.
    for k in range(IN * OUT):
        assert_equal(lin_b.grad_w[k], Scalar[DT](0.0))
    for k in range(OUT):
        assert_equal(lin_b.grad_b[k], Scalar[DT](0.0))

    # And the reference layer's grad_w / grad_b should be non-zero.
    var saw_nonzero_gw: Bool = False
    for k in range(IN * OUT):
        if lin_a.grad_w[k] != 0.0:
            saw_nonzero_gw = True
            break
    if not saw_nonzero_gw:
        raise Error("test setup degenerate: lin_a.grad_w all zero")

    in_buf.free(); out_buf.free(); go_buf.free()
    gi_a_buf.free(); gi_b_buf.free()
    print("  test_linear_backward_input_parity PASSED")


def test_linear_backward_input_preserves_existing_grad() raises:
    """Pre-load grad_w / grad_b with non-zero sentinels; verify
    backward_input does not change them."""
    comptime IN = 3
    comptime OUT = 2
    comptime BATCH = 2

    var lin = Linear[IN, OUT].make[target="cpu", INIT=Xavier]()

    # Sentinel-load grads.
    for k in range(IN * OUT):
        lin.grad_w[k] = Scalar[DT](7.7)
    for k in range(OUT):
        lin.grad_b[k] = Scalar[DT](-3.3)

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var go_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var gi_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    _fill_input(in_buf, BATCH * IN)
    _fill_grad(go_buf, BATCH * OUT)

    var input  = TileTensor(in_buf,  row_major[BATCH, IN]())
    var output = TileTensor(out_buf, row_major[BATCH, OUT]())
    lin.forward["cpu", BATCH](input, output)

    var grad_output = TileTensor(go_buf, row_major[BATCH, OUT]())
    var grad_input  = TileTensor(gi_buf, row_major[BATCH, IN]())
    lin.backward_input["cpu", BATCH](grad_output, grad_input)

    for k in range(IN * OUT):
        assert_equal(lin.grad_w[k], Scalar[DT](7.7))
    for k in range(OUT):
        assert_equal(lin.grad_b[k], Scalar[DT](-3.3))

    in_buf.free(); out_buf.free(); go_buf.free(); gi_buf.free()
    print("  test_linear_backward_input_preserves_existing_grad PASSED")


# ──────────────────────────────────────────────────────────────────────────
# LayerNorm.backward_input
# ──────────────────────────────────────────────────────────────────────────


def test_layer_norm_backward_input_parity() raises:
    comptime DIM = 6
    comptime BATCH = 3

    var ln_a = LayerNorm[DIM].make[target="cpu", INIT=Xavier]()
    var ln_b = LayerNorm[DIM].make[target="cpu", INIT=Zero]()
    for k in range(DIM):
        ln_b.gamma[k] = ln_a.gamma[k]
        ln_b.beta[k]  = ln_a.beta[k]

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var go_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_a_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_b_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    _fill_input(in_buf, BATCH * DIM)
    _fill_grad(go_buf, BATCH * DIM)
    for k in range(BATCH * DIM):
        gi_a_buf[k] = 0.0
        gi_b_buf[k] = 0.0

    var input = TileTensor(in_buf, row_major[BATCH, DIM]())
    var out_a = TileTensor(out_buf, row_major[BATCH, DIM]())
    var out_b = TileTensor(out_buf, row_major[BATCH, DIM]())
    ln_a.forward["cpu", BATCH](input, out_a)
    ln_b.forward["cpu", BATCH](input, out_b)

    var grad_output = TileTensor(go_buf, row_major[BATCH, DIM]())
    var gi_a = TileTensor(gi_a_buf, row_major[BATCH, DIM]())
    var gi_b = TileTensor(gi_b_buf, row_major[BATCH, DIM]())
    ln_a.backward["cpu", BATCH](grad_output, gi_a)
    ln_b.backward_input["cpu", BATCH](grad_output, gi_b)

    for k in range(BATCH * DIM):
        assert_almost_equal(gi_a_buf[k], gi_b_buf[k], atol=1e-5)

    for k in range(DIM):
        assert_equal(ln_b.grad_gamma[k], Scalar[DT](0.0))
        assert_equal(ln_b.grad_beta[k],  Scalar[DT](0.0))

    in_buf.free(); out_buf.free(); go_buf.free()
    gi_a_buf.free(); gi_b_buf.free()
    print("  test_layer_norm_backward_input_parity PASSED")


# ──────────────────────────────────────────────────────────────────────────
# Sequential.backward_input — chained across N children.
# ──────────────────────────────────────────────────────────────────────────


def test_sequential_backward_input_chain() raises:
    """Sequential[Linear, ReLU, Linear].backward_input: grad_input matches
    backward; both Linears' grad_w stay zero."""
    comptime IN = 4
    comptime HID = 6
    comptime OUT = 3
    comptime BATCH = 2

    comptime Net = Sequential[Linear[IN, HID], ReLU[HID], Linear[HID, OUT]]

    var net_a = Net.make[target="cpu", INIT=Xavier]()
    var net_b = Net.make[target="cpu", INIT=Zero]()
    # Copy weights net_a -> net_b across all three children.
    for k in range(IN * HID):
        net_b.children[0].weight[k] = net_a.children[0].weight[k]
    for k in range(HID):
        net_b.children[0].bias[k] = net_a.children[0].bias[k]
    for k in range(HID * OUT):
        net_b.children[2].weight[k] = net_a.children[2].weight[k]
    for k in range(OUT):
        net_b.children[2].bias[k] = net_a.children[2].bias[k]

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var go_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var gi_a_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var gi_b_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    _fill_input(in_buf, BATCH * IN)
    _fill_grad(go_buf, BATCH * OUT)
    for k in range(BATCH * IN):
        gi_a_buf[k] = 0.0; gi_b_buf[k] = 0.0

    var input = TileTensor(in_buf, row_major[BATCH, IN]())
    var out_a = TileTensor(out_buf, row_major[BATCH, OUT]())
    var out_b = TileTensor(out_buf, row_major[BATCH, OUT]())
    net_a.forward["cpu", BATCH](input, out_a)
    net_b.forward["cpu", BATCH](input, out_b)

    var grad_output = TileTensor(go_buf, row_major[BATCH, OUT]())
    var gi_a = TileTensor(gi_a_buf, row_major[BATCH, IN]())
    var gi_b = TileTensor(gi_b_buf, row_major[BATCH, IN]())
    net_a.backward["cpu", BATCH](grad_output, gi_a)
    net_b.backward_input["cpu", BATCH](grad_output, gi_b)

    for k in range(BATCH * IN):
        assert_almost_equal(gi_a_buf[k], gi_b_buf[k], atol=1e-5)
    # Inner Linears in net_b must have zero grads.
    for k in range(IN * HID):
        assert_equal(net_b.children[0].grad_w[k], Scalar[DT](0.0))
    for k in range(HID):
        assert_equal(net_b.children[0].grad_b[k], Scalar[DT](0.0))
    for k in range(HID * OUT):
        assert_equal(net_b.children[2].grad_w[k], Scalar[DT](0.0))
    for k in range(OUT):
        assert_equal(net_b.children[2].grad_b[k], Scalar[DT](0.0))

    in_buf.free(); out_buf.free(); go_buf.free()
    gi_a_buf.free(); gi_b_buf.free()
    print("  test_sequential_backward_input_chain PASSED")


# ──────────────────────────────────────────────────────────────────────────
# StopGradParams[Linear] — forward passthrough, backward → backward_input.
# ──────────────────────────────────────────────────────────────────────────


def test_stop_grad_params_forward_passthrough() raises:
    comptime IN = 4
    comptime OUT = 3
    comptime BATCH = 2

    var lin = Linear[IN, OUT].make[target="cpu", INIT=Xavier]()

    # Build a wrapper that owns a *copy* of lin (we can't share — Move
    # semantics). For this test we make a fresh wrapper and re-copy.
    var lin_for_wrap = Linear[IN, OUT].make[target="cpu", INIT=Zero]()
    for k in range(IN * OUT):
        lin_for_wrap.weight[k] = lin.weight[k]
    for k in range(OUT):
        lin_for_wrap.bias[k] = lin.bias[k]
    var frozen = StopGradParams[Linear[IN, OUT]](lin_for_wrap^)

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var out_a_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var out_b_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    _fill_input(in_buf, BATCH * IN)

    var input = TileTensor(in_buf, row_major[BATCH, IN]())
    var out_a = TileTensor(out_a_buf, row_major[BATCH, OUT]())
    var out_b = TileTensor(out_b_buf, row_major[BATCH, OUT]())

    lin.forward["cpu", BATCH](input, out_a)
    frozen.forward["cpu", BATCH](input, out_b)

    for k in range(BATCH * OUT):
        assert_almost_equal(out_a_buf[k], out_b_buf[k], atol=1e-6)

    in_buf.free(); out_a_buf.free(); out_b_buf.free()
    print("  test_stop_grad_params_forward_passthrough PASSED")


def test_stop_grad_params_backward_routes_to_backward_input() raises:
    """StopGradParams[Linear].backward must match Linear.backward_input
    on grad_input AND leave Inner.grad_w / grad_b at zero."""
    comptime IN = 5
    comptime OUT = 4
    comptime BATCH = 3

    var lin_ref = Linear[IN, OUT].make[target="cpu", INIT=Xavier]()
    var inner = Linear[IN, OUT].make[target="cpu", INIT=Zero]()
    for k in range(IN * OUT):
        inner.weight[k] = lin_ref.weight[k]
    for k in range(OUT):
        inner.bias[k] = lin_ref.bias[k]
    var frozen = StopGradParams[Linear[IN, OUT]](inner^)

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var out_a_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var out_b_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var go_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var gi_a_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var gi_b_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    _fill_input(in_buf, BATCH * IN)
    _fill_grad(go_buf, BATCH * OUT)
    for k in range(BATCH * IN):
        gi_a_buf[k] = 0.0; gi_b_buf[k] = 0.0

    var input = TileTensor(in_buf, row_major[BATCH, IN]())
    var out_a = TileTensor(out_a_buf, row_major[BATCH, OUT]())
    var out_b = TileTensor(out_b_buf, row_major[BATCH, OUT]())
    lin_ref.forward["cpu", BATCH](input, out_a)
    frozen.forward["cpu", BATCH](input, out_b)

    var grad_output = TileTensor(go_buf, row_major[BATCH, OUT]())
    var gi_a = TileTensor(gi_a_buf, row_major[BATCH, IN]())
    var gi_b = TileTensor(gi_b_buf, row_major[BATCH, IN]())
    lin_ref.backward_input["cpu", BATCH](grad_output, gi_a)
    frozen.backward["cpu", BATCH](grad_output, gi_b)

    for k in range(BATCH * IN):
        assert_almost_equal(gi_a_buf[k], gi_b_buf[k], atol=1e-5)
    for k in range(IN * OUT):
        assert_equal(frozen.inner.grad_w[k], Scalar[DT](0.0))
    for k in range(OUT):
        assert_equal(frozen.inner.grad_b[k], Scalar[DT](0.0))

    in_buf.free(); out_a_buf.free(); out_b_buf.free(); go_buf.free()
    gi_a_buf.free(); gi_b_buf.free()
    print("  test_stop_grad_params_backward_routes_to_backward_input PASSED")


def main() raises:
    print("=" * 70)
    print("Phase 8.2 backward_input + StopGradParams[Inner] tests")
    print("=" * 70)
    test_linear_backward_input_parity()
    test_linear_backward_input_preserves_existing_grad()
    test_layer_norm_backward_input_parity()
    test_sequential_backward_input_chain()
    test_stop_grad_params_forward_passthrough()
    test_stop_grad_params_backward_routes_to_backward_input()
    print("ALL PASSED")
