"""Linear[IN, OUT] CPU tests — Phase 1 (cache-internal trait).

Covers:
  - forward: hand-set weights + bias, verify output[b, j] == expected,
    and verify internal cache mirrors input
  - backward grad_input: analytical against hand-computed values
  - backward grad_w / grad_b accumulation: verify += semantics
  - param-tree walk yields {"weight", "bias"} with correct sizes
  - zero_grad clears accumulators

Tests construct TileTensor views over local pointer buffers (input,
output, grad_output, grad_input) for the I/O tensors. Internal layer
state (weight, bias, grad_w, grad_b, cache) is accessed via TileTensor
views over the layer's owned Lists.

For backward-in-isolation tests we pre-populate the layer's internal
cache directly via `lin.cache.resize(...)` + TileTensor view.
"""

from std.memory import alloc
from std.testing import assert_equal
from layout import TileTensor, TensorLayout, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import ParamVisitor
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.initializer import Zero


# ──────────────────────────────────────────────────────────────────────────
# CountVisitor — records visit order for the param-walk test.
# ──────────────────────────────────────────────────────────────────────────

struct CountVisitor(ParamVisitor):
    var names: List[String]
    var sizes: List[Int]

    def __init__(out self):
        self.names = List[String]()
        self.sizes = List[Int]()

    def visit[
        L: TensorLayout, OP: MutOrigin, OG: MutOrigin,
    ](
        mut self,
        name: String,
        param: TileTensor[DT, L, OP],
        grad: TileTensor[DT, L, OG],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        self.names.append(name)
        self.sizes.append(n_elems)


# ──────────────────────────────────────────────────────────────────────────
# test_forward
# ──────────────────────────────────────────────────────────────────────────

def test_forward() raises:
    """Hand-set weights + bias; check output[b, j] = bias[j] + sum_i x[b,i] * w[i,j].
    Verify internal cache mirrors input after forward."""
    comptime IN = 2
    comptime OUT = 3
    comptime BATCH = 1

    var lin = Linear[IN, OUT].make[target="cpu", INIT=Zero]()
    var w = TileTensor(lin.weight, row_major[IN, OUT]())
    var b = TileTensor(lin.bias,   row_major[OUT]())

    # weight = [[1, 2, 3], [4, 5, 6]]
    w[0, 0] = 1.0
    w[0, 1] = 2.0
    w[0, 2] = 3.0
    w[1, 0] = 4.0
    w[1, 1] = 5.0
    w[1, 2] = 6.0
    # bias = [10, 20, 30]
    b[0] = 10.0
    b[1] = 20.0
    b[2] = 30.0

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    in_buf[0] = 1.0
    in_buf[1] = 1.0
    for k in range(BATCH * OUT):
        out_buf[k] = -999.0

    var input = TileTensor(in_buf, row_major[BATCH, IN]())
    var output = TileTensor(out_buf, row_major[BATCH, OUT]())

    lin.forward["cpu", BATCH](input, output)

    # Expected: out[0,0]=15, out[0,1]=27, out[0,2]=39
    assert_equal(output[0, 0], 15.0)
    assert_equal(output[0, 1], 27.0)
    assert_equal(output[0, 2], 39.0)

    # Internal cache should mirror input
    var cache = TileTensor(lin.cache, row_major[BATCH, IN]())
    assert_equal(cache[0, 0], 1.0)
    assert_equal(cache[0, 1], 1.0)

    in_buf.free()
    out_buf.free()
    print("  test_forward PASSED")


# ──────────────────────────────────────────────────────────────────────────
# test_backward
# ──────────────────────────────────────────────────────────────────────────

def test_backward() raises:
    """Hand-set grad_output; check grad_input, grad_w, grad_b.
    Pre-populate the layer's internal cache directly to test backward
    in isolation."""
    comptime IN = 2
    comptime OUT = 3
    comptime BATCH = 2

    var lin = Linear[IN, OUT].make[target="cpu", INIT=Zero]()
    var w = TileTensor(lin.weight, row_major[IN, OUT]())
    w[0, 0] = 1.0
    w[0, 1] = 2.0
    w[0, 2] = 3.0
    w[1, 0] = 4.0
    w[1, 1] = 5.0
    w[1, 2] = 6.0

    # Pre-populate cache = [[1, 1], [2, 3]] (mimics what forward would have written).
    lin.cache.resize(BATCH * IN, 0.0)
    var cache_view = TileTensor(lin.cache, row_major[BATCH, IN]())
    cache_view[0, 0] = 1.0
    cache_view[0, 1] = 1.0
    cache_view[1, 0] = 2.0
    cache_view[1, 1] = 3.0

    # grad_output = [[1, 1, 1], [1, 1, 1]]
    var go_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    for k in range(BATCH * OUT):
        go_buf[k] = 1.0
    var gi_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    for k in range(BATCH * IN):
        gi_buf[k] = -999.0

    var grad_out = TileTensor(go_buf, row_major[BATCH, OUT]())
    var grad_in = TileTensor(gi_buf, row_major[BATCH, IN]())

    lin.backward["cpu", BATCH](grad_out, grad_in)

    # grad_input
    assert_equal(grad_in[0, 0], 6.0)
    assert_equal(grad_in[0, 1], 15.0)
    assert_equal(grad_in[1, 0], 6.0)
    assert_equal(grad_in[1, 1], 15.0)

    # grad_w
    var gw = TileTensor(lin.grad_w, row_major[IN, OUT]())
    assert_equal(gw[0, 0], 3.0)
    assert_equal(gw[0, 1], 3.0)
    assert_equal(gw[0, 2], 3.0)
    assert_equal(gw[1, 0], 4.0)
    assert_equal(gw[1, 1], 4.0)
    assert_equal(gw[1, 2], 4.0)

    # grad_b
    var gb = TileTensor(lin.grad_b, row_major[OUT]())
    assert_equal(gb[0], 2.0)
    assert_equal(gb[1], 2.0)
    assert_equal(gb[2], 2.0)

    go_buf.free()
    gi_buf.free()
    print("  test_backward PASSED")


# ──────────────────────────────────────────────────────────────────────────
# test_grad_accumulation
# ──────────────────────────────────────────────────────────────────────────

def test_grad_accumulation() raises:
    """Backward must accumulate (+=) into grad_w/grad_b. Two consecutive
    backward() calls with identical (forward, grad_output) must double
    the gradient accumulator."""
    comptime IN = 2
    comptime OUT = 2
    comptime BATCH = 1

    var lin = Linear[IN, OUT].make[target="cpu", INIT=Zero]()
    var w = TileTensor(lin.weight, row_major[IN, OUT]())
    w[0, 0] = 1.0
    w[0, 1] = 0.0
    w[1, 0] = 0.0
    w[1, 1] = 1.0

    # Populate cache via a forward call (input = [[1, 1]]).
    var in_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    in_buf[0] = 1.0
    in_buf[1] = 1.0
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var input = TileTensor(in_buf, row_major[BATCH, IN]())
    var output = TileTensor(out_buf, row_major[BATCH, OUT]())
    lin.forward["cpu", BATCH](input, output)

    var go_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    go_buf[0] = 1.0
    go_buf[1] = 1.0
    var gi_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    gi_buf[0] = 0.0
    gi_buf[1] = 0.0
    var grad_out = TileTensor(go_buf, row_major[BATCH, OUT]())
    var grad_in = TileTensor(gi_buf, row_major[BATCH, IN]())

    lin.backward["cpu", BATCH](grad_out, grad_in)
    var gw_after_one = TileTensor(lin.grad_w, row_major[IN, OUT]())
    var gw00_after_one = gw_after_one[0, 0]
    var gb_after_one = TileTensor(lin.grad_b, row_major[OUT]())
    var gb0_after_one = gb_after_one[0]

    lin.backward["cpu", BATCH](grad_out, grad_in)
    var gw_after_two = TileTensor(lin.grad_w, row_major[IN, OUT]())
    var gb_after_two = TileTensor(lin.grad_b, row_major[OUT]())
    assert_equal(gw_after_two[0, 0], gw00_after_one * 2.0)
    assert_equal(gb_after_two[0], gb0_after_one * 2.0)

    in_buf.free()
    out_buf.free()
    go_buf.free()
    gi_buf.free()
    print("  test_grad_accumulation PASSED")


# ──────────────────────────────────────────────────────────────────────────
# test_zero_grad
# ──────────────────────────────────────────────────────────────────────────

def test_zero_grad() raises:
    """zero_grad() clears grad_w + grad_b to 0.0."""
    var lin = Linear[3, 2].make[target="cpu", INIT=Zero]()
    var gw = TileTensor(lin.grad_w, row_major[3, 2]())
    var gb = TileTensor(lin.grad_b, row_major[2]())
    for i in range(3):
        for j in range(2):
            gw[i, j] = Float32(i * 2 + j + 1)
    gb[0] = 7.0
    gb[1] = 8.0

    lin.zero_grad["cpu"]()

    var gw2 = TileTensor(lin.grad_w, row_major[3, 2]())
    var gb2 = TileTensor(lin.grad_b, row_major[2]())
    for i in range(3):
        for j in range(2):
            assert_equal(gw2[i, j], 0.0)
    for j in range(2):
        assert_equal(gb2[j], 0.0)
    print("  test_zero_grad PASSED")


# ──────────────────────────────────────────────────────────────────────────
# test_for_each_param
# ──────────────────────────────────────────────────────────────────────────

def test_for_each_param() raises:
    """Walk yields ("prefix.weight", W_SIZE) and ("prefix.bias", B_SIZE)."""
    var lin = Linear[4, 5].make[target="cpu", INIT=Zero]()
    var v = CountVisitor()
    lin.for_each_param["cpu"](String("layer0"), v)

    assert_equal(len(v.names), 2)
    assert_equal(v.names[0], String("layer0.weight"))
    assert_equal(v.names[1], String("layer0.bias"))
    assert_equal(v.sizes[0], 20)
    assert_equal(v.sizes[1], 5)

    var v2 = CountVisitor()
    lin.for_each_param["cpu"](String(""), v2)
    assert_equal(v2.names[0], String("weight"))
    assert_equal(v2.names[1], String("bias"))
    print("  test_for_each_param PASSED")


# ──────────────────────────────────────────────────────────────────────────
def main() raises:
    print("=" * 60)
    print("nn2 Linear unit tests (CPU, Phase 1, cache-internal trait)")
    print("=" * 60)
    test_forward()
    test_backward()
    test_grad_accumulation()
    test_zero_grad()
    test_for_each_param()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
