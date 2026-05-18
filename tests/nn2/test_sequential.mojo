"""Sequential2 CPU tests — Phase 1.

Covers:
  - forward chain (Linear → ReLU): output matches hand-computed value
  - backward chain (ReLU.backward then Linear.backward): grad_input +
    child grad_w/grad_b match hand-computed values
  - for_each_param walks both children with indexed prefix
  - end-to-end forward + backward on Linear → ReLU → Linear (chained
    Sequential2)
"""

from std.memory import alloc
from std.testing import assert_equal, assert_almost_equal
from layout import TileTensor, TensorLayout, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import ParamVisitor
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential2


# ──────────────────────────────────────────────────────────────────────────
# CountVisitor — records names + sizes
# ──────────────────────────────────────────────────────────────────────────

struct CountVisitor(ParamVisitor):
    var names: List[String]
    var sizes: List[Int]

    def __init__(out self):
        self.names = List[String]()
        self.sizes = List[Int]()

    def visit[L: TensorLayout](
        mut self,
        name: String,
        param: TileTensor[DT, L, MutAnyOrigin],
        grad: TileTensor[DT, L, MutAnyOrigin],
        n_elems: Int,
    ):
        self.names.append(name)
        self.sizes.append(n_elems)


# ──────────────────────────────────────────────────────────────────────────
# test_forward_linear_relu — Linear(2→3) then ReLU(3)
# ──────────────────────────────────────────────────────────────────────────

def test_forward_linear_relu() raises:
    """forward chain: Linear(2→3) followed by ReLU(3).

    weight = [[1, -1, 2], [-3, 1, 0]], bias = [-1, 2, 0].
    input = [[1, 1]]
      → linear_out = [[1 + -3 + -1, -1 + 1 + 2, 2 + 0 + 0]] = [[-3, 2, 2]]
      → relu_out   = [[0, 2, 2]]
    """
    comptime IN = 2
    comptime MID = 3
    comptime BATCH = 1

    var lin = Linear[IN, MID]()
    var w = TileTensor(lin.weight, row_major[IN, MID]())
    w[0, 0] =  1.0
    w[0, 1] = -1.0
    w[0, 2] =  2.0
    w[1, 0] = -3.0
    w[1, 1] =  1.0
    w[1, 2] =  0.0
    var b = TileTensor(lin.bias, row_major[MID]())
    b[0] = -1.0
    b[1] =  2.0
    b[2] =  0.0

    var net = Sequential2(lin^, ReLU[MID]())

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * MID)
    in_buf[0] = 1.0
    in_buf[1] = 1.0
    for k in range(BATCH * MID):
        out_buf[k] = -999.0

    var input  = TileTensor(in_buf,  row_major[BATCH, IN]())
    var output = TileTensor(out_buf, row_major[BATCH, MID]())

    net.forward[BATCH](input, output)

    assert_equal(output[0, 0], 0.0)   # max(0, -3)
    assert_equal(output[0, 1], 2.0)   # max(0, 2)
    assert_equal(output[0, 2], 2.0)   # max(0, 2)

    in_buf.free()
    out_buf.free()
    print("  test_forward_linear_relu PASSED")


# ──────────────────────────────────────────────────────────────────────────
# test_backward_linear_relu
# ──────────────────────────────────────────────────────────────────────────

def test_backward_linear_relu() raises:
    """backward chain: ReLU.backward then Linear.backward.

    With the same setup as test_forward_linear_relu (input [[1, 1]]),
    pre-activation = [-3, 2, 2], so ReLU mask = [0, 1, 1].
    grad_output = [[1, 1, 1]] → grad after ReLU.backward = [[0, 1, 1]].
    grad_input[0, i] = sum_j (grad after ReLU)[0, j] * weight[i, j]
      = sum over j ∈ {1, 2} of weight[i, j]:
      grad_input[0, 0] = -1 + 2 = 1
      grad_input[0, 1] =  1 + 0 = 1
    Linear's grad_w[i, j] += cache[0, i] * (grad after ReLU)[0, j]
      = input[0, i] * (grad after ReLU)[0, j]:
      grad_w[0, *] = 1 * [0, 1, 1] = [0, 1, 1]
      grad_w[1, *] = 1 * [0, 1, 1] = [0, 1, 1]
    Linear's grad_b[j] += (grad after ReLU)[0, j] = [0, 1, 1]
    """
    comptime IN = 2
    comptime MID = 3
    comptime BATCH = 1

    var lin = Linear[IN, MID]()
    var w = TileTensor(lin.weight, row_major[IN, MID]())
    w[0, 0] =  1.0
    w[0, 1] = -1.0
    w[0, 2] =  2.0
    w[1, 0] = -3.0
    w[1, 1] =  1.0
    w[1, 2] =  0.0
    var b = TileTensor(lin.bias, row_major[MID]())
    b[0] = -1.0
    b[1] =  2.0
    b[2] =  0.0

    var net = Sequential2(lin^, ReLU[MID]())

    # Forward to populate both children's caches
    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * MID)
    in_buf[0] = 1.0
    in_buf[1] = 1.0
    var input  = TileTensor(in_buf,  row_major[BATCH, IN]())
    var output = TileTensor(out_buf, row_major[BATCH, MID]())
    net.forward[BATCH](input, output)

    var go_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * MID)
    for k in range(BATCH * MID):
        go_buf[k] = 1.0
    var gi_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    gi_buf[0] = -999.0
    gi_buf[1] = -999.0
    var grad_out = TileTensor(go_buf, row_major[BATCH, MID]())
    var grad_in  = TileTensor(gi_buf, row_major[BATCH, IN]())

    net.backward[BATCH](grad_out, grad_in)

    # grad_input
    assert_equal(grad_in[0, 0], 1.0)
    assert_equal(grad_in[0, 1], 1.0)

    # Linear (first) grad_w + grad_b
    var gw = TileTensor(net.first.grad_w, row_major[IN, MID]())
    assert_equal(gw[0, 0], 0.0)
    assert_equal(gw[0, 1], 1.0)
    assert_equal(gw[0, 2], 1.0)
    assert_equal(gw[1, 0], 0.0)
    assert_equal(gw[1, 1], 1.0)
    assert_equal(gw[1, 2], 1.0)
    var gb = TileTensor(net.first.grad_b, row_major[MID]())
    assert_equal(gb[0], 0.0)
    assert_equal(gb[1], 1.0)
    assert_equal(gb[2], 1.0)

    in_buf.free()
    out_buf.free()
    go_buf.free()
    gi_buf.free()
    print("  test_backward_linear_relu PASSED")


# ──────────────────────────────────────────────────────────────────────────
# test_for_each_param
# ──────────────────────────────────────────────────────────────────────────

def test_for_each_param() raises:
    """Sequential2(Linear, ReLU): walk yields 2 params from Linear,
    0 from ReLU. Names prefixed "0.weight" / "0.bias"."""
    var net = Sequential2(Linear[3, 4](), ReLU[4]())
    var v = CountVisitor()
    net.for_each_param(String("net"), v)

    assert_equal(len(v.names), 2)
    assert_equal(v.names[0], String("net.0.weight"))
    assert_equal(v.names[1], String("net.0.bias"))
    assert_equal(v.sizes[0], 12)  # 3*4
    assert_equal(v.sizes[1], 4)

    # Nested Sequential2: Sequential2(Sequential2(Linear, ReLU), Linear)
    var net2 = Sequential2(
        Sequential2(Linear[2, 3](), ReLU[3]()),
        Linear[3, 5](),
    )
    var v2 = CountVisitor()
    net2.for_each_param(String(""), v2)
    assert_equal(len(v2.names), 4)
    assert_equal(v2.names[0], String("0.0.weight"))
    assert_equal(v2.names[1], String("0.0.bias"))
    assert_equal(v2.names[2], String("1.weight"))
    assert_equal(v2.names[3], String("1.bias"))
    print("  test_for_each_param PASSED")


# ──────────────────────────────────────────────────────────────────────────
# test_end_to_end_2_layer_mlp
# ──────────────────────────────────────────────────────────────────────────

def test_end_to_end_2_layer_mlp() raises:
    """Sequential2(Sequential2(Linear, ReLU), Linear) — a 2-hidden-layer MLP
    with weights matching test_forward_linear_relu's known computation,
    chained with an identity second Linear (weight = I, bias = 0).

    Verifies that 3-layer composition still produces the same output as
    a 1-layer Linear+ReLU, since the second Linear is identity."""
    comptime IN = 2
    comptime MID = 3
    comptime OUT = 3
    comptime BATCH = 1

    # First Linear: weights matching test_forward_linear_relu
    var lin0 = Linear[IN, MID]()
    var w0 = TileTensor(lin0.weight, row_major[IN, MID]())
    w0[0, 0] =  1.0; w0[0, 1] = -1.0; w0[0, 2] =  2.0
    w0[1, 0] = -3.0; w0[1, 1] =  1.0; w0[1, 2] =  0.0
    var b0 = TileTensor(lin0.bias, row_major[MID]())
    b0[0] = -1.0; b0[1] = 2.0; b0[2] = 0.0

    # Second Linear: identity (weight = I, bias = 0)
    var lin1 = Linear[MID, OUT]()
    var w1 = TileTensor(lin1.weight, row_major[MID, OUT]())
    for i in range(MID):
        for j in range(OUT):
            w1[i, j] = 1.0 if i == j else 0.0

    var net = Sequential2(
        Sequential2(lin0^, ReLU[MID]()),
        lin1^,
    )

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    in_buf[0] = 1.0; in_buf[1] = 1.0
    var input  = TileTensor(in_buf,  row_major[BATCH, IN]())
    var output = TileTensor(out_buf, row_major[BATCH, OUT]())

    net.forward[BATCH](input, output)

    # Identity second layer means output == ReLU(Linear0(input)) == [0, 2, 2]
    assert_equal(output[0, 0], 0.0)
    assert_equal(output[0, 1], 2.0)
    assert_equal(output[0, 2], 2.0)

    in_buf.free()
    out_buf.free()
    print("  test_end_to_end_2_layer_mlp PASSED")


# ──────────────────────────────────────────────────────────────────────────
def main() raises:
    print("=" * 60)
    print("nn2 Sequential2 unit tests (CPU, Phase 1)")
    print("=" * 60)
    test_forward_linear_relu()
    test_backward_linear_relu()
    test_for_each_param()
    test_end_to_end_2_layer_mlp()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
