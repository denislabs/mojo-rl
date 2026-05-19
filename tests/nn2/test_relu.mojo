"""ReLU[DIM] CPU tests — Phase 1 (cache-internal trait).

Covers:
  - forward: output = max(0, input); internal cache = input
  - backward: grad_input = grad_output * (input > 0)
  - x == 0 edge: gradient is 0
  - for_each_param yields zero params
"""

from std.memory import alloc
from std.testing import assert_equal
from layout import TileTensor, TensorLayout, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.core import ParamVisitor
from mojo_rl.nn2.primitives.relu import ReLU


struct CountVisitor(ParamVisitor):
    var visits: Int

    def __init__(out self):
        self.visits = 0

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
        self.visits += 1


def test_forward() raises:
    """max(0, x). Covers negative, zero, and positive inputs."""
    comptime DIM = 4
    comptime BATCH = 2

    var relu = ReLU[DIM].make["cpu", INIT=Zero]()

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    in_buf[0] = -1.5
    in_buf[1] =  0.0
    in_buf[2] =  0.5
    in_buf[3] =  2.0
    in_buf[4] = -3.0
    in_buf[5] =  4.0
    in_buf[6] = -0.1
    in_buf[7] =  0.0
    for k in range(BATCH * DIM):
        out_buf[k] = -999.0

    var input = TileTensor(in_buf, row_major[BATCH, DIM]())
    var output = TileTensor(out_buf, row_major[BATCH, DIM]())

    relu.forward["cpu", BATCH](input, output)

    assert_equal(output[0, 0], 0.0)
    assert_equal(output[0, 1], 0.0)
    assert_equal(output[0, 2], 0.5)
    assert_equal(output[0, 3], 2.0)
    assert_equal(output[1, 0], 0.0)
    assert_equal(output[1, 1], 4.0)
    assert_equal(output[1, 2], 0.0)
    assert_equal(output[1, 3], 0.0)

    # Internal cache mirrors input
    var cache = TileTensor(relu.cache, row_major[BATCH, DIM]())
    for b in range(BATCH):
        for d in range(DIM):
            assert_equal(cache[b, d], input[b, d])

    in_buf.free()
    out_buf.free()
    print("  test_forward PASSED")


def test_backward() raises:
    """grad_input = grad_output where input > 0, else 0. Includes x==0
    edge case (gradient is 0)."""
    comptime DIM = 3
    comptime BATCH = 2

    var relu = ReLU[DIM].make["cpu", INIT=Zero]()

    # Pre-populate internal cache (simulates a forward with these inputs)
    relu.cache.resize(BATCH * DIM, 0.0)
    var cache_view = TileTensor(relu.cache, row_major[BATCH, DIM]())
    cache_view[0, 0] = -1.0
    cache_view[0, 1] =  0.0
    cache_view[0, 2] =  2.0
    cache_view[1, 0] =  3.0
    cache_view[1, 1] =  0.5
    cache_view[1, 2] = -0.1

    var go_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    go_buf[0] = 10.0
    go_buf[1] = 20.0
    go_buf[2] = 30.0
    go_buf[3] = 40.0
    go_buf[4] = 50.0
    go_buf[5] = 60.0
    var gi_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        gi_buf[k] = -999.0

    var grad_out = TileTensor(go_buf, row_major[BATCH, DIM]())
    var grad_in  = TileTensor(gi_buf, row_major[BATCH, DIM]())

    relu.backward["cpu", BATCH](grad_out, grad_in)

    assert_equal(grad_in[0, 0], 0.0)
    assert_equal(grad_in[0, 1], 0.0)
    assert_equal(grad_in[0, 2], 30.0)
    assert_equal(grad_in[1, 0], 40.0)
    assert_equal(grad_in[1, 1], 50.0)
    assert_equal(grad_in[1, 2], 0.0)

    go_buf.free()
    gi_buf.free()
    print("  test_backward PASSED")


def test_for_each_param() raises:
    var relu = ReLU[16].make["cpu", INIT=Zero]()
    var v = CountVisitor()
    relu.for_each_param["cpu"](String("act0"), v)
    assert_equal(v.visits, 0)
    print("  test_for_each_param PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 ReLU unit tests (CPU, Phase 1, cache-internal trait)")
    print("=" * 60)
    test_forward()
    test_backward()
    test_for_each_param()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
