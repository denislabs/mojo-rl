"""Tanh[DIM] CPU tests — Phase 5.1.

Covers:
  - forward: output = tanh(input); internal cache = output (y, not x)
  - backward: grad_input = grad_output * (1 - y^2) where y = cache
  - for_each_param yields zero params

Reference values (tanh):
  tanh(0)   = 0
  tanh(1)   ≈ 0.7615941559557649
  tanh(0.5) ≈ 0.46211715726000974
  tanh(-1)  ≈ -0.7615941559557649
  tanh(2)   ≈ 0.9640275800758169
"""

from std.math import abs as fabs
from std.memory import alloc
from std.testing import assert_equal, assert_true
from layout import TileTensor, TensorLayout, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.core import ParamVisitor
from mojo_rl.nn2.primitives.tanh import Tanh


struct CountVisitor(ParamVisitor):
    var visits: Int

    def __init__(out self):
        self.visits = 0

    def visit(
        mut self,
        name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int,
        apply_decay: Bool,
        ) raises:
        self.visits += 1


def test_forward() raises:
    """Tanh(x) for canonical values."""
    comptime DIM = 5
    comptime BATCH = 1
    comptime TOL: Scalar[DT] = 1e-5

    var t = Tanh[DIM].make["cpu", INIT=Zero]()

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    in_buf[0] =  0.0
    in_buf[1] =  1.0
    in_buf[2] =  0.5
    in_buf[3] = -1.0
    in_buf[4] =  2.0
    for k in range(BATCH * DIM):
        out_buf[k] = -999.0

    var input = TileTensor(in_buf, row_major[BATCH, DIM]())
    var output = TileTensor(out_buf, row_major[BATCH, DIM]())

    t.forward["cpu", BATCH](input, output)

    assert_true(fabs(output[0, 0] - 0.0) < TOL,
        "tanh(0): " + String(output[0, 0]))
    assert_true(fabs(output[0, 1] - 0.7615941559557649) < TOL,
        "tanh(1): " + String(output[0, 1]))
    assert_true(fabs(output[0, 2] - 0.46211715726000974) < TOL,
        "tanh(0.5): " + String(output[0, 2]))
    assert_true(fabs(output[0, 3] - (-0.7615941559557649)) < TOL,
        "tanh(-1): " + String(output[0, 3]))
    assert_true(fabs(output[0, 4] - 0.9640275800758169) < TOL,
        "tanh(2): " + String(output[0, 4]))

    # Cache must mirror output (y), not input (x). This matters for backward.
    var cache = TileTensor(t.cache, row_major[BATCH, DIM]())
    for d in range(DIM):
        assert_equal(cache[0, d], output[0, d])

    in_buf.free()
    out_buf.free()
    print("  test_forward PASSED")


def test_backward() raises:
    """Grad_input = grad_output * (1 - y^2). Hand-compute against known y."""
    comptime DIM = 3
    comptime BATCH = 2
    comptime TOL: Scalar[DT] = 1e-5

    var t = Tanh[DIM].make["cpu", INIT=Zero]()

    # Seed the cache with known y values (no forward needed for this test).
    t.cache.resize(BATCH * DIM, 0.0)
    var cache_view = TileTensor(t.cache, row_major[BATCH, DIM]())
    cache_view[0, 0] = 0.0        # 1 - 0   = 1.0
    cache_view[0, 1] = 0.5        # 1 - .25 = 0.75
    cache_view[0, 2] = 0.8        # 1 - .64 = 0.36
    cache_view[1, 0] = -0.5       # 1 - .25 = 0.75
    cache_view[1, 1] = 1.0        # 1 - 1   = 0.0  (saturated, no flow)
    cache_view[1, 2] = -0.9       # 1 - .81 = 0.19

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

    t.backward["cpu", BATCH](grad_out, grad_in)

    assert_true(fabs(grad_in[0, 0] - 10.0)  < TOL, "(1.0)*10:  " + String(grad_in[0, 0]))
    assert_true(fabs(grad_in[0, 1] - 15.0)  < TOL, "(0.75)*20: " + String(grad_in[0, 1]))
    assert_true(fabs(grad_in[0, 2] - 10.8)  < TOL, "(0.36)*30: " + String(grad_in[0, 2]))
    assert_true(fabs(grad_in[1, 0] - 30.0)  < TOL, "(0.75)*40: " + String(grad_in[1, 0]))
    assert_true(fabs(grad_in[1, 1] - 0.0)   < TOL, "(0.0)*50:  " + String(grad_in[1, 1]))
    assert_true(fabs(grad_in[1, 2] - 11.4)  < TOL, "(0.19)*60: " + String(grad_in[1, 2]))

    go_buf.free()
    gi_buf.free()
    print("  test_backward PASSED")


def test_for_each_param() raises:
    var t = Tanh[16].make["cpu", INIT=Zero]()
    var v = CountVisitor()
    t.for_each_param["cpu"](String("tanh0"), v)
    assert_equal(v.visits, 0)
    print("  test_for_each_param PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 Tanh unit tests (CPU, Phase 5.1)")
    print("=" * 60)
    test_forward()
    test_backward()
    test_for_each_param()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
