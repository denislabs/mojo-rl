"""Initializer + `.make[INIT]` factory tests — Linear and ReLU on CPU + GPU.

Verifies that:
  - `Linear[IN, OUT].make["cpu", INIT=Kaiming]()` produces nonzero weights within the
    expected sqrt(6/fan_in) bound, with bias all zero.
  - `Linear[IN, OUT].make["gpu", INIT=Kaiming](ctx)` round-trips weights to
    device memory correctly.
  - `ReLU[DIM].make[INIT]()` and the GPU variant construct cleanly
    (INIT is ignored — parameterless layer).
"""

from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.initializer import Kaiming, Zero


def test_linear_make_cpu() raises:
    var lin = Linear[4, 6].make["cpu", INIT=Kaiming]()
    var w = TileTensor(lin.weight, row_major[4, 6]())
    # Kaiming gives nonzero weights bounded by sqrt(6/4) ≈ 1.2247
    var has_nonzero: Bool = False
    var max_abs: Scalar[DT] = 0.0
    for i in range(4):
        for j in range(6):
            if abs(w[i, j]) > 0.0:
                has_nonzero = True
            if abs(w[i, j]) > max_abs:
                max_abs = abs(w[i, j])
    assert_true(has_nonzero, "Kaiming produced all-zero weights")
    assert_true(max_abs < Scalar[DT](1.3), "Kaiming weights out of expected bound")
    # Bias should be all zeros
    for j in range(6):
        assert_true(lin.bias[j] == Scalar[DT](0.0), "bias not zero")
    print("  test_linear_make_cpu PASSED (max_abs=" + String(max_abs) + ")")


def test_linear_make_gpu() raises:
    var ctx = DeviceContext()
    var lin = Linear[4, 6].make["gpu", INIT=Kaiming](ctx)
    # Copy weights back and inspect
    var w_host = ctx.enqueue_create_host_buffer[DT](24)
    var b_host = ctx.enqueue_create_host_buffer[DT](6)
    ctx.enqueue_copy(w_host, lin.weight_dev.value())
    ctx.enqueue_copy(b_host, lin.bias_dev.value())
    ctx.synchronize()
    var has_nonzero: Bool = False
    for i in range(24):
        if w_host.unsafe_ptr()[i] != Scalar[DT](0.0):
            has_nonzero = True
    assert_true(has_nonzero, "GPU Kaiming produced all-zero weights")
    for j in range(6):
        assert_true(b_host.unsafe_ptr()[j] == Scalar[DT](0.0),
            "GPU bias not zero")
    print("  test_linear_make_gpu PASSED")


def test_relu_make() raises:
    var r1 = ReLU[8].make["cpu", INIT=Kaiming]()                          # CPU, INIT ignored
    var ctx = DeviceContext()
    var r2 = ReLU[8].make["gpu", INIT=Kaiming](ctx)                # GPU
    print("  test_relu_make PASSED")


def main() raises:
    seed(42)
    print("=" * 60)
    print("nn2 make[INIT] spike")
    print("=" * 60)
    test_linear_make_cpu()
    test_linear_make_gpu()
    test_relu_make()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
