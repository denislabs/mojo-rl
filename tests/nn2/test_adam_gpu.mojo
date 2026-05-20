"""AdamGPU parity test vs CPU Adam.

Same model, same hand-set gradient, run one Adam step on each, verify
weights match.
"""

from std.math import abs as fabs
from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.optimizer import Adam


def test_adam_gpu_parity() raises:
    comptime IN = 4
    comptime OUT = 6

    var ctx = DeviceContext()

    # Build matching Linear layers
    var lin_cpu = Linear[IN, OUT].make["cpu", INIT=Zero]()
    var lin_gpu = Linear[IN, OUT].make["gpu", INIT=Zero](ctx)

    # Set identical weights on both
    var w_host = ctx.enqueue_create_host_buffer[DT](IN * OUT)
    var b_host = ctx.enqueue_create_host_buffer[DT](OUT)
    ctx.synchronize()
    for i in range(IN):
        for j in range(OUT):
            w_host.unsafe_ptr()[i * OUT + j] = Scalar[DT](
                0.1 * Float32(i + 1) - 0.05 * Float32(j)
            )
    for j in range(OUT):
        b_host.unsafe_ptr()[j] = Scalar[DT](0.01 * Float32(j + 1))

    var w_cpu = TileTensor(lin_cpu.weight, row_major[IN, OUT]())
    var b_cpu = TileTensor(lin_cpu.bias, row_major[OUT]())
    for i in range(IN):
        for j in range(OUT):
            w_cpu[i, j] = w_host.unsafe_ptr()[i * OUT + j]
    for j in range(OUT):
        b_cpu[j] = b_host.unsafe_ptr()[j]
    ctx.enqueue_copy(lin_gpu.weight_dev.value(), w_host)
    ctx.enqueue_copy(lin_gpu.bias_dev.value(), b_host)

    # Set identical gradients
    var gw_host = ctx.enqueue_create_host_buffer[DT](IN * OUT)
    var gb_host = ctx.enqueue_create_host_buffer[DT](OUT)
    ctx.synchronize()
    for i in range(IN):
        for j in range(OUT):
            gw_host.unsafe_ptr()[i * OUT + j] = Scalar[DT](
                0.2 * Float32(i + 1) + 0.1 * Float32(j) - 0.3
            )
    for j in range(OUT):
        gb_host.unsafe_ptr()[j] = Scalar[DT](0.5 - 0.1 * Float32(j))

    var gw_cpu = TileTensor(lin_cpu.grad_w, row_major[IN, OUT]())
    var gb_cpu = TileTensor(lin_cpu.grad_b, row_major[OUT]())
    for i in range(IN):
        for j in range(OUT):
            gw_cpu[i, j] = gw_host.unsafe_ptr()[i * OUT + j]
    for j in range(OUT):
        gb_cpu[j] = gb_host.unsafe_ptr()[j]
    ctx.enqueue_copy(lin_gpu.grad_w_dev.value(), gw_host)
    ctx.enqueue_copy(lin_gpu.grad_b_dev.value(), gb_host)

    # Construct optimizers
    var adam_cpu = Adam.make["cpu"](lin_cpu, lr=0.01)
    var adam_gpu = Adam.make["gpu"](lin_gpu, ctx, lr=0.01)

    # One step on each
    adam_cpu.step["cpu"](lin_cpu)
    adam_gpu.step["gpu"](lin_gpu)

    # Copy GPU weights back
    var w_after_host = ctx.enqueue_create_host_buffer[DT](IN * OUT)
    var b_after_host = ctx.enqueue_create_host_buffer[DT](OUT)
    ctx.enqueue_copy(w_after_host, lin_gpu.weight_dev.value())
    ctx.enqueue_copy(b_after_host, lin_gpu.bias_dev.value())
    ctx.synchronize()

    # Compare
    var max_w_diff: Scalar[DT] = 0.0
    var w_after_cpu = TileTensor(lin_cpu.weight, row_major[IN, OUT]())
    for i in range(IN):
        for j in range(OUT):
            var d = fabs(w_after_cpu[i, j] - w_after_host.unsafe_ptr()[i * OUT + j])
            if d > max_w_diff: max_w_diff = d
    var max_b_diff: Scalar[DT] = 0.0
    var b_after_cpu = TileTensor(lin_cpu.bias, row_major[OUT]())
    for j in range(OUT):
        var d = fabs(b_after_cpu[j] - b_after_host.unsafe_ptr()[j])
        if d > max_b_diff: max_b_diff = d
    print("max-diff weight =", max_w_diff)
    print("max-diff bias   =", max_b_diff)
    assert_true(max_w_diff < Scalar[DT](1e-6),
        "weight parity failed: " + String(max_w_diff))
    assert_true(max_b_diff < Scalar[DT](1e-6),
        "bias parity failed: " + String(max_b_diff))

    print("  test_adam_gpu_parity PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 AdamGPU parity vs Adam (CPU)")
    print("=" * 60)
    test_adam_gpu_parity()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
