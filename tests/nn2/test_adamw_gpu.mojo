"""AdamWGPU parity test vs CPU AdamW — Phase 4.

Verifies CPU and GPU AdamW produce identical updates after multiple steps,
covering both apply_decay=True (weight) and apply_decay=False (bias) paths.

The device-side step prep kernel (`_adamw_step_prep_kernel`) on each
`step()` call mutates the bc_dev / step_dev buffers and the per-param
update kernel reads from them — the multi-step loop verifies that
state machinery is correct.
"""

from std.math import abs as fabs
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.optimizer import AdamW


def test_adamw_gpu_parity() raises:
    comptime IN = 4
    comptime OUT = 6
    comptime N_STEPS = 5

    var ctx = DeviceContext()

    var lin_cpu = Linear[IN, OUT].make["cpu", INIT=Zero]()
    var lin_gpu = Linear[IN, OUT].make["gpu", INIT=Zero](ctx)

    # Identical weights + biases.
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

    # Identical gradients (re-applied each step — easier to control than
    # running a full forward/backward and isolates the optimizer math).
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

    var adam_cpu = AdamW.make_with_wd["cpu"](
        lin_cpu, lr=0.01, weight_decay=0.05,
    )
    var adam_gpu = AdamW.make_with_wd["gpu"](
        lin_gpu, ctx, lr=0.01, weight_decay=0.05,
    )

    # Sanity: GPU also recorded the apply_decay table correctly.
    assert_true(adam_gpu.apply_decay[0], "GPU expected weight.apply_decay=True")
    assert_true(not adam_gpu.apply_decay[1], "GPU expected bias.apply_decay=False")

    for step_i in range(N_STEPS):
        # Re-set the identical gradients on both sides for each step.
        var gw_cpu = TileTensor(lin_cpu.grad_w, row_major[IN, OUT]())
        var gb_cpu = TileTensor(lin_cpu.grad_b, row_major[OUT]())
        for i in range(IN):
            for j in range(OUT):
                gw_cpu[i, j] = gw_host.unsafe_ptr()[i * OUT + j]
        for j in range(OUT):
            gb_cpu[j] = gb_host.unsafe_ptr()[j]
        ctx.enqueue_copy(lin_gpu.grad_w_dev.value(), gw_host)
        ctx.enqueue_copy(lin_gpu.grad_b_dev.value(), gb_host)

        adam_cpu.step["cpu"](lin_cpu)
        adam_gpu.step["gpu"](lin_gpu)

    var w_after_host = ctx.enqueue_create_host_buffer[DT](IN * OUT)
    var b_after_host = ctx.enqueue_create_host_buffer[DT](OUT)
    ctx.enqueue_copy(w_after_host, lin_gpu.weight_dev.value())
    ctx.enqueue_copy(b_after_host, lin_gpu.bias_dev.value())
    ctx.synchronize()

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
    print("after " + String(N_STEPS) + " AdamW steps:")
    print("  max-diff weight = " + String(max_w_diff))
    print("  max-diff bias   = " + String(max_b_diff))
    assert_true(max_w_diff < Scalar[DT](1e-5),
        "weight parity failed: " + String(max_w_diff))
    assert_true(max_b_diff < Scalar[DT](1e-5),
        "bias parity failed: " + String(max_b_diff))
    print("  test_adamw_gpu_parity PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 AdamWGPU parity vs AdamW (CPU)")
    print("=" * 60)
    test_adamw_gpu_parity()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
