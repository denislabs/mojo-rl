"""LinearGPU backward parity test — vs CPU Linear with identical state.

Strategy:
  - Construct Linear[IN, OUT] (CPU) and LinearGPU[IN, OUT].
  - Set identical weights, biases on both.
  - Run forward on both with the same input (populates each layer's
    internal cache).
  - Run backward on both with the same grad_output.
  - Verify max-abs-diff(cpu_grad_in, gpu_grad_in) <= 1e-4
  - Verify max-abs-diff(cpu_grad_w, gpu_grad_w) <= 1e-4
  - Verify max-abs-diff(cpu_grad_b, gpu_grad_b) <= 1e-4

Run:
    pixi run -e apple  mojo run -I . tests/nn2/test_linear_gpu_backward.mojo
"""

from std.math import abs as fabs
from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.linear import Linear


def test_linear_gpu_backward_parity() raises:
    comptime IN = 4
    comptime OUT = 6
    comptime BATCH = 3

    var ctx = DeviceContext()
    var lin_cpu = Linear[IN, OUT].make["cpu", INIT=Zero]()
    var lin_gpu = Linear[IN, OUT].make["gpu", INIT=Zero](ctx)

    # ── Identical hand-picked weights and bias on both ────────────────
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

    # ── Input batch ───────────────────────────────────────────────────
    var in_host = ctx.enqueue_create_host_buffer[DT](BATCH * IN)
    ctx.synchronize()
    for b in range(BATCH):
        for i in range(IN):
            in_host.unsafe_ptr()[b * IN + i] = Scalar[DT](
                0.5 * Float32(b) - 0.2 * Float32(i)
            )

    var in_buf_cpu:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var out_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    for k in range(BATCH * IN):
        in_buf_cpu[k] = in_host.unsafe_ptr()[k]
    var input_cpu  = TileTensor(in_buf_cpu,  row_major[BATCH, IN]())
    var output_cpu = TileTensor(out_buf_cpu, row_major[BATCH, OUT]())

    var in_dev  = ctx.enqueue_create_buffer[DT](BATCH * IN)
    var out_dev = ctx.enqueue_create_buffer[DT](BATCH * OUT)
    ctx.enqueue_copy(in_dev, in_host)
    var input_gpu  = TileTensor(in_dev,  row_major[BATCH, IN]())
    var output_gpu = TileTensor(out_dev, row_major[BATCH, OUT]())

    # ── Forward (populates each layer's cache) ────────────────────────
    lin_cpu.forward["cpu", BATCH](input_cpu, output_cpu)
    lin_gpu.forward["gpu", BATCH](input_gpu, output_gpu)

    # ── grad_output: distinct values per (b, j) ───────────────────────
    var go_host = ctx.enqueue_create_host_buffer[DT](BATCH * OUT)
    ctx.synchronize()
    for b in range(BATCH):
        for j in range(OUT):
            go_host.unsafe_ptr()[b * OUT + j] = Scalar[DT](
                0.3 * Float32(b + 1) + 0.1 * Float32(j)
            )

    var go_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var gi_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    for k in range(BATCH * OUT):
        go_buf_cpu[k] = go_host.unsafe_ptr()[k]
    var grad_out_cpu = TileTensor(go_buf_cpu, row_major[BATCH, OUT]())
    var grad_in_cpu  = TileTensor(gi_buf_cpu, row_major[BATCH, IN]())

    var go_dev = ctx.enqueue_create_buffer[DT](BATCH * OUT)
    var gi_dev = ctx.enqueue_create_buffer[DT](BATCH * IN)
    ctx.enqueue_copy(go_dev, go_host)
    var grad_out_gpu = TileTensor(go_dev, row_major[BATCH, OUT]())
    var grad_in_gpu  = TileTensor(gi_dev, row_major[BATCH, IN]())

    # ── Backward on both ──────────────────────────────────────────────
    lin_cpu.backward["cpu", BATCH](grad_out_cpu, grad_in_cpu)
    lin_gpu.backward["gpu", BATCH](grad_out_gpu, grad_in_gpu)

    # ── Copy GPU results back ─────────────────────────────────────────
    var gi_host_out = ctx.enqueue_create_host_buffer[DT](BATCH * IN)
    var gw_host_out = ctx.enqueue_create_host_buffer[DT](IN * OUT)
    var gb_host_out = ctx.enqueue_create_host_buffer[DT](OUT)
    ctx.enqueue_copy(gi_host_out, gi_dev)
    ctx.enqueue_copy(gw_host_out, lin_gpu.grad_w_dev.value())
    ctx.enqueue_copy(gb_host_out, lin_gpu.grad_b_dev.value())
    ctx.synchronize()

    # ── Compare grad_input ────────────────────────────────────────────
    var max_gi: Scalar[DT] = 0.0
    for b in range(BATCH):
        for i in range(IN):
            var diff = fabs(grad_in_cpu[b, i] - gi_host_out.unsafe_ptr()[b * IN + i])
            if diff > max_gi: max_gi = diff
    print("max-diff grad_input  = " + String(max_gi))
    assert_true(max_gi < Scalar[DT](1e-4),
        "grad_input parity failed: " + String(max_gi))

    # ── Compare grad_w ────────────────────────────────────────────────
    var gw_cpu_view = TileTensor(lin_cpu.grad_w, row_major[IN, OUT]())
    var max_gw: Scalar[DT] = 0.0
    for i in range(IN):
        for j in range(OUT):
            var diff = fabs(gw_cpu_view[i, j] - gw_host_out.unsafe_ptr()[i * OUT + j])
            if diff > max_gw: max_gw = diff
    print("max-diff grad_weight = " + String(max_gw))
    assert_true(max_gw < Scalar[DT](1e-4),
        "grad_weight parity failed: " + String(max_gw))

    # ── Compare grad_b ────────────────────────────────────────────────
    var gb_cpu_view = TileTensor(lin_cpu.grad_b, row_major[OUT]())
    var max_gb: Scalar[DT] = 0.0
    for j in range(OUT):
        var diff = fabs(gb_cpu_view[j] - gb_host_out.unsafe_ptr()[j])
        if diff > max_gb: max_gb = diff
    print("max-diff grad_bias   = " + String(max_gb))
    assert_true(max_gb < Scalar[DT](1e-5),
        "grad_bias parity failed: " + String(max_gb))

    in_buf_cpu.free()
    out_buf_cpu.free()
    go_buf_cpu.free()
    gi_buf_cpu.free()
    print("  test_linear_gpu_backward_parity PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 LinearGPU backward parity test (CPU vs GPU)")
    print("=" * 60)
    test_linear_gpu_backward_parity()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
