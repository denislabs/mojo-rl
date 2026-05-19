"""LinearGPU forward parity test — vs CPU Linear with identical weights.

Strategy:
  - Construct Linear[IN, OUT] (CPU) and LinearGPU[IN, OUT].
  - Set identical hand-picked weights + biases on both.
  - Run forward on each with the same input.
  - Verify max-abs-diff(cpu_out, gpu_out) <= 1e-5 (fp32, should be ~ULP).

If this passes on Apple Metal + NVIDIA, the LinearGPU forward dispatch
is correct.

Run:
    pixi run -e apple  mojo run -I . tests/nn2/test_linear_gpu_forward.mojo
    pixi run -e nvidia mojo run -I . tests/nn2/test_linear_gpu_forward.mojo
"""

from std.math import abs as fabs
from std.memory import alloc
from std.testing import assert_true, assert_almost_equal
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.linear import Linear


def test_linear_gpu_forward_parity() raises:
    comptime IN = 4
    comptime OUT = 6
    comptime BATCH = 3

    # ── Build the two layers ──────────────────────────────────────────
    var ctx = DeviceContext()
    var lin_cpu = Linear[IN, OUT].make["cpu", INIT=Zero]()                  # default target="cpu"
    var lin_gpu = Linear[IN, OUT].make["gpu", INIT=Zero](ctx)        # unified Linear, GPU variant

    # ── Hand-pick deterministic weights/bias, write to both ──────────
    # Weight pattern: w[i, j] = 0.1 * (i + 1) - 0.05 * j  (varied, small)
    var w_host = ctx.enqueue_create_host_buffer[DT](IN * OUT)
    var b_host = ctx.enqueue_create_host_buffer[DT](OUT)
    ctx.synchronize()
    for i in range(IN):
        for j in range(OUT):
            var v: Scalar[DT] = 0.1 * Float32(i + 1) - 0.05 * Float32(j)
            w_host.unsafe_ptr()[i * OUT + j] = v
    for j in range(OUT):
        b_host.unsafe_ptr()[j] = Scalar[DT](0.01 * Float32(j + 1))

    # Mirror into the CPU layer via TileTensor views.
    var w_cpu = TileTensor(lin_cpu.weight, row_major[IN, OUT]())
    var b_cpu = TileTensor(lin_cpu.bias,   row_major[OUT]())
    for i in range(IN):
        for j in range(OUT):
            w_cpu[i, j] = w_host.unsafe_ptr()[i * OUT + j]
    for j in range(OUT):
        b_cpu[j] = b_host.unsafe_ptr()[j]

    # Upload to the GPU layer.
    ctx.enqueue_copy(lin_gpu.weight_dev.value(), w_host)
    ctx.enqueue_copy(lin_gpu.bias_dev.value(),   b_host)

    # ── Input batch ───────────────────────────────────────────────────
    var in_host = ctx.enqueue_create_host_buffer[DT](BATCH * IN)
    ctx.synchronize()
    for b in range(BATCH):
        for i in range(IN):
            in_host.unsafe_ptr()[b * IN + i] = Scalar[DT](0.5 * Float32(b) - 0.2 * Float32(i))

    # CPU input/output buffers
    var in_buf_cpu:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var out_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    for k in range(BATCH * IN):
        in_buf_cpu[k] = in_host.unsafe_ptr()[k]
    var input_cpu  = TileTensor(in_buf_cpu,  row_major[BATCH, IN]())
    var output_cpu = TileTensor(out_buf_cpu, row_major[BATCH, OUT]())

    # GPU input/output buffers
    var in_dev  = ctx.enqueue_create_buffer[DT](BATCH * IN)
    var out_dev = ctx.enqueue_create_buffer[DT](BATCH * OUT)
    ctx.enqueue_copy(in_dev, in_host)
    var input_gpu  = TileTensor(in_dev,  row_major[BATCH, IN]())
    var output_gpu = TileTensor(out_dev, row_major[BATCH, OUT]())

    # ── Forward on both ───────────────────────────────────────────────
    lin_cpu.forward["cpu", BATCH](input_cpu, output_cpu)
    lin_gpu.forward["gpu", BATCH](input_gpu, output_gpu)

    # ── Copy GPU output back ──────────────────────────────────────────
    var out_host = ctx.enqueue_create_host_buffer[DT](BATCH * OUT)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    # ── Compare ────────────────────────────────────────────────────────
    var max_diff: Scalar[DT] = 0.0
    for b in range(BATCH):
        for j in range(OUT):
            var c = output_cpu[b, j]
            var g = out_host.unsafe_ptr()[b * OUT + j]
            var d = fabs(c - g)
            if d > max_diff:
                max_diff = d
    print("max-abs-diff CPU vs GPU = " + String(max_diff))
    assert_true(max_diff < Scalar[DT](1e-5),
        "Forward parity failed: max-abs-diff = " + String(max_diff))

    in_buf_cpu.free()
    out_buf_cpu.free()
    print("  test_linear_gpu_forward_parity PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 LinearGPU forward parity test (CPU vs GPU)")
    print("=" * 60)
    test_linear_gpu_forward_parity()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
