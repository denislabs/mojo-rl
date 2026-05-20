"""CrossEntropyLossGPU parity vs CPU CrossEntropyLoss.

Same logits + targets on both. Compare loss scalar and grad_logits.
"""

from std.math import abs as fabs
from std.memory import alloc
from std.testing import assert_true, assert_almost_equal
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.loss import CrossEntropyLoss


def test_ce_gpu_parity() raises:
    comptime N = 6
    comptime BATCH = 4

    var ctx = DeviceContext()
    var ce_cpu = CrossEntropyLoss[N].make["cpu"]()
    var ce_gpu = CrossEntropyLoss[N].make["gpu"](ctx)

    # ── Distinct logits and one-hot targets ──────────────────────────
    var lg_host = ctx.enqueue_create_host_buffer[DT](BATCH * N)
    var tg_host = ctx.enqueue_create_host_buffer[DT](BATCH * N)
    ctx.synchronize()
    for b in range(BATCH):
        for c in range(N):
            lg_host.unsafe_ptr()[b * N + c] = Scalar[DT](
                0.2 * Float32(b + 1) + 0.3 * Float32(c) - 0.5
            )
            tg_host.unsafe_ptr()[b * N + c] = 0.0
        # true class = b % N
        tg_host.unsafe_ptr()[b * N + (b % N)] = 1.0

    # CPU views
    var lg_cpu_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N)
    var tg_cpu_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N)
    for k in range(BATCH * N):
        lg_cpu_buf[k] = lg_host.unsafe_ptr()[k]
        tg_cpu_buf[k] = tg_host.unsafe_ptr()[k]
    var logits_cpu  = TileTensor(lg_cpu_buf, row_major[BATCH, N]())
    var targets_cpu = TileTensor(tg_cpu_buf, row_major[BATCH, N]())

    # GPU views (DeviceBuffer + upload)
    var lg_dev = ctx.enqueue_create_buffer[DT](BATCH * N)
    var tg_dev = ctx.enqueue_create_buffer[DT](BATCH * N)
    ctx.enqueue_copy(lg_dev, lg_host)
    ctx.enqueue_copy(tg_dev, tg_host)
    var logits_gpu  = TileTensor(lg_dev, row_major[BATCH, N]())
    var targets_gpu = TileTensor(tg_dev, row_major[BATCH, N]())

    # ── Forward ───────────────────────────────────────────────────────
    var loss_cpu = ce_cpu.forward["cpu", BATCH](logits_cpu, targets_cpu)
    var loss_gpu = ce_gpu.forward["gpu", BATCH](logits_gpu, targets_gpu)
    print("loss_cpu =", loss_cpu, " loss_gpu =", loss_gpu)
    assert_almost_equal(loss_cpu, loss_gpu, atol=1e-5)

    # ── Backward ──────────────────────────────────────────────────────
    var gl_cpu_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * N)
    var gl_dev = ctx.enqueue_create_buffer[DT](BATCH * N)
    var grad_logits_cpu = TileTensor(gl_cpu_buf, row_major[BATCH, N]())
    var grad_logits_gpu = TileTensor(gl_dev, row_major[BATCH, N]())

    ce_cpu.backward["cpu", BATCH](targets_cpu, grad_logits_cpu)
    ce_gpu.backward["gpu", BATCH](targets_gpu, grad_logits_gpu)

    var gl_host = ctx.enqueue_create_host_buffer[DT](BATCH * N)
    ctx.enqueue_copy(gl_host, gl_dev)
    ctx.synchronize()

    var max_diff: Scalar[DT] = 0.0
    for b in range(BATCH):
        for c in range(N):
            var diff = fabs(grad_logits_cpu[b, c] - gl_host.unsafe_ptr()[b * N + c])
            if diff > max_diff: max_diff = diff
    print("max-diff grad_logits =", max_diff)
    assert_true(max_diff < Scalar[DT](1e-5),
        "grad_logits parity failed: " + String(max_diff))

    lg_cpu_buf.free()
    tg_cpu_buf.free()
    gl_cpu_buf.free()
    print("  test_ce_gpu_parity PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 CrossEntropyLossGPU parity vs CPU")
    print("=" * 60)
    test_ce_gpu_parity()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
