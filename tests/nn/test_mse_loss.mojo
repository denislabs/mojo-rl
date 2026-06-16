"""MSELoss[OUT_DIM] CPU + GPU tests — Phase 6.3.

Covers:
  - forward: L = (1/BATCH) * sum_b 0.5 * sum_j (logits[b,j] - t[b,j])^2
  - backward: grad_logits[b, j] = (logits[b, j] - t[b, j]) / BATCH
  - GPU parity vs CPU
  - FD gradcheck (linear-in-logits closed form makes this near-exact)
"""

from std.math import abs as fabs
from std.memory import alloc
from std.testing import assert_equal, assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.loss.mse import MSELoss


# ──────────────────────────────────────────────────────────────────────────
# CPU: forward + backward correctness on a hand-checkable case
# ──────────────────────────────────────────────────────────────────────────


def test_forward_backward_cpu() raises:
    """BATCH=2, OUT_DIM=1.
    logits = [3.0, 1.0], targets = [1.0, 4.0].
    diffs = [2.0, -3.0]. L = (1/2) * (0.5*4 + 0.5*9) = 6.5 / 2 = 3.25.
    grad = [2.0/2, -3.0/2] = [1.0, -1.5]."""
    comptime BATCH = 2
    comptime OUT = 1
    var loss = MSELoss[OUT].make["cpu"]()
    var l_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var t_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    l_buf[0] = 3.0
    l_buf[1] = 1.0
    t_buf[0] = 1.0
    t_buf[1] = 4.0
    var logits = TileTensor(l_buf, row_major[BATCH, OUT]())
    var targets = TileTensor(t_buf, row_major[BATCH, OUT]())
    var L = loss.forward["cpu", BATCH](logits, targets)
    assert_true(fabs(L - 3.25) < 1e-6, "wrong loss: " + String(L))

    var g_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    g_buf[0] = 0.0
    g_buf[1] = 0.0
    var grad = TileTensor(g_buf, row_major[BATCH, OUT]())
    loss.vjp["cpu", BATCH](targets, grad)
    assert_true(fabs(grad[0, 0] - 1.0) < 1e-6)
    assert_true(fabs(grad[1, 0] - (-1.5)) < 1e-6)

    l_buf.free()
    t_buf.free()
    g_buf.free()
    print("  test_forward_backward_cpu PASSED")


# ──────────────────────────────────────────────────────────────────────────
# CPU FD gradcheck (closed-form linear-in-residual makes this near-exact)
# ──────────────────────────────────────────────────────────────────────────


def test_gradcheck_fd() raises:
    comptime BATCH = 3
    comptime OUT = 2
    comptime EPS: Scalar[DT] = 1e-2
    comptime TOL_REL: Scalar[DT] = 1e-3

    var loss = MSELoss[OUT].make["cpu"]()
    var l_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var t_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    for b in range(BATCH):
        for j in range(OUT):
            l_buf[b * OUT + j] = Scalar[DT](0.1 + 0.05 * Float64(b * OUT + j))
            t_buf[b * OUT + j] = Scalar[DT](
                0.5 - 0.03 * Float64(b * OUT + j)
            )
    var logits = TileTensor(l_buf, row_major[BATCH, OUT]())
    var targets = TileTensor(t_buf, row_major[BATCH, OUT]())

    _ = loss.forward["cpu", BATCH](logits, targets)
    var g_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    for k in range(BATCH * OUT):
        g_buf[k] = 0.0
    var grad = TileTensor(g_buf, row_major[BATCH, OUT]())
    loss.vjp["cpu", BATCH](targets, grad)

    var max_rel: Scalar[DT] = 0.0
    for b in range(BATCH):
        for j in range(OUT):
            var saved = l_buf[b * OUT + j]
            l_buf[b * OUT + j] = saved + EPS
            var Lp = loss.forward["cpu", BATCH](logits, targets)
            l_buf[b * OUT + j] = saved - EPS
            var Lm = loss.forward["cpu", BATCH](logits, targets)
            l_buf[b * OUT + j] = saved
            var fd = (Lp - Lm) / (Scalar[DT](2.0) * EPS)
            var an = grad[b, j]
            var denom = fabs(an) + Scalar[DT](1e-6)
            var rel = fabs(fd - an) / denom
            if rel > max_rel:
                max_rel = rel

    # Re-do the analytical backward since forward calls above thrashed cache.
    _ = loss.forward["cpu", BATCH](logits, targets)
    loss.vjp["cpu", BATCH](targets, grad)

    print("  MSE FD gradcheck max_rel = ", max_rel)
    assert_true(max_rel < TOL_REL)

    l_buf.free()
    t_buf.free()
    g_buf.free()
    print("  test_gradcheck_fd PASSED")


# ──────────────────────────────────────────────────────────────────────────
# GPU parity vs CPU
# ──────────────────────────────────────────────────────────────────────────


def test_gpu_parity() raises:
    comptime BATCH = 4
    comptime OUT = 3
    comptime TOL_FWD: Scalar[DT] = 1e-5
    comptime TOL_BWD: Scalar[DT] = 1e-6

    var ctx = DeviceContext()
    var loss_cpu = MSELoss[OUT].make["cpu"]()
    var loss_gpu = MSELoss[OUT].make["gpu"](ctx)

    var l_host = ctx.enqueue_create_host_buffer[DT](BATCH * OUT)
    var t_host = ctx.enqueue_create_host_buffer[DT](BATCH * OUT)
    ctx.synchronize()
    for k in range(BATCH * OUT):
        l_host.unsafe_ptr()[k] = Scalar[DT](0.1 + 0.07 * Float64(k))
        t_host.unsafe_ptr()[k] = Scalar[DT](0.4 - 0.03 * Float64(k))

    var l_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var t_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    for k in range(BATCH * OUT):
        l_cpu[k] = l_host.unsafe_ptr()[k]
        t_cpu[k] = t_host.unsafe_ptr()[k]
    var logits_cpu = TileTensor(l_cpu, row_major[BATCH, OUT]())
    var targets_cpu = TileTensor(t_cpu, row_major[BATCH, OUT]())

    var l_dev = ctx.enqueue_create_buffer[DT](BATCH * OUT)
    var t_dev = ctx.enqueue_create_buffer[DT](BATCH * OUT)
    ctx.enqueue_copy(l_dev, l_host)
    ctx.enqueue_copy(t_dev, t_host)
    var logits_gpu = TileTensor(l_dev, row_major[BATCH, OUT]())
    var targets_gpu = TileTensor(t_dev, row_major[BATCH, OUT]())

    var L_cpu = loss_cpu.forward["cpu", BATCH](logits_cpu, targets_cpu)
    var L_gpu = loss_gpu.forward["gpu", BATCH](logits_gpu, targets_gpu)
    print("L_cpu = ", L_cpu, "  L_gpu = ", L_gpu)
    assert_true(fabs(L_cpu - L_gpu) < TOL_FWD, "forward parity failed")

    var g_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    for k in range(BATCH * OUT):
        g_cpu[k] = 0.0
    var grad_cpu = TileTensor(g_cpu, row_major[BATCH, OUT]())
    loss_cpu.vjp["cpu", BATCH](targets_cpu, grad_cpu)

    var g_dev = ctx.enqueue_create_buffer[DT](BATCH * OUT)
    var grad_gpu = TileTensor(g_dev, row_major[BATCH, OUT]())
    loss_gpu.vjp["gpu", BATCH](targets_gpu, grad_gpu)

    var g_host = ctx.enqueue_create_host_buffer[DT](BATCH * OUT)
    ctx.enqueue_copy(g_host, g_dev)
    ctx.synchronize()

    var max_diff: Scalar[DT] = 0.0
    for k in range(BATCH * OUT):
        var d = fabs(grad_cpu[k // OUT, k % OUT] - g_host.unsafe_ptr()[k])
        if d > max_diff:
            max_diff = d
    print("max-diff grad = " + String(max_diff))
    assert_true(max_diff < TOL_BWD, "backward parity failed")

    l_cpu.free()
    t_cpu.free()
    g_cpu.free()
    print("  test_gpu_parity PASSED")


# ──────────────────────────────────────────────────────────────────────────
# GPU device-accumulate path (Slice 3 — CUDA-graph capturable, no per-step D2H)
# ──────────────────────────────────────────────────────────────────────────


def test_gpu_accumulate() raises:
    """`forward_accumulate` + `read_accum` must match `forward` within the
    block-reduce tolerance (~1e-5), and average correctly across steps."""
    comptime BATCH = 8
    comptime OUT = 3
    comptime TOL: Scalar[DT] = 1e-5

    var ctx = DeviceContext()
    var loss = MSELoss[OUT].make["gpu"](ctx)

    var l_host = ctx.enqueue_create_host_buffer[DT](BATCH * OUT)
    var t_host = ctx.enqueue_create_host_buffer[DT](BATCH * OUT)
    ctx.synchronize()
    for k in range(BATCH * OUT):
        l_host.unsafe_ptr()[k] = Scalar[DT](0.1 + 0.07 * Float64(k))
        t_host.unsafe_ptr()[k] = Scalar[DT](0.4 - 0.03 * Float64(k))
    var l_dev = ctx.enqueue_create_buffer[DT](BATCH * OUT)
    var t_dev = ctx.enqueue_create_buffer[DT](BATCH * OUT)
    ctx.enqueue_copy(l_dev, l_host)
    ctx.enqueue_copy(t_dev, t_host)
    var logits = TileTensor(l_dev, row_major[BATCH, OUT]())
    var targets = TileTensor(t_dev, row_major[BATCH, OUT]())

    # Reference: the legacy D2H forward.
    var L_ref = loss.forward["gpu", BATCH](logits, targets)

    # Accumulate the same batch 3 times → mean must equal L_ref.
    loss.reset_accum["gpu"]()
    for _ in range(3):
        loss.forward_accumulate["gpu", BATCH](logits, targets)
    var L_acc = loss.read_accum["gpu"]()
    print("  L_ref =", L_ref, "  L_acc(3x same) =", L_acc)
    assert_true(fabs(L_acc - L_ref) < TOL, "accumulate mean != forward")

    # Second batch with different data; mean of the two single-step means.
    for k in range(BATCH * OUT):
        l_host.unsafe_ptr()[k] = Scalar[DT](0.2 + 0.05 * Float64(k))
    ctx.enqueue_copy(l_dev, l_host)
    var L_ref2 = loss.forward["gpu", BATCH](logits, targets)
    loss.reset_accum["gpu"]()
    loss.forward_accumulate["gpu", BATCH](logits, targets)  # L_ref2
    ctx.enqueue_copy(l_dev, t_dev)  # logits == targets → loss 0
    loss.forward_accumulate["gpu", BATCH](logits, targets)  # 0
    var L_avg = loss.read_accum["gpu"]()
    var expect = L_ref2 / Scalar[DT](2.0)
    print("  L_avg(ref2, 0) =", L_avg, "  expect =", expect)
    assert_true(fabs(L_avg - expect) < TOL, "two-step average wrong")

    print("  test_gpu_accumulate PASSED")


def main() raises:
    print("=" * 60)
    print("nn MSELoss unit tests (CPU + GPU, Phase 6.3)")
    print("=" * 60)
    test_forward_backward_cpu()
    test_gradcheck_fd()
    test_gpu_parity()
    test_gpu_accumulate()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
