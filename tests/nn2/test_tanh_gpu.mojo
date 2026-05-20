"""TanhGPU parity vs CPU Tanh — forward + backward.

Tanh is element-wise — same parity expectations as ReLU (bit-exact on
CPU vs bit-near-exact on GPU; minor ULP differences possible due to
different tanh implementations between std.math and Metal/CUDA).
"""

from std.math import abs as fabs
from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.tanh import Tanh


def test_tanh_gpu_parity() raises:
    comptime DIM = 5
    comptime BATCH = 3
    comptime TOL: Scalar[DT] = 1e-5    # element-wise tanh; allow ULP drift

    var ctx = DeviceContext()
    var tanh_cpu = Tanh[DIM].make["cpu", INIT=Zero]()
    var tanh_gpu = Tanh[DIM].make["gpu", INIT=Zero](ctx)

    # ── Mixed input: zero, small, large positive, large negative ────────
    var in_host = ctx.enqueue_create_host_buffer[DT](BATCH * DIM)
    ctx.synchronize()
    var ip = in_host.unsafe_ptr()
    ip[0] =  0.0;  ip[1] =  0.1; ip[2] =  0.5; ip[3] =  1.0; ip[4] =  2.0
    ip[5] = -0.1; ip[6] = -0.5; ip[7] = -1.0; ip[8] = -2.0; ip[9] =  3.0
    ip[10] = -3.0; ip[11] = 0.25; ip[12] = -0.75; ip[13] = 1.5; ip[14] = -1.5

    var in_buf_cpu:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var out_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        in_buf_cpu[k] = ip[k]
    var input_cpu  = TileTensor(in_buf_cpu, row_major[BATCH, DIM]())
    var output_cpu = TileTensor(out_buf_cpu, row_major[BATCH, DIM]())

    var in_dev  = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    var out_dev = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    ctx.enqueue_copy(in_dev, in_host)
    var input_gpu  = TileTensor(in_dev, row_major[BATCH, DIM]())
    var output_gpu = TileTensor(out_dev, row_major[BATCH, DIM]())

    # ── Forward ──────────────────────────────────────────────────────────
    tanh_cpu.forward["cpu", BATCH](input_cpu, output_cpu)
    tanh_gpu.forward["gpu", BATCH](input_gpu, output_gpu)

    var out_host = ctx.enqueue_create_host_buffer[DT](BATCH * DIM)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    var max_diff_fwd: Scalar[DT] = 0.0
    for b in range(BATCH):
        for d in range(DIM):
            var diff = fabs(output_cpu[b, d] - out_host.unsafe_ptr()[b * DIM + d])
            if diff > max_diff_fwd: max_diff_fwd = diff
    print("max-diff forward  = " + String(max_diff_fwd))
    assert_true(max_diff_fwd < TOL,
        "Forward not within ULP tol: " + String(max_diff_fwd))

    # ── grad_output for backward ─────────────────────────────────────────
    var go_host = ctx.enqueue_create_host_buffer[DT](BATCH * DIM)
    ctx.synchronize()
    for k in range(BATCH * DIM):
        go_host.unsafe_ptr()[k] = Scalar[DT](1.0 + Float32(k) * 0.1)

    var go_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        go_buf_cpu[k] = go_host.unsafe_ptr()[k]
    var grad_out_cpu = TileTensor(go_buf_cpu, row_major[BATCH, DIM]())
    var grad_in_cpu  = TileTensor(gi_buf_cpu, row_major[BATCH, DIM]())

    var go_dev = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    var gi_dev = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    ctx.enqueue_copy(go_dev, go_host)
    var grad_out_gpu = TileTensor(go_dev, row_major[BATCH, DIM]())
    var grad_in_gpu  = TileTensor(gi_dev, row_major[BATCH, DIM]())

    tanh_cpu.backward["cpu", BATCH](grad_out_cpu, grad_in_cpu)
    tanh_gpu.backward["gpu", BATCH](grad_out_gpu, grad_in_gpu)

    var gi_host = ctx.enqueue_create_host_buffer[DT](BATCH * DIM)
    ctx.enqueue_copy(gi_host, gi_dev)
    ctx.synchronize()

    var max_diff_bwd: Scalar[DT] = 0.0
    for b in range(BATCH):
        for d in range(DIM):
            var diff = fabs(grad_in_cpu[b, d] - gi_host.unsafe_ptr()[b * DIM + d])
            if diff > max_diff_bwd: max_diff_bwd = diff
    print("max-diff backward = " + String(max_diff_bwd))
    assert_true(max_diff_bwd < TOL,
        "Backward not within ULP tol: " + String(max_diff_bwd))

    in_buf_cpu.free()
    out_buf_cpu.free()
    go_buf_cpu.free()
    gi_buf_cpu.free()
    print("  test_tanh_gpu_parity PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 TanhGPU parity vs Tanh (CPU)")
    print("=" * 60)
    test_tanh_gpu_parity()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
