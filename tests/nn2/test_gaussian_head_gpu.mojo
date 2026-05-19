"""GaussianHead GPU parity vs CPU GaussianHead — forward + backward.

Verifies the GPU kernels match CPU semantics:
  - mu column = state-dependent Linear(input)
  - log_std columns = broadcast(clamp(log_std_param))
  - backward grad_input, grad_w, grad_b, grad_log_std match CPU within ULP tol
"""

from std.math import abs as fabs
from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.gaussian_head import GaussianHead


def test_gaussian_head_gpu_parity() raises:
    comptime IN = 4
    comptime ACT = 2
    comptime BATCH = 3
    comptime TOL_FWD: Scalar[DT] = 1e-5
    comptime TOL_BWD: Scalar[DT] = 1e-4

    var ctx = DeviceContext()
    var h_cpu = GaussianHead[IN, ACT].make[target="cpu", INIT=Zero]()
    var h_gpu = GaussianHead[IN, ACT].make[target="gpu", INIT=Zero](ctx)

    # ── Hand-set params on CPU, then mirror into GPU buffers ────────────
    var w_cpu = TileTensor(h_cpu.weight, row_major[IN, ACT]())
    var b_cpu = TileTensor(h_cpu.bias, row_major[ACT]())
    var ls_cpu = TileTensor(h_cpu.log_std, row_major[ACT]())
    for i in range(IN):
        for j in range(ACT):
            w_cpu[i, j] = Scalar[DT](0.1 * Float64(i * ACT + j + 1))
    for j in range(ACT):
        b_cpu[j] = Scalar[DT](0.05 * Float64(j + 1))
        ls_cpu[j] = Scalar[DT](0.5 - 0.1 * Float64(j))

    # Upload to GPU via host buffers.
    var w_host = ctx.enqueue_create_host_buffer[DT](IN * ACT)
    var b_host = ctx.enqueue_create_host_buffer[DT](ACT)
    var ls_host = ctx.enqueue_create_host_buffer[DT](ACT)
    ctx.synchronize()
    for i in range(IN):
        for j in range(ACT):
            w_host.unsafe_ptr()[i * ACT + j] = w_cpu[i, j]
    for j in range(ACT):
        b_host.unsafe_ptr()[j] = b_cpu[j]
        ls_host.unsafe_ptr()[j] = ls_cpu[j]
    ctx.enqueue_copy(h_gpu.weight_dev.value(), w_host)
    ctx.enqueue_copy(h_gpu.bias_dev.value(), b_host)
    ctx.enqueue_copy(h_gpu.log_std_dev.value(), ls_host)
    ctx.synchronize()

    # ── Input ───────────────────────────────────────────────────────────
    var in_host = ctx.enqueue_create_host_buffer[DT](BATCH * IN)
    ctx.synchronize()
    for bi in range(BATCH):
        for i in range(IN):
            in_host.unsafe_ptr()[bi * IN + i] = Scalar[DT](
                0.1 + 0.01 * Float64(bi * IN + i)
            )

    var in_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var out_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    for k in range(BATCH * IN):
        in_buf_cpu[k] = in_host.unsafe_ptr()[k]
    var input_cpu = TileTensor(in_buf_cpu, row_major[BATCH, IN]())
    var output_cpu = TileTensor(out_buf_cpu, row_major[BATCH, 2 * ACT]())

    var in_dev = ctx.enqueue_create_buffer[DT](BATCH * IN)
    var out_dev = ctx.enqueue_create_buffer[DT](BATCH * 2 * ACT)
    ctx.enqueue_copy(in_dev, in_host)
    var input_gpu = TileTensor(in_dev, row_major[BATCH, IN]())
    var output_gpu = TileTensor(out_dev, row_major[BATCH, 2 * ACT]())

    # ── Forward parity ──────────────────────────────────────────────────
    h_cpu.forward["cpu", BATCH](input_cpu, output_cpu)
    h_gpu.forward["gpu", BATCH](input_gpu, output_gpu)

    var out_host = ctx.enqueue_create_host_buffer[DT](BATCH * 2 * ACT)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    var max_diff_fwd: Scalar[DT] = 0.0
    for b in range(BATCH):
        for j in range(2 * ACT):
            var d = fabs(
                output_cpu[b, j] - out_host.unsafe_ptr()[b * 2 * ACT + j]
            )
            if d > max_diff_fwd:
                max_diff_fwd = d
    print("max-diff forward = " + String(max_diff_fwd))
    assert_true(max_diff_fwd < TOL_FWD, "forward parity failed")

    # ── Backward parity ─────────────────────────────────────────────────
    var go_host = ctx.enqueue_create_host_buffer[DT](BATCH * 2 * ACT)
    ctx.synchronize()
    for k in range(BATCH * 2 * ACT):
        go_host.unsafe_ptr()[k] = Scalar[DT](0.5 + 0.05 * Float32(k))

    var go_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var gi_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    for k in range(BATCH * 2 * ACT):
        go_buf_cpu[k] = go_host.unsafe_ptr()[k]
    for k in range(BATCH * IN):
        gi_buf_cpu[k] = 0.0
    var grad_out_cpu = TileTensor(go_buf_cpu, row_major[BATCH, 2 * ACT]())
    var grad_in_cpu = TileTensor(gi_buf_cpu, row_major[BATCH, IN]())

    var go_dev = ctx.enqueue_create_buffer[DT](BATCH * 2 * ACT)
    var gi_dev = ctx.enqueue_create_buffer[DT](BATCH * IN)
    ctx.enqueue_copy(go_dev, go_host)
    var grad_out_gpu = TileTensor(go_dev, row_major[BATCH, 2 * ACT]())
    var grad_in_gpu = TileTensor(gi_dev, row_major[BATCH, IN]())

    h_cpu.backward["cpu", BATCH](grad_out_cpu, grad_in_cpu)
    h_gpu.backward["gpu", BATCH](grad_out_gpu, grad_in_gpu)

    # Pull GPU grads back.
    var gi_host = ctx.enqueue_create_host_buffer[DT](BATCH * IN)
    var gw_host = ctx.enqueue_create_host_buffer[DT](IN * ACT)
    var gb_host = ctx.enqueue_create_host_buffer[DT](ACT)
    var gls_host = ctx.enqueue_create_host_buffer[DT](ACT)
    ctx.enqueue_copy(gi_host, gi_dev)
    ctx.enqueue_copy(gw_host, h_gpu.grad_w_dev.value())
    ctx.enqueue_copy(gb_host, h_gpu.grad_b_dev.value())
    ctx.enqueue_copy(gls_host, h_gpu.grad_ls_dev.value())
    ctx.synchronize()

    # grad_input
    var max_diff_gi: Scalar[DT] = 0.0
    for b in range(BATCH):
        for i in range(IN):
            var d = fabs(grad_in_cpu[b, i] - gi_host.unsafe_ptr()[b * IN + i])
            if d > max_diff_gi:
                max_diff_gi = d
    print("max-diff grad_input  = " + String(max_diff_gi))
    assert_true(max_diff_gi < TOL_BWD, "grad_input parity failed")

    # grad_w
    var gw_cpu = TileTensor(h_cpu.grad_w, row_major[IN, ACT]())
    var max_diff_gw: Scalar[DT] = 0.0
    for i in range(IN):
        for j in range(ACT):
            var d = fabs(gw_cpu[i, j] - gw_host.unsafe_ptr()[i * ACT + j])
            if d > max_diff_gw:
                max_diff_gw = d
    print("max-diff grad_w      = " + String(max_diff_gw))
    assert_true(max_diff_gw < TOL_BWD, "grad_w parity failed")

    # grad_b
    var gb_cpu = TileTensor(h_cpu.grad_b, row_major[ACT]())
    var max_diff_gb: Scalar[DT] = 0.0
    for j in range(ACT):
        var d = fabs(gb_cpu[j] - gb_host.unsafe_ptr()[j])
        if d > max_diff_gb:
            max_diff_gb = d
    print("max-diff grad_b      = " + String(max_diff_gb))
    assert_true(max_diff_gb < TOL_BWD, "grad_b parity failed")

    # grad_log_std
    var gls_cpu = TileTensor(h_cpu.grad_ls, row_major[ACT]())
    var max_diff_gls: Scalar[DT] = 0.0
    for j in range(ACT):
        var d = fabs(gls_cpu[j] - gls_host.unsafe_ptr()[j])
        if d > max_diff_gls:
            max_diff_gls = d
    print("max-diff grad_log_std= " + String(max_diff_gls))
    assert_true(max_diff_gls < TOL_BWD, "grad_log_std parity failed")

    in_buf_cpu.free()
    out_buf_cpu.free()
    go_buf_cpu.free()
    gi_buf_cpu.free()
    print("  test_gaussian_head_gpu_parity PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 GaussianHead GPU parity vs CPU")
    print("=" * 60)
    test_gaussian_head_gpu_parity()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
