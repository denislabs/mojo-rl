"""StopGrad[DIM] CPU+GPU tests — Phase 5.2.

Covers:
  - forward: identity copy
  - backward: grad_input is zeroed regardless of grad_output content
  - for_each_param yields zero params
  - GPU parity vs CPU
"""

from std.math import abs as fabs
from std.memory import alloc
from std.testing import assert_equal, assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.core import ParamVisitor
from mojo_rl.nn2.primitives.stop_grad import StopGrad


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


def test_forward_cpu() raises:
    """Forward is identity — output bit-equal to input."""
    comptime DIM = 4
    comptime BATCH = 2

    var sg = StopGrad[DIM].make["cpu", INIT=Zero]()

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        in_buf[k] = Scalar[DT](Float32(k) * 0.5 - 1.0)
        out_buf[k] = -999.0

    var input  = TileTensor(in_buf,  row_major[BATCH, DIM]())
    var output = TileTensor(out_buf, row_major[BATCH, DIM]())

    sg.forward["cpu", BATCH](input, output)

    for b in range(BATCH):
        for d in range(DIM):
            assert_equal(output[b, d], input[b, d])

    in_buf.free()
    out_buf.free()
    print("  test_forward_cpu PASSED")


def test_backward_zeros_grad_cpu() raises:
    """Backward writes zero regardless of grad_output content."""
    comptime DIM = 3
    comptime BATCH = 2

    var sg = StopGrad[DIM].make["cpu", INIT=Zero]()

    var go_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        go_buf[k] = Scalar[DT](100.0 + Float32(k))   # large nonzero values
        gi_buf[k] = Scalar[DT](999.0)                # poison: must be overwritten to 0

    var grad_out = TileTensor(go_buf, row_major[BATCH, DIM]())
    var grad_in  = TileTensor(gi_buf, row_major[BATCH, DIM]())

    sg.backward["cpu", BATCH](grad_out, grad_in)

    for b in range(BATCH):
        for d in range(DIM):
            assert_equal(grad_in[b, d], 0.0)

    go_buf.free()
    gi_buf.free()
    print("  test_backward_zeros_grad_cpu PASSED")


def test_for_each_param_no_params() raises:
    var sg = StopGrad[16].make["cpu", INIT=Zero]()
    var v = CountVisitor()
    sg.for_each_param["cpu"](String("sg0"), v)
    assert_equal(v.visits, 0)
    print("  test_for_each_param_no_params PASSED")


def test_gpu_parity() raises:
    """GPU forward/backward bit-exact vs CPU."""
    comptime DIM = 5
    comptime BATCH = 3

    var ctx = DeviceContext()
    var sg_cpu = StopGrad[DIM].make["cpu", INIT=Zero]()
    var sg_gpu = StopGrad[DIM].make["gpu", INIT=Zero](ctx)

    # ── Forward parity ──────────────────────────────────────────────────
    var in_host = ctx.enqueue_create_host_buffer[DT](BATCH * DIM)
    ctx.synchronize()
    var ip = in_host.unsafe_ptr()
    for k in range(BATCH * DIM):
        ip[k] = Scalar[DT](Float32(k) * 0.3 - 2.0)

    var in_buf_cpu:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var out_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        in_buf_cpu[k] = ip[k]
    var input_cpu  = TileTensor(in_buf_cpu, row_major[BATCH, DIM]())
    var output_cpu = TileTensor(out_buf_cpu, row_major[BATCH, DIM]())

    var in_dev  = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    var out_dev = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    ctx.enqueue_copy(in_dev, in_host)
    var input_gpu  = TileTensor(in_dev,  row_major[BATCH, DIM]())
    var output_gpu = TileTensor(out_dev, row_major[BATCH, DIM]())

    sg_cpu.forward["cpu", BATCH](input_cpu, output_cpu)
    sg_gpu.forward["gpu", BATCH](input_gpu, output_gpu)

    var out_host = ctx.enqueue_create_host_buffer[DT](BATCH * DIM)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    var max_diff_fwd: Scalar[DT] = 0.0
    for b in range(BATCH):
        for d in range(DIM):
            var diff = fabs(output_cpu[b, d] - out_host.unsafe_ptr()[b * DIM + d])
            if diff > max_diff_fwd: max_diff_fwd = diff
    assert_true(max_diff_fwd == Scalar[DT](0.0),
        "Forward not bit-exact: " + String(max_diff_fwd))

    # ── Backward parity: both should produce all-zero grad_input ─────────
    var go_host = ctx.enqueue_create_host_buffer[DT](BATCH * DIM)
    ctx.synchronize()
    for k in range(BATCH * DIM):
        go_host.unsafe_ptr()[k] = Scalar[DT](777.0 + Float32(k))

    var go_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        go_buf_cpu[k] = go_host.unsafe_ptr()[k]
        gi_buf_cpu[k] = Scalar[DT](999.0)        # poison
    var grad_out_cpu = TileTensor(go_buf_cpu, row_major[BATCH, DIM]())
    var grad_in_cpu  = TileTensor(gi_buf_cpu, row_major[BATCH, DIM]())

    var go_dev = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    var gi_dev = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    ctx.enqueue_copy(go_dev, go_host)
    var poison_host = ctx.enqueue_create_host_buffer[DT](BATCH * DIM)
    ctx.synchronize()
    for k in range(BATCH * DIM):
        poison_host.unsafe_ptr()[k] = Scalar[DT](999.0)
    ctx.enqueue_copy(gi_dev, poison_host)
    var grad_out_gpu = TileTensor(go_dev, row_major[BATCH, DIM]())
    var grad_in_gpu  = TileTensor(gi_dev, row_major[BATCH, DIM]())

    sg_cpu.backward["cpu", BATCH](grad_out_cpu, grad_in_cpu)
    sg_gpu.backward["gpu", BATCH](grad_out_gpu, grad_in_gpu)

    var gi_host = ctx.enqueue_create_host_buffer[DT](BATCH * DIM)
    ctx.enqueue_copy(gi_host, gi_dev)
    ctx.synchronize()

    var max_diff_bwd: Scalar[DT] = 0.0
    var max_abs_gi: Scalar[DT] = 0.0
    for b in range(BATCH):
        for d in range(DIM):
            var v = gi_host.unsafe_ptr()[b * DIM + d]
            if fabs(v) > max_abs_gi: max_abs_gi = fabs(v)
            var diff = fabs(grad_in_cpu[b, d] - v)
            if diff > max_diff_bwd: max_diff_bwd = diff
    assert_true(max_diff_bwd == Scalar[DT](0.0),
        "Backward not bit-exact: " + String(max_diff_bwd))
    assert_true(max_abs_gi == Scalar[DT](0.0),
        "GPU grad_input not all zero: max-abs = " + String(max_abs_gi))

    in_buf_cpu.free()
    out_buf_cpu.free()
    go_buf_cpu.free()
    gi_buf_cpu.free()
    print("  test_gpu_parity PASSED (forward bit-exact + backward all-zero)")


def main() raises:
    print("=" * 60)
    print("nn2 StopGrad unit + parity tests (CPU + GPU, Phase 5.2)")
    print("=" * 60)
    test_forward_cpu()
    test_backward_zeros_grad_cpu()
    test_for_each_param_no_params()
    test_gpu_parity()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
