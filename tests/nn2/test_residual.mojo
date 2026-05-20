"""Residual[Inner] CPU + GPU tests — Phase 5.5.

Strategy:
  - Forward + backward correctness over Residual[Tanh[DIM]] (parameterless,
    bit-exactly hand-checkable).
  - Gradient flow check: residual's grad_input must include the
    +grad_output direct path AND inner.backward(grad_output).
  - for_each_param over Residual[Linear[DIM, DIM]] — names should be
    "res0.inner.weight" / "res0.inner.bias".
  - CPU↔GPU parity over both.
"""

from std.math import abs as fabs, tanh as math_tanh
from std.memory import alloc
from std.testing import assert_equal, assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.core import ParamVisitor
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.tanh import Tanh
from mojo_rl.nn2.combinators import Residual


struct ParamRecord(Movable & ImplicitlyDestructible):
    var name: String
    var n_elems: Int
    var apply_decay: Bool

    def __init__(out self, name: String, n_elems: Int, apply_decay: Bool):
        self.name = name
        self.n_elems = n_elems
        self.apply_decay = apply_decay


struct RecordVisitor(ParamVisitor):
    var records: List[ParamRecord]

    def __init__(out self):
        self.records = List[ParamRecord]()

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
        self.records.append(ParamRecord(name, n_elems, apply_decay))


def test_forward_residual_tanh_cpu() raises:
    """Residual[Tanh[DIM]]: output = tanh(x) + x."""
    comptime DIM = 4
    comptime BATCH = 2
    comptime TOL: Scalar[DT] = 1e-5

    var r = Residual[Tanh[DIM]].make["cpu", INIT=Zero]()

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    in_buf[0] =  0.5; in_buf[1] = -0.5; in_buf[2] =  1.0; in_buf[3] = -1.0
    in_buf[4] =  0.0; in_buf[5] =  2.0; in_buf[6] = -2.0; in_buf[7] =  0.25
    for k in range(BATCH * DIM):
        out_buf[k] = -999.0

    var input = TileTensor(in_buf, row_major[BATCH, DIM]())
    var output = TileTensor(out_buf, row_major[BATCH, DIM]())
    r.forward["cpu", BATCH](input, output)

    for b in range(BATCH):
        for d in range(DIM):
            var x = input[b, d]
            var expected = math_tanh(x) + x
            assert_true(fabs(output[b, d] - expected) < TOL,
                "b=" + String(b) + " d=" + String(d)
                + " expected " + String(expected)
                + " got " + String(output[b, d]))

    in_buf.free()
    out_buf.free()
    print("  test_forward_residual_tanh_cpu PASSED")


def test_backward_residual_tanh_cpu_gradcheck() raises:
    """Residual[Tanh]: grad_input = grad_output * (1 - tanh^2(x))
                                     + grad_output (residual path)."""
    comptime DIM = 4
    comptime BATCH = 2
    comptime EPS_FD: Scalar[DT] = 1e-2
    comptime TOL_REL: Scalar[DT] = 1e-2

    var r = Residual[Tanh[DIM]].make["cpu", INIT=Zero]()

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var go_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    in_buf[0] =  0.3; in_buf[1] = -0.7; in_buf[2] =  1.2; in_buf[3] = -0.4
    in_buf[4] =  0.8; in_buf[5] = -0.1; in_buf[6] =  0.5; in_buf[7] = -1.5
    for k in range(BATCH * DIM):
        go_buf[k] = Scalar[DT](0.2 + Float32(k) * 0.1)
        gi_buf[k] = -999.0

    var input    = TileTensor(in_buf,  row_major[BATCH, DIM]())
    var output   = TileTensor(out_buf, row_major[BATCH, DIM]())
    var grad_out = TileTensor(go_buf,  row_major[BATCH, DIM]())
    var grad_in  = TileTensor(gi_buf,  row_major[BATCH, DIM]())

    r.forward["cpu", BATCH](input, output)
    r.backward["cpu", BATCH](grad_out, grad_in)

    # FD gradcheck: ∂L/∂x[bi,di] where L = Σ grad_output * output.
    var max_rel: Scalar[DT] = 0.0
    for bi in range(BATCH):
        for di in range(DIM):
            in_buf[bi * DIM + di] += EPS_FD
            r.forward["cpu", BATCH](input, output)
            var L_plus: Scalar[DT] = 0.0
            for b2 in range(BATCH):
                for d2 in range(DIM):
                    L_plus += go_buf[b2 * DIM + d2] * output[b2, d2]
            in_buf[bi * DIM + di] -= Scalar[DT](2.0) * EPS_FD
            r.forward["cpu", BATCH](input, output)
            var L_minus: Scalar[DT] = 0.0
            for b2 in range(BATCH):
                for d2 in range(DIM):
                    L_minus += go_buf[b2 * DIM + d2] * output[b2, d2]
            in_buf[bi * DIM + di] += EPS_FD
            var fd = (L_plus - L_minus) / (Scalar[DT](2.0) * EPS_FD)
            var an = grad_in[bi, di]
            var denom = fabs(an) + Scalar[DT](1e-6)
            var rel = fabs(fd - an) / denom
            if rel > max_rel: max_rel = rel

    print("Residual[Tanh] gradcheck max-rel-err = " + String(max_rel))
    assert_true(max_rel < TOL_REL,
        "Residual gradcheck failed: " + String(max_rel))

    in_buf.free()
    out_buf.free()
    go_buf.free()
    gi_buf.free()
    print("  test_backward_residual_tanh_cpu_gradcheck PASSED")


def test_for_each_param_residual_linear() raises:
    """Residual[Linear[D, D]]: prefix should be 'res0.inner.weight' /
    'res0.inner.bias' with apply_decay (True, False)."""
    var r = Residual[Linear[4, 4]].make["cpu", INIT=Zero]()
    var v = RecordVisitor()
    r.for_each_param["cpu"](String("res0"), v)
    assert_equal(len(v.records), 2)
    assert_equal(v.records[0].name, String("res0.inner.weight"))
    assert_equal(v.records[0].n_elems, 16)
    assert_true(v.records[0].apply_decay, "weight must have apply_decay=True")
    assert_equal(v.records[1].name, String("res0.inner.bias"))
    assert_equal(v.records[1].n_elems, 4)
    assert_true(not v.records[1].apply_decay, "bias must have apply_decay=False")
    print("  test_for_each_param_residual_linear PASSED")


def test_gpu_parity_residual_tanh() raises:
    """Forward + backward parity vs CPU on Residual[Tanh[DIM]]."""
    comptime DIM = 5
    comptime BATCH = 3
    comptime TOL: Scalar[DT] = 1e-5

    var ctx = DeviceContext()
    var r_cpu = Residual[Tanh[DIM]].make["cpu", INIT=Zero]()
    var r_gpu = Residual[Tanh[DIM]].make["gpu", INIT=Zero](ctx)

    # Input.
    var in_host = ctx.enqueue_create_host_buffer[DT](BATCH * DIM)
    ctx.synchronize()
    var ip = in_host.unsafe_ptr()
    for k in range(BATCH * DIM):
        ip[k] = Scalar[DT](Float32(k) * 0.2 - 1.5)

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

    r_cpu.forward["cpu", BATCH](input_cpu, output_cpu)
    r_gpu.forward["gpu", BATCH](input_gpu, output_gpu)

    var out_host = ctx.enqueue_create_host_buffer[DT](BATCH * DIM)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    var max_fwd: Scalar[DT] = 0.0
    for b in range(BATCH):
        for d in range(DIM):
            var diff = fabs(output_cpu[b, d] - out_host.unsafe_ptr()[b * DIM + d])
            if diff > max_fwd: max_fwd = diff
    print("forward max-diff = " + String(max_fwd))
    assert_true(max_fwd < TOL, "forward: " + String(max_fwd))

    # Backward.
    var go_host = ctx.enqueue_create_host_buffer[DT](BATCH * DIM)
    ctx.synchronize()
    for k in range(BATCH * DIM):
        go_host.unsafe_ptr()[k] = Scalar[DT](0.1 + Float32(k) * 0.05)

    var go_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        go_buf_cpu[k] = go_host.unsafe_ptr()[k]
        gi_buf_cpu[k] = -999.0
    var grad_out_cpu = TileTensor(go_buf_cpu, row_major[BATCH, DIM]())
    var grad_in_cpu  = TileTensor(gi_buf_cpu, row_major[BATCH, DIM]())

    var go_dev = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    var gi_dev = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    ctx.enqueue_copy(go_dev, go_host)
    var grad_out_gpu = TileTensor(go_dev, row_major[BATCH, DIM]())
    var grad_in_gpu  = TileTensor(gi_dev, row_major[BATCH, DIM]())

    r_cpu.backward["cpu", BATCH](grad_out_cpu, grad_in_cpu)
    r_gpu.backward["gpu", BATCH](grad_out_gpu, grad_in_gpu)

    var gi_host = ctx.enqueue_create_host_buffer[DT](BATCH * DIM)
    ctx.enqueue_copy(gi_host, gi_dev)
    ctx.synchronize()

    var max_bwd: Scalar[DT] = 0.0
    for b in range(BATCH):
        for d in range(DIM):
            var diff = fabs(grad_in_cpu[b, d] - gi_host.unsafe_ptr()[b * DIM + d])
            if diff > max_bwd: max_bwd = diff
    print("backward max-diff = " + String(max_bwd))
    assert_true(max_bwd < TOL, "backward: " + String(max_bwd))

    in_buf_cpu.free()
    out_buf_cpu.free()
    go_buf_cpu.free()
    gi_buf_cpu.free()
    print("  test_gpu_parity_residual_tanh PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 Residual tests (CPU + GPU, Phase 5.5)")
    print("=" * 60)
    test_forward_residual_tanh_cpu()
    test_backward_residual_tanh_cpu_gradcheck()
    test_for_each_param_residual_linear()
    test_gpu_parity_residual_tanh()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
