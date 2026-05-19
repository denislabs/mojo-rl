"""Parallel[A, B] CPU + GPU tests — Phase 5.6.

Strategy:
  - Forward correctness: Parallel[Tanh[D], Tanh[D]] — output = [tanh(x) | tanh(x)]
  - Backward gradcheck: same topology, FD vs analytical
  - for_each_param: Parallel[Linear, Linear] produces 4 visits with
    'p0.a.weight'/'p0.a.bias'/'p0.b.weight'/'p0.b.bias' prefixes
  - GPU parity vs CPU
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
from mojo_rl.nn2.combinators import Parallel


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

    def visit[
        L: TensorLayout, OP: MutOrigin, OG: MutOrigin,
    ](
        mut self,
        name: String,
        param: TileTensor[DT, L, OP],
        grad: TileTensor[DT, L, OG],
        n_elems: Int,
        apply_decay: Bool,
    ) raises:
        self.records.append(ParamRecord(name, n_elems, apply_decay))


def test_forward_cpu() raises:
    """Parallel[Tanh[D], Tanh[D]]: output = [tanh(x) | tanh(x)] packed."""
    comptime D = 3
    comptime BATCH = 2
    comptime OUT = 2 * D
    comptime TOL: Scalar[DT] = 1e-5

    var p = Parallel[Tanh[D], Tanh[D]].make["cpu", INIT=Zero]()

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    in_buf[0] =  0.5; in_buf[1] = -1.0; in_buf[2] = 0.0
    in_buf[3] =  1.5; in_buf[4] =  0.2; in_buf[5] = -0.7
    for k in range(BATCH * OUT):
        out_buf[k] = -999.0

    var input  = TileTensor(in_buf,  row_major[BATCH, D]())
    var output = TileTensor(out_buf, row_major[BATCH, OUT]())
    p.forward["cpu", BATCH](input, output)

    for b in range(BATCH):
        for d in range(D):
            var expected = math_tanh(input[b, d])
            # Branch A's output at [b, d]
            assert_true(fabs(output[b, d] - expected) < TOL,
                "A: b=" + String(b) + " d=" + String(d)
                + " expected " + String(expected) + " got " + String(output[b, d]))
            # Branch B's output at [b, D + d]
            assert_true(fabs(output[b, D + d] - expected) < TOL,
                "B: b=" + String(b) + " d=" + String(d)
                + " expected " + String(expected) + " got " + String(output[b, D + d]))

    in_buf.free()
    out_buf.free()
    print("  test_forward_cpu PASSED")


def test_backward_gradcheck_cpu() raises:
    """FD gradcheck on Parallel[Tanh[D], Tanh[D]]."""
    comptime D = 3
    comptime BATCH = 2
    comptime OUT = 2 * D
    comptime EPS_FD: Scalar[DT] = 1e-2
    comptime TOL_REL: Scalar[DT] = 1e-2

    var p = Parallel[Tanh[D], Tanh[D]].make["cpu", INIT=Zero]()

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var go_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var gi_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D)
    in_buf[0] = 0.3; in_buf[1] = -0.5; in_buf[2] = 1.0
    in_buf[3] = -0.8; in_buf[4] = 0.4; in_buf[5] = -0.1
    for k in range(BATCH * OUT):
        go_buf[k] = Scalar[DT](0.1 + Float32(k) * 0.05)
    for k in range(BATCH * D):
        gi_buf[k] = -999.0

    var input    = TileTensor(in_buf,  row_major[BATCH, D]())
    var output   = TileTensor(out_buf, row_major[BATCH, OUT]())
    var grad_out = TileTensor(go_buf,  row_major[BATCH, OUT]())
    var grad_in  = TileTensor(gi_buf,  row_major[BATCH, D]())

    p.forward["cpu", BATCH](input, output)
    p.backward["cpu", BATCH](grad_out, grad_in)

    var max_rel: Scalar[DT] = 0.0
    for bi in range(BATCH):
        for di in range(D):
            in_buf[bi * D + di] += EPS_FD
            p.forward["cpu", BATCH](input, output)
            var L_plus: Scalar[DT] = 0.0
            for b2 in range(BATCH):
                for j2 in range(OUT):
                    L_plus += go_buf[b2 * OUT + j2] * output[b2, j2]
            in_buf[bi * D + di] -= Scalar[DT](2.0) * EPS_FD
            p.forward["cpu", BATCH](input, output)
            var L_minus: Scalar[DT] = 0.0
            for b2 in range(BATCH):
                for j2 in range(OUT):
                    L_minus += go_buf[b2 * OUT + j2] * output[b2, j2]
            in_buf[bi * D + di] += EPS_FD
            var fd = (L_plus - L_minus) / (Scalar[DT](2.0) * EPS_FD)
            var an = grad_in[bi, di]
            var denom = fabs(an) + Scalar[DT](1e-6)
            var rel = fabs(fd - an) / denom
            if rel > max_rel: max_rel = rel

    print("Parallel[Tanh, Tanh] gradcheck max-rel-err = " + String(max_rel))
    assert_true(max_rel < TOL_REL,
        "gradcheck failed: " + String(max_rel))

    in_buf.free()
    out_buf.free()
    go_buf.free()
    gi_buf.free()
    print("  test_backward_gradcheck_cpu PASSED")


def test_for_each_param() raises:
    """Parallel[Linear[D, A], Linear[D, B]] yields 4 params with
    a./b. prefixes."""
    var p = Parallel[Linear[3, 5], Linear[3, 7]].make["cpu", INIT=Zero]()
    var v = RecordVisitor()
    p.for_each_param["cpu"](String("p0"), v)
    assert_equal(len(v.records), 4)
    assert_equal(v.records[0].name, String("p0.a.weight"))
    assert_equal(v.records[0].n_elems, 15)
    assert_true(v.records[0].apply_decay)
    assert_equal(v.records[1].name, String("p0.a.bias"))
    assert_equal(v.records[1].n_elems, 5)
    assert_true(not v.records[1].apply_decay)
    assert_equal(v.records[2].name, String("p0.b.weight"))
    assert_equal(v.records[2].n_elems, 21)
    assert_true(v.records[2].apply_decay)
    assert_equal(v.records[3].name, String("p0.b.bias"))
    assert_equal(v.records[3].n_elems, 7)
    assert_true(not v.records[3].apply_decay)
    print("  test_for_each_param PASSED (4 visits, a./b. prefixes)")


def test_gpu_parity() raises:
    """Forward + backward parity over Parallel[Tanh[D], Tanh[D]]."""
    comptime D = 4
    comptime BATCH = 3
    comptime OUT = 2 * D
    comptime TOL: Scalar[DT] = 1e-5

    var ctx = DeviceContext()
    var p_cpu = Parallel[Tanh[D], Tanh[D]].make["cpu", INIT=Zero]()
    var p_gpu = Parallel[Tanh[D], Tanh[D]].make["gpu", INIT=Zero](ctx)

    var in_host = ctx.enqueue_create_host_buffer[DT](BATCH * D)
    ctx.synchronize()
    for k in range(BATCH * D):
        in_host.unsafe_ptr()[k] = Scalar[DT](Float32(k) * 0.2 - 1.0)

    var in_buf_cpu:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D)
    var out_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    for k in range(BATCH * D):
        in_buf_cpu[k] = in_host.unsafe_ptr()[k]
    var input_cpu  = TileTensor(in_buf_cpu, row_major[BATCH, D]())
    var output_cpu = TileTensor(out_buf_cpu, row_major[BATCH, OUT]())

    var in_dev  = ctx.enqueue_create_buffer[DT](BATCH * D)
    var out_dev = ctx.enqueue_create_buffer[DT](BATCH * OUT)
    ctx.enqueue_copy(in_dev, in_host)
    var input_gpu  = TileTensor(in_dev,  row_major[BATCH, D]())
    var output_gpu = TileTensor(out_dev, row_major[BATCH, OUT]())

    p_cpu.forward["cpu", BATCH](input_cpu, output_cpu)
    p_gpu.forward["gpu", BATCH](input_gpu, output_gpu)

    var out_host = ctx.enqueue_create_host_buffer[DT](BATCH * OUT)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()
    var max_fwd: Scalar[DT] = 0.0
    for b in range(BATCH):
        for j in range(OUT):
            var diff = fabs(output_cpu[b, j] - out_host.unsafe_ptr()[b * OUT + j])
            if diff > max_fwd: max_fwd = diff
    print("forward max-diff = " + String(max_fwd))
    assert_true(max_fwd < TOL, "forward parity: " + String(max_fwd))

    # Backward.
    var go_host = ctx.enqueue_create_host_buffer[DT](BATCH * OUT)
    ctx.synchronize()
    for k in range(BATCH * OUT):
        go_host.unsafe_ptr()[k] = Scalar[DT](0.1 + Float32(k) * 0.05)

    var go_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var gi_buf_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D)
    for k in range(BATCH * OUT):
        go_buf_cpu[k] = go_host.unsafe_ptr()[k]
    for k in range(BATCH * D):
        gi_buf_cpu[k] = -999.0
    var grad_out_cpu = TileTensor(go_buf_cpu, row_major[BATCH, OUT]())
    var grad_in_cpu  = TileTensor(gi_buf_cpu, row_major[BATCH, D]())

    var go_dev = ctx.enqueue_create_buffer[DT](BATCH * OUT)
    var gi_dev = ctx.enqueue_create_buffer[DT](BATCH * D)
    ctx.enqueue_copy(go_dev, go_host)
    var grad_out_gpu = TileTensor(go_dev, row_major[BATCH, OUT]())
    var grad_in_gpu  = TileTensor(gi_dev, row_major[BATCH, D]())

    p_cpu.backward["cpu", BATCH](grad_out_cpu, grad_in_cpu)
    p_gpu.backward["gpu", BATCH](grad_out_gpu, grad_in_gpu)

    var gi_host = ctx.enqueue_create_host_buffer[DT](BATCH * D)
    ctx.enqueue_copy(gi_host, gi_dev)
    ctx.synchronize()
    var max_bwd: Scalar[DT] = 0.0
    for b in range(BATCH):
        for d in range(D):
            var diff = fabs(grad_in_cpu[b, d] - gi_host.unsafe_ptr()[b * D + d])
            if diff > max_bwd: max_bwd = diff
    print("backward max-diff = " + String(max_bwd))
    assert_true(max_bwd < TOL, "backward parity: " + String(max_bwd))

    in_buf_cpu.free()
    out_buf_cpu.free()
    go_buf_cpu.free()
    gi_buf_cpu.free()
    print("  test_gpu_parity PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 Parallel tests (CPU + GPU, Phase 5.6)")
    print("=" * 60)
    test_forward_cpu()
    test_backward_gradcheck_cpu()
    test_for_each_param()
    test_gpu_parity()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
