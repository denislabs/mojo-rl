"""Symlog[DIM] CPU + GPU tests (Block D-1).

Covers:
  - forward: y = sign(x) * log(1 + |x|) on a hand-checkable batch
  - backward: dx = dy / (1 + |x|)
  - GPU parity vs CPU
  - FD gradcheck (composed downstream loss = sum(y))
"""

from std.math import abs as fabs, log
from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.symlog import Symlog
from mojo_rl.nn2.initializer import Kaiming


def _hand_symlog(x: Scalar[DT]) -> Scalar[DT]:
    var ax = x if x >= 0 else -x
    var sgn: Scalar[DT] = 1 if x >= 0 else -1
    return sgn * log(Scalar[DT](1) + ax)


def test_forward_backward_cpu() raises:
    """BATCH=4, DIM=1 — covers positive, negative, zero, large magnitude."""
    comptime BATCH = 4
    comptime DIM = 1
    var sym = Symlog[DIM].make[target="cpu", INIT=Kaiming]()

    var in_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    in_buf[0] = 0.0
    in_buf[1] = 1.0
    in_buf[2] = -1.0
    in_buf[3] = 100.0

    var in_t = TileTensor(in_buf, row_major[BATCH, DIM]())
    var out_t = TileTensor(out_buf, row_major[BATCH, DIM]())
    sym.forward["cpu", BATCH](in_t, out_t)

    for k in range(BATCH):
        var expected = _hand_symlog(in_buf[k])
        assert_true(
            fabs(out_buf[k] - expected) < Scalar[DT](1e-5),
            "forward mismatch",
        )

    # Backward: grad_out = 1.0 everywhere → grad_in = 1 / (1 + |x|)
    var go_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        go_buf[k] = 1.0
        gi_buf[k] = 0.0
    var go_t = TileTensor(go_buf, row_major[BATCH, DIM]())
    var gi_t = TileTensor(gi_buf, row_major[BATCH, DIM]())
    sym.backward["cpu", BATCH](go_t, gi_t)

    for k in range(BATCH):
        var x = in_buf[k]
        var ax = x if x >= 0 else -x
        var expected = Scalar[DT](1.0) / (Scalar[DT](1.0) + ax)
        assert_true(
            fabs(gi_buf[k] - expected) < Scalar[DT](1e-6),
            "backward mismatch",
        )

    in_buf.free()
    out_buf.free()
    go_buf.free()
    gi_buf.free()
    print("  test_forward_backward_cpu PASSED")


def test_gradcheck_fd_cpu() raises:
    """FD gradcheck — sum(y) loss, grad = 1 / (1 + |x|)."""
    comptime BATCH = 3
    comptime DIM = 2
    comptime EPS: Scalar[DT] = 1e-3
    comptime TOL_REL: Scalar[DT] = 1e-3

    var sym = Symlog[DIM].make[target="cpu", INIT=Kaiming]()
    var in_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        in_buf[k] = Scalar[DT](-0.3 + 0.4 * Float64(k))  # mix of signs
    var in_t = TileTensor(in_buf, row_major[BATCH, DIM]())
    var out_t = TileTensor(out_buf, row_major[BATCH, DIM]())

    # Analytical: forward + backward with grad_out=1 → grad_in = 1/(1+|x|)
    sym.forward["cpu", BATCH](in_t, out_t)
    var go_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        go_buf[k] = 1.0
        gi_buf[k] = 0.0
    var go_t = TileTensor(go_buf, row_major[BATCH, DIM]())
    var gi_t = TileTensor(gi_buf, row_major[BATCH, DIM]())
    sym.backward["cpu", BATCH](go_t, gi_t)

    # FD: perturb each input, compare to grad_in[k]
    var max_rel: Scalar[DT] = 0.0
    for k in range(BATCH * DIM):
        var saved = in_buf[k]
        in_buf[k] = saved + EPS
        sym.forward["cpu", BATCH](in_t, out_t)
        var Lp: Scalar[DT] = 0.0
        for j in range(BATCH * DIM):
            Lp += out_buf[j]
        in_buf[k] = saved - EPS
        sym.forward["cpu", BATCH](in_t, out_t)
        var Lm: Scalar[DT] = 0.0
        for j in range(BATCH * DIM):
            Lm += out_buf[j]
        in_buf[k] = saved
        var fd = (Lp - Lm) / (Scalar[DT](2.0) * EPS)
        var an = gi_buf[k]
        var rel = fabs(fd - an) / (fabs(an) + Scalar[DT](1e-6))
        if rel > max_rel:
            max_rel = rel

    print("  Symlog FD gradcheck max_rel = ", max_rel)
    assert_true(max_rel < TOL_REL, "FD gradcheck failed")

    in_buf.free()
    out_buf.free()
    go_buf.free()
    gi_buf.free()
    print("  test_gradcheck_fd_cpu PASSED")


def test_gpu_parity() raises:
    """GPU forward/backward must agree with CPU."""
    comptime BATCH = 4
    comptime DIM = 3
    comptime TOL: Scalar[DT] = 1e-5

    var ctx = DeviceContext()
    var sym_cpu = Symlog[DIM].make[target="cpu", INIT=Kaiming]()
    var sym_gpu = Symlog[DIM].make[target="gpu", INIT=Kaiming](ctx)

    var in_host = ctx.enqueue_create_host_buffer[DT](BATCH * DIM)
    var go_host = ctx.enqueue_create_host_buffer[DT](BATCH * DIM)
    ctx.synchronize()
    for k in range(BATCH * DIM):
        in_host.unsafe_ptr()[k] = Scalar[DT](-1.5 + 0.3 * Float64(k))
        go_host.unsafe_ptr()[k] = Scalar[DT](0.5 + 0.1 * Float64(k))

    # CPU reference
    var in_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var out_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var go_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        in_cpu[k] = in_host.unsafe_ptr()[k]
        go_cpu[k] = go_host.unsafe_ptr()[k]
        gi_cpu[k] = 0.0
    var in_t_cpu  = TileTensor(in_cpu,  row_major[BATCH, DIM]())
    var out_t_cpu = TileTensor(out_cpu, row_major[BATCH, DIM]())
    var go_t_cpu  = TileTensor(go_cpu,  row_major[BATCH, DIM]())
    var gi_t_cpu  = TileTensor(gi_cpu,  row_major[BATCH, DIM]())
    sym_cpu.forward["cpu", BATCH](in_t_cpu, out_t_cpu)
    sym_cpu.backward["cpu", BATCH](go_t_cpu, gi_t_cpu)

    # GPU
    var in_dev  = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    var out_dev = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    var go_dev  = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    var gi_dev  = ctx.enqueue_create_buffer[DT](BATCH * DIM)
    ctx.enqueue_copy(in_dev, in_host)
    ctx.enqueue_copy(go_dev, go_host)
    var in_t_gpu  = TileTensor(in_dev,  row_major[BATCH, DIM]())
    var out_t_gpu = TileTensor(out_dev, row_major[BATCH, DIM]())
    var go_t_gpu  = TileTensor(go_dev,  row_major[BATCH, DIM]())
    var gi_t_gpu  = TileTensor(gi_dev,  row_major[BATCH, DIM]())
    sym_gpu.forward["gpu", BATCH](in_t_gpu, out_t_gpu)
    sym_gpu.backward["gpu", BATCH](go_t_gpu, gi_t_gpu)

    var out_host = ctx.enqueue_create_host_buffer[DT](BATCH * DIM)
    var gi_host  = ctx.enqueue_create_host_buffer[DT](BATCH * DIM)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.enqueue_copy(gi_host, gi_dev)
    ctx.synchronize()

    var max_out: Scalar[DT] = 0.0
    var max_gi: Scalar[DT] = 0.0
    for k in range(BATCH * DIM):
        var d_out = fabs(out_cpu[k] - out_host.unsafe_ptr()[k])
        var d_gi = fabs(gi_cpu[k] - gi_host.unsafe_ptr()[k])
        if d_out > max_out:
            max_out = d_out
        if d_gi > max_gi:
            max_gi = d_gi

    print("  max |y_cpu - y_gpu| = ", max_out)
    print("  max |gi_cpu - gi_gpu| = ", max_gi)
    assert_true(max_out < TOL, "forward GPU parity failed")
    assert_true(max_gi < TOL, "backward GPU parity failed")

    in_cpu.free()
    out_cpu.free()
    go_cpu.free()
    gi_cpu.free()
    print("  test_gpu_parity PASSED")


def main() raises:
    print("=" * 60)
    print("nn2 Symlog tests (Block D-1)")
    print("=" * 60)
    test_forward_backward_cpu()
    test_gradcheck_fd_cpu()
    test_gpu_parity()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
