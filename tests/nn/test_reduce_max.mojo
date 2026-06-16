"""Smoke + CPU↔GPU parity test for ReduceMax[NA].

Validates:
  - forward: out[b, 0] == max_a in[b, a]
  - vjp:     grad_input zero-filled (forward-only op)
  - CPU and GPU produce identical forward outputs

CPU + GPU. GPU only runs when DeviceContext can be created.
"""

from std.memory import alloc
from std.gpu.host import DeviceContext
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.reduce_max import ReduceMax
from mojo_rl.nn.initializer import Zero


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


def test_forward_cpu() raises:
    print("test_forward_cpu ...")
    comptime BATCH = 3
    comptime NA = 4
    comptime N = BATCH * NA

    var rm = ReduceMax[NA].make[target="cpu", INIT=Zero]()

    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    var y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    # Row 0: max at idx 2 (=0.9). Row 1: max at idx 0 (=1.5). Row 2: max at idx 3 (=2.0).
    x[0] = 0.1;  x[1] = 0.5;  x[2] = 0.9;  x[3] = -0.3
    x[4] = 1.5;  x[5] = 0.5;  x[6] = -1.2; x[7] =  0.7
    x[8] = -0.5; x[9] = 0.0;  x[10] = 1.0; x[11] = 2.0

    var xp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](x)
    var yp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](y)
    var x_t = TileTensor(xp, row_major[BATCH, NA]())
    var y_t = TileTensor(yp, row_major[BATCH, 1]())
    rm.forward["cpu", BATCH](x_t, output=y_t)

    assert_true(y[0] == Scalar[DT](0.9), "row 0 max")
    assert_true(y[1] == Scalar[DT](1.5), "row 1 max")
    assert_true(y[2] == Scalar[DT](2.0), "row 2 max")
    print("  ok")


def test_vjp_cpu_zero_fill() raises:
    print("test_vjp_cpu_zero_fill ...")
    comptime BATCH = 2
    comptime NA = 3
    comptime N = BATCH * NA

    var rm = ReduceMax[NA].make[target="cpu", INIT=Zero]()

    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var gi: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
    go[0] = 7.0; go[1] = -3.5
    # Pre-fill gi with non-zero junk to prove vjp overwrites with zero.
    for i in range(N):
        gi[i] = Scalar[DT](99.0)

    var gop = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go)
    var gip = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi)
    var go_t = TileTensor(gop, row_major[BATCH, 1]())
    var gi_t = TileTensor(gip, row_major[BATCH, NA]())
    rm.vjp["cpu", BATCH](go_t, gi_t)

    for i in range(N):
        assert_true(gi[i] == Scalar[DT](0.0), "grad_input must be zero-filled")
    print("  ok")


def test_cpu_gpu_parity() raises:
    print("test_cpu_gpu_parity ...")
    comptime BATCH = 8
    comptime NA = 5
    comptime N = BATCH * NA

    try:
        var ctx = DeviceContext()

        var x_cpu_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](N)
        var y_cpu_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
        var y_gpu_host: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)

        # Deterministic non-trivial fill — distinct max position per row.
        for b in range(BATCH):
            for a in range(NA):
                # Make slot ((b * 3 + 1) % NA) the row max so positions differ.
                var base = Scalar[DT](0.1 * Float64(b) - 0.05 * Float64(a))
                var hit_pos = (b * 3 + 1) % NA
                if a == hit_pos:
                    base = base + Scalar[DT](5.0)
                x_cpu_buf[b * NA + a] = base

        # CPU run.
        var rm_cpu = ReduceMax[NA].make[target="cpu", INIT=Zero]()
        var xp_cpu = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](x_cpu_buf)
        var yp_cpu = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](y_cpu_buf)
        var x_cpu_t = TileTensor(xp_cpu, row_major[BATCH, NA]())
        var y_cpu_t = TileTensor(yp_cpu, row_major[BATCH, 1]())
        rm_cpu.forward["cpu", BATCH](x_cpu_t, output=y_cpu_t)

        # GPU run — H2D, forward, D2H.
        var rm_gpu = ReduceMax[NA].make[target="gpu", INIT=Zero](ctx=ctx)
        var x_dev = ctx.enqueue_create_buffer[DT](N)
        var y_dev = ctx.enqueue_create_buffer[DT](BATCH)
        ctx.enqueue_copy(x_dev, x_cpu_buf)
        var x_dev_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](x_dev.unsafe_ptr())
        var y_dev_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](y_dev.unsafe_ptr())
        var x_gpu_t = TileTensor(x_dev_p, row_major[BATCH, NA]())
        var y_gpu_t = TileTensor(y_dev_p, row_major[BATCH, 1]())
        rm_gpu.forward["gpu", BATCH](x_gpu_t, output=y_gpu_t)
        ctx.enqueue_copy(y_gpu_host, y_dev)
        ctx.synchronize()

        var max_diff: Scalar[DT] = 0.0
        for b in range(BATCH):
            var d = _abs(y_cpu_buf[b] - y_gpu_host[b])
            if d > max_diff:
                max_diff = d
        print("  max |cpu - gpu| =", max_diff)
        assert_true(max_diff == Scalar[DT](0), "CPU/GPU forward mismatch")
        print("  ok")
    except e:
        print("  (skipped — no GPU available:", e, ")")


def main() raises:
    print("=" * 70)
    print("ReduceMax[NA] smoke + CPU/GPU parity")
    print("=" * 70)
    test_forward_cpu()
    test_vjp_cpu_zero_fill()
    test_cpu_gpu_parity()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
